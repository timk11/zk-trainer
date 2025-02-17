// This is a first draft, based on the Winterfell documentation and
// yet to be proofread or debugged.

use winterfell::{
    crypto::{hashers::Blake3_256, DefaultRandomCoin},
    math::{fields::f128::BaseElement, FieldElement, ToElements},
    matrix::ColMatrix,
    Air, AirContext, Assertion, EvaluationFrame, DefaultConstraintEvaluator,
    DefaultTraceLde, FieldExtension, HashFunction, ProofOptions, Prover,
    StarkDomain,Trace, TraceInfo,
    TraceTable, TracePolyTable, TransitionConstraintDegree,
};
use sha2::{Sha256, Digest};
use std::convert::TryInto;


// TRACE

pub struct Model {
    pub weights_input_hidden: Vec<BaseElement>,
    pub weights_hidden_output: Vec<BaseElement>,
    pub bias_hidden: Vec<BaseElement>,
    pub bias_output: BaseElement,
};

pub struct Dataset {
    pub input_matrix: Vec<Vec<BaseElement>>, // Each row is an input vector
    pub expected_output: Vec<BaseElement>,
};

// A simple sigmoid approximation using Taylor series
fn sigmoid_approx(x: BaseElement) -> BaseElement {
    BaseElement::new(0.5) + BaseElement::new(0.25) * x - (x.pow(3) / BaseElement::new(48.0))
};

fn is_power_of_2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
};

fn update_hash(
    current_hash: BaseElement,
    inputs: &[BaseElement],
) -> BaseElement {
    let mut hasher = Sha256::new();
    hasher.update(current_hash.as_int().to_le_bytes());
    for value in inputs.iter() {
        hasher.update(value.as_int().to_le_bytes());
    }
    let hash_bytes = hasher.finalize();
    let hash_value = u64::from_le_bytes(hash_bytes[..8].try_into().unwrap());
    BaseElement::from(hash_value)
};

fn generate_nn_trace(
    num_epochs: usize,
    learning_rate: BaseElement,
    mut model: Model,
    dataset: Dataset,
) -> TraceTable<BaseElement> {
    let num_samples = dataset.input_matrix.len();
    let len_sample = dataset.input_matrix[0].len();
    let hidden_size = model. weights_input_hidden.len();
    assert!(is_power_of_2(num_epochs), "Number of epochs must be a power of 2");
    assert!(is_power_of_2(num_samples + 1), "Number of samples + 1 must be a power of 2");
    let num_columns = (len_sample + 2)(hidden_size + 1) + 5; // Includes flag, len_sample, hidden_size, learning_rate, sample, expected, hash, weights and biases
    assert!(num_columns <= 255, "Too many columns");
    let mut trace = TraceTable::new(num_columns, num_epochs * num_samples + 1);
    let mut dataset_hash = BaseElement::ZERO;
    let mut w_b_hash = BaseElement::ZERO;

    for epoch in 0..num_epochs {
        dataset_hash = BaseElement::ZERO;
        for sample_idx in 0..num_samples {
            let input = &dataset.input_matrix[sample_idx];
            let expected = dataset.expected_output[sample_idx];
            w_b_hash = update_hash(w_b_hash, [
                model.weights_input_hidden.clone(),
                model.weights_hidden_output.clone(),
                model.bias_hidden.clone(),
                vec![model.bias_output]
            ].concat());
            trace.update_row(epoch * num_samples + sample_idx, [
                vec![BaseElement::new(0)],
                vec![BaseElement::new(len_sample)],
                vec![BaseElement::new(hidden_size)],
                vec![learning_rate],
                vec![dataset_hash], // Current dataset hash (previous row)
                vec![w_b_hash], // Hash of weights and biases (current row)
                input.clone(), // Dataset sample
                vec![expected], // Expected output
                model.weights_input_hidden.clone(),
                model.weights_hidden_output.clone(),
                model.bias_hidden.clone(),
                vec![model.bias_output]
            ].concat());
            dataset_hash = update_hash(dataset_hash, [input, vec![expected]].concat());
            let mut hidden_activations = vec![BaseElement::ZERO; hidden_size];
            let mut hidden_errors = vec![BaseElement::ZERO; hidden_size];
            let mut hidden_gradients = vec![BaseElement::ZERO; hidden_size];
            let mut output = model.bias_output;

            for j in 0..hidden_size {
                let mut activation = model.bias_hidden[j];
                for i in 0..input.len() {
                    activation += input[i] * model.weights_input_hidden[j];
                }
                hidden_activations[j] = activation * (BaseElement::new(0.5) + BaseElement::new(0.25) * activation - (activation.pow(3) / BaseElement::new(48.0)));
                output += hidden_activations[j] * model.weights_hidden_output[j];
            }

            let error = output - expected;
            for j in 0..hidden_size {
                hidden_errors[j] = error;
                hidden_gradients[j] = error * hidden_activations[j] * (BaseElement::one() - hidden_activations[j]);
                model.weights_input_hidden[j] += learning_rate * hidden_gradients[j];
                model.weights_hidden_output[j] += learning_rate * hidden_gradients[j];
                model.bias_hidden[j] += learning_rate * hidden_gradients[j];
            }
            model.bias_output += learning_rate * error;

        }

        // Append separator row (n+1) with zero sample data
        let zero_sample = vec![BaseElement::ZERO; dataset.inputs[0].len()];
        trace.update_row((epoch + 1) * num_samples - 1, [
            vec![BaseElement::new(1)],
            vec![BaseElement::new(len_sample)],
            vec![BaseElement::new(hidden_size)],
            vec![learning_rate],
            vec![dataset_hash],
            vec![w_b_hash],
            zero_sample,
            vec![BaseElement::new(0)],
            model.weights_input_hidden.clone(),
            model.weights_hidden_output.clone(),
            model.bias_hidden.clone(),
            vec![model.bias_output]
        ].concat());
    }

    trace
};

// AIR
// Public inputs for our computation will consist of the starting value and the end result.
pub struct PublicInputs {
    start: BaseElement,                // hash of initial weights and biases
    updated: BaseElement,               // hash of final weights and biases
    datahash: BaseElement,             // hash of dataset
};

// We need to describe how public inputs can be converted to field elements.
impl ToElements<BaseElement> for PublicInputs {
    fn to_elements(&self) -> Vec<BaseElement> {
        vec![self.start, self.updated, self.datahash]
    };
};

// For a specific instance of our computation, we'll keep track of the public inputs and
// the computation's context which we'll build in the constructor. The context is used
// internally by the Winterfell prover/verifier when interpreting this AIR.
pub struct WorkAir {
    context: AirContext<BaseElement>,
    start: BaseElement,
    result: Vec<BaseElement>,
};

impl Air for WorkAir {
    // First, we'll specify which finite field to use for our computation, and also how
    // the public inputs must look like.
    type BaseField = BaseElement;
    type PublicInputs = PublicInputs;


    // Here, we'll construct a new instance of our computation which is defined by 3 parameters:
    // starting value, number of steps, and the end result. Another way to think about it is
    // that an instance of our computation is a specific invocation of the do_work() function.
    fn new(trace_info: TraceInfo, pub_inputs: PublicInputs, options: ProofOptions) -> Self {
        // Our computation requires a single transition constraint. The constraint itself
        // is defined in the evaluate_transition() method below, but here we need to specify
        // the expected degree of the constraint.
        let degrees = vec![TransitionConstraintDegree::new(2)];

        // We also need to specify the exact number of assertions we will place against the
        // execution trace. This number must be the same as the number of items in a vector
        // returned from the get_assertions() method below.
        let num_assertions = 2;

        WorkAir {
            context: AirContext::new(trace_info, degrees, num_assertions, options),
            start: pub_inputs.start,
            result: pub_inputs.result,
        }
    };


    // In this method we'll define our transition constraints; a computation is considered to
    // be valid, if for all valid state transitions, transition constraints evaluate to all
    // zeros, and for any invalid transition, at least one constraint evaluates to a non-zero
    // value. The `frame` parameter will contain current and next states of the computation.
    // TODO - how to handle this if start and result are vectors?
    fn evaluate_transition<E: FieldElement + From<Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E], // as vector?
    ) {
        // First, we'll read the current state, and use it to compute the expected next state.

        // Then, we'll subtract the expected next state from the actual next state; this will
        // evaluate to zero if and only if the expected and actual states are the same.

        let current_state = frame.current()[0];
        let flag = current_state[0];
        let len_sample = current_state[1];
        let hidden_size = current_state[2];
        let learning_rate = current_state[3];
        let mut dataset_hash =  current_state[4]; // Current dataset hash
        let mut w_b_hash =  current_state.get[5]; // Hash of weights and biases
        let input = current_state.get(6..(6 + len_sample)); // Input feature
        let expected =  current_state.get[6 + len_sample]; // Expected output
        let mut weights_ih = current_state.get((7 + len_sample)..(7 + len_sample + len_sample * hidden_size)); // Weight from input to hidden layer
        let mut weights_ho = current_state.get((7 + len_sample + len_sample * hidden_size)..(7 + len_sample + len_sample * hidden_size + hidden_size)); // Bias for hidden layer
        let mut biases_h = current_state.get((7 + len_sample + len_sample * hidden_size + hidden_size)..(7 + len_sample + len_sample * hidden_size + 2 * hidden_size)); // Weight from hidden to output layer
        let mut bias_o = current_state[7 + len_sample + len_sample * hidden_size + 2 * hidden_size]; // Bias for output layer

        let mut hidden_activations = vec![BaseElement::ZERO; hidden_size];
        let mut hidden_errors = vec![BaseElement::ZERO; hidden_size];
        let mut hidden_gradients = vec![BaseElement::ZERO; hidden_size];
        let mut output = bias_o;

        if (flag == 1) { dataset_hash = BaseElement::ZERO; };
        dataset_hash = update_hash(dataset_hash, [input, vec![expected]].concat());

        if (flag == 0) {
            for j in 0..hidden_size {
                let mut activation = biases_h[j];
                for i in 0..len_sample {
                    activation += input[i] * weights_ih[j];
                }
                hidden_activations[j] = activation * (BaseElement::new(0.5) + BaseElement::new(0.25) * activation - (activation.pow(3) / BaseElement::new(48.0)));
                output += hidden_activations[j] * weights_ho[j];
            }

            let error = output - expected;
            for j in 0..hidden_size {
                hidden_errors[j] = error;
                hidden_gradients[j] = error * hidden_activations[j] * (BaseElement::ONE - hidden_activations[j]);
                weights_ih[j] += learning_rate * hidden_gradients[j];
                weights_ho[j] += learning_rate * hidden_gradients[j];
                biases_h[j] += learning_rate * hidden_gradients[j];
            };
            bias_o += learning_rate * error;
            w_b_hash = update_hash(w_b_hash, [
                model.weights_input_hidden.clone(),
                model.weights_hidden_output.clone(),
                model.bias_hidden.clone(),
                vec![model.bias_output]
            ].concat());
        };

        let next_state = [
            vec![dataset_hash], // Current dataset hash (previous row)
            vec![w_b_hash] // Hash of weights and bias (current row)
        ].concat());

        for i in 0..2 {
            result[i] = frame.next()[0][i + 4] - next_state[i];
        };
    };


    // Here, we'll define a set of assertions about the execution trace which must be satisfied
    // for the computation to be valid. Essentially, this ties computation's execution trace
    // to the public inputs.
    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        // for our computation to be valid, value in column 0 at step 0 must be equal to the
        // starting value, and at the last step it must be equal to the result.
        let last_step = self.trace_length() - 1;
        vec![
            Assertion::single(4, 0, self.start), // hash of initial weights & biases
            Assertion::single(4, last_step, self.updated), // hash of final weights & biases
            Assertion::single(5, last_step, self.datahash), // dataset hash
        ]
    };


    // This is just boilerplate which is used by the Winterfell prover/verifier to retrieve
    // the context of the computation.
    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    };
};

// PROVER
// We'll use BLAKE3 as the hash function during proof generation.
type Blake3 = Blake3_256<BaseElement>;

// Our prover needs to hold STARK protocol parameters which are specified via ProofOptions
// struct.
struct WorkProver {
    options: ProofOptions,
};

impl WorkProver {
    pub fn new(options: ProofOptions) -> Self {
        Self { options }
    };
};

// When implementing the Prover trait we set the `Air` associated type to the AIR of the
// computation we defined previously, and set the `Trace` associated type to `TraceTable`
// struct as we don't need to define a custom trace for our computation. For other
// associated types, we'll use default implementation provided by Winterfell.
impl Prover for WorkProver {
    type BaseField = BaseElement;
    type Air = WorkAir;
    type Trace = TraceTable<BaseElement>;
    type HashFn = Blake3;
    type RandomCoin = DefaultRandomCoin<Blake3>;
    type TraceLde<E: FieldElement<BaseField = BaseElement>> = DefaultTraceLde<E, Blake3>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = BaseElement>> =
        DefaultConstraintEvaluator<'a, WorkAir, E>;
    type ConstraintCommitment<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintCommitment<E, H, Self::VC>;

    // Our public inputs consist of the first and last value in the execution trace.
    fn get_pub_inputs(&self, trace: &Self::Trace) -> PublicInputs {
        let last_step = trace.length() - 1;
        PublicInputs {
            start: trace.get(4, 0),
            updated: trace.get(4, last_step),
            datahash: trace.get(5, last_step),
        }
    };

    // We'll use the default trace low-degree extension.
    fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Self::BaseField>,
        domain: &StarkDomain<Self::BaseField>,
    ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
        DefaultTraceLde::new(trace_info, main_trace, domain)
    };

    // We'll use the default constraint evaluator to evaluate AIR constraints.
    fn new_evaluator<'a, E: FieldElement<BaseField = BaseElement>>(
        &self,
        air: &'a WorkAir,
        aux_rand_elements: Option<Self::AuxRandElements<E>>,
        composition_coefficients: winterfell::ConstraintCompositionCoefficients<E>,
    ) -> Self::ConstraintEvaluator<'a, E> {
        DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
    };

    // We'll use the default constraint commitment.
    fn build_constraint_commitment<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        composition_poly_trace: CompositionPolyTrace<E>,
        num_constraint_composition_columns: usize,
        domain: &StarkDomain<Self::BaseField>,
        partition_options: PartitionOptions,
    ) -> (Self::ConstraintCommitment<E>, CompositionPoly<E>) {
        DefaultConstraintCommitment::new(
            composition_poly_trace,
            num_constraint_composition_columns,
            domain,
            partition_options,
        )
    };

    fn options(&self) -> &ProofOptions {
        &self.options
    };
};
pub fn prove_work(
    num_epochs: usize,
    learning_rate: BaseElement,
    mut model: Model,
    dataset: Dataset,
) -> (Model, BaseElement, Proof) {
    // Build the execution trace and get the result from the last step.
    let trace = generate_nn_trace(num_epochs, learning_rate, model, dataset);

    let final_row = &trace[n - 1];
    let weights_input_hidden = final_row.get((7 + len_sample)..(7 + len_sample + len_sample * hidden_size));
    let weights_hidden_output = final_row.get((7 + len_sample + len_sample * hidden_size)..(7 + len_sample + len_sample * hidden_size + hidden_size));
    let bias_hidden = final_row.get((7 + len_sample + len_sample * hidden_size + hidden_size)..(7 + len_sample + len_sample * hidden_size + 2 * hidden_size));
    let bias_output = final_row.get(7 + len_sample + len_sample * hidden_size + 2 * hidden_size);
    let updated_model = Model {
        weights_input_hidden: weights_input_hidden,
        weights_hidden_output: weights_hidden_output,
        bias_hidden: bias_hidden,
        bias_output:  bias_output
    };

    let dataset_hash = trace.get(5, n - 1);

    // Define proof options; these will be enough for ~96-bit security level.
    let options = ProofOptions::new(
        32, // number of queries
        8,  // blowup factor
        0,  // grinding factor
        FieldExtension::None,
        8,   // FRI folding factor
        127, // FRI remainder max degree
    );

    // Instantiate the prover and generate the proof.
    let prover = WorkProver::new(options);
    let proof = prover.prove(trace).unwrap();

    (updated_model, dataset_hash, proof)
};
