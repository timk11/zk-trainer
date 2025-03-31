use winterfell::{
    crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree},
    math::{fields::f128::BaseElement, FieldElement, StarkField, ToElements},
    matrix::ColMatrix,
    Air, AirContext, Assertion, AuxRandElements, BatchingMethod,
    CompositionPoly, CompositionPolyTrace,
    DefaultConstraintCommitment, DefaultConstraintEvaluator, DefaultTraceLde,
    EvaluationFrame, FieldExtension, PartitionOptions,
    Proof, ProofOptions, Prover, StarkDomain,
    Trace, TraceInfo, TraceTable, TracePolyTable,
    TransitionConstraintDegree,
};
use sha2::{Sha256, Digest};
use std::convert::TryInto;
use serde::{Serialize, Deserialize};
use candid::CandidType;



// TRACE

// Wrapped BaseElement for serialisation
#[derive(CandidType, Serialize, Deserialize, Clone, Copy, Debug)]
pub struct WrappedBaseElement(pub u128);

impl WrappedBaseElement {
    pub fn wrap(element: BaseElement) -> Self {
        WrappedBaseElement(element.as_int())
    }

    pub fn unwrap(self) -> BaseElement {
        BaseElement::new(self.0)
    }

    pub fn wrap_vec(vec: Vec<BaseElement>) -> Vec<Self> {
        vec.into_iter().map(WrappedBaseElement::wrap).collect()
    }

    pub fn unwrap_vec(vec: &[Self]) -> Vec<BaseElement> {
        vec.iter().map(|x| x.clone().unwrap()).collect()
    }
}

impl From<WrappedBaseElement> for usize {
    fn from(value: WrappedBaseElement) -> Self {
        value.0 as usize
    }
}

#[derive(Deserialize, CandidType, Clone)]
pub struct Model {
    pub weights_input_hidden: Vec<WrappedBaseElement>,
    pub weights_hidden_output: Vec<WrappedBaseElement>,
    pub bias_hidden: Vec<WrappedBaseElement>,
    pub bias_output: WrappedBaseElement,
}

#[derive(Deserialize, CandidType, Clone)]
pub struct Dataset {
    pub input_matrix: Vec<Vec<WrappedBaseElement>>, // Each row is an input vector
    pub expected_output: Vec<WrappedBaseElement>,
}

pub const E4: BaseElement = BaseElement::new(10_000);
const E8: BaseElement = BaseElement::new(100_000_000);

// Taylor series approximation of sigmoid: sigmoid(x) ≈ 0.5 + x/4 - x^3/48
pub fn sigmoid_approx(x: BaseElement) -> BaseElement {
    BaseElement::new(5000) + x / BaseElement::new(4) - (x.exp(3) / BaseElement::new(4_800_000_000))
}

fn is_power_of_2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

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
    BaseElement::new(hash_value.into())
}

fn generate_nn_trace(
    num_epochs: usize,
    learning_rate: BaseElement,
    model: &mut Model,
    dataset: &Dataset,
) -> TraceTable<BaseElement> {
    let num_samples = dataset.input_matrix.len();
    let len_sample = dataset.input_matrix[0].len();
    let hidden_size = model.weights_hidden_output.len();
    assert!(is_power_of_2(num_epochs), "Number of epochs must be a power of 2");
    assert!(is_power_of_2(num_samples + 1), "Number of samples + 1 must be a power of 2");
    let num_columns = (len_sample + 2) * (hidden_size + 1) + 6; // Includes flag, len_sample, hidden_size, learning_rate, sample, expected, dataset_hash, wb_hash, weights and biases
    println!("num_samples: {num_samples}, len_sample: {len_sample}, hidden_size: {hidden_size}, num_columns: {num_columns}");
    assert!(num_columns <= 255, "Too many columns");
    let mut trace = TraceTable::new(num_columns, num_epochs * (num_samples + 1));
    let mut dataset_hash: BaseElement;
    let mut w_b_hash = BaseElement::ZERO;

    for epoch in 0..num_epochs {
        dataset_hash = BaseElement::ZERO;
        for sample_idx in 0..num_samples {
            let input = WrappedBaseElement::unwrap_vec(&dataset.input_matrix[sample_idx]);
            let expected = WrappedBaseElement::unwrap_vec(&dataset.expected_output)[sample_idx];
            w_b_hash = update_hash(w_b_hash, &[
                WrappedBaseElement::unwrap_vec(&model.weights_input_hidden),
                WrappedBaseElement::unwrap_vec(&model.weights_hidden_output),
                WrappedBaseElement::unwrap_vec(&model.bias_hidden),
                vec![model.bias_output.clone().unwrap()]
            ].concat());
            trace.update_row(epoch * (num_samples + 1) + sample_idx, &[
                vec![BaseElement::new(0)],
                vec![BaseElement::new(len_sample.try_into().unwrap())],
                vec![BaseElement::new(hidden_size.try_into().unwrap())],
                vec![learning_rate],
                vec![dataset_hash], // Current dataset hash (previous row)
                vec![w_b_hash], // Hash of weights and biases (current row)
                input.clone(), // Dataset sample
                vec![expected], // Expected output
                WrappedBaseElement::unwrap_vec(&model.weights_input_hidden),
                WrappedBaseElement::unwrap_vec(&model.weights_hidden_output),
                WrappedBaseElement::unwrap_vec(&model.bias_hidden),
                vec![model.bias_output.clone().unwrap()]
            ].concat());
            dataset_hash = update_hash(dataset_hash, &[input.clone(), vec![expected]].concat());
            let mut hidden_activations = vec![BaseElement::ZERO; hidden_size];
            let mut hidden_errors = vec![BaseElement::ZERO; hidden_size];
            let mut hidden_gradients = vec![BaseElement::ZERO; hidden_size];
            let mut output = model.bias_output.unwrap();

            for j in 0..hidden_size {
                let mut activation = WrappedBaseElement::unwrap_vec(&model.bias_hidden)[j];
                for i in 0..input.len() {
                    activation += input[i] * WrappedBaseElement::unwrap_vec(&model.weights_input_hidden)[j * input.len() + i] / E4;
                }
                hidden_activations[j] = sigmoid_approx(activation);
                output += hidden_activations[j] * WrappedBaseElement::unwrap_vec(&model.weights_hidden_output)[j] / E4;
            }

            let error = output - expected;
            let output_gradient = error * sigmoid_approx(output) * (E4 - sigmoid_approx(output)) / E8;

            for j in 0..hidden_size {
                hidden_errors[j] = output_gradient * WrappedBaseElement::unwrap_vec(&model.weights_hidden_output)[j] / E4;
                hidden_gradients[j] = hidden_errors[j] * hidden_activations[j] * (E4 - hidden_activations[j]) / E8;

                let new_weight_hidden_output = WrappedBaseElement::unwrap_vec(&model.weights_hidden_output)[j]
                    + learning_rate * output_gradient * hidden_activations[j] / E8;
                model.weights_hidden_output[j] = WrappedBaseElement::wrap(new_weight_hidden_output);

                for i in 0..input.len() {
                    let new_weight_input_hidden = WrappedBaseElement::unwrap_vec(&model.weights_input_hidden)[j * input.len() + i]
                        + learning_rate * hidden_gradients[j] * input[i] / E8;
                    model.weights_input_hidden[j * input.len() + i] = WrappedBaseElement::wrap(new_weight_input_hidden);
                }

                let new_bias_hidden = WrappedBaseElement::unwrap_vec(&model.bias_hidden)[j] + learning_rate * hidden_gradients[j] / E4;
                model.bias_hidden[j] = WrappedBaseElement::wrap(new_bias_hidden);
            }
            let new_bias_output = WrappedBaseElement::unwrap(model.bias_output) + learning_rate * output_gradient / E4;
            model.bias_output = WrappedBaseElement::wrap(new_bias_output);
        }

        // Append separator row (n+1) with zero sample data
        let zero_sample = vec![BaseElement::ZERO; dataset.input_matrix[0].len()];
        w_b_hash = update_hash(w_b_hash, &[
            WrappedBaseElement::unwrap_vec(&model.weights_input_hidden),
            WrappedBaseElement::unwrap_vec(&model.weights_hidden_output),
            WrappedBaseElement::unwrap_vec(&model.bias_hidden),
            vec![model.bias_output.clone().unwrap()]
        ].concat());
        trace.update_row((epoch + 1) * (num_samples + 1) - 1, &[
            vec![BaseElement::new(1)],
            vec![BaseElement::new(len_sample.try_into().unwrap())],
            vec![BaseElement::new(hidden_size.try_into().unwrap())],
            vec![learning_rate],
            vec![dataset_hash],
            vec![w_b_hash],
            zero_sample,
            vec![BaseElement::new(0)],
            WrappedBaseElement::unwrap_vec(&model.weights_input_hidden),
            WrappedBaseElement::unwrap_vec(&model.weights_hidden_output),
            WrappedBaseElement::unwrap_vec(&model.bias_hidden),
            vec![model.bias_output.clone().unwrap()]
        ].concat());
    }

    trace
}

// AIR
// Public inputs for our computation will consist of the starting value and the end result.
pub struct PublicInputs {
    pub start: BaseElement,           // hash of initial weights and biases
    pub updated: BaseElement,         // hash of final weights and biases
    pub datahash: BaseElement,        // hash of dataset
}

// We need to describe how public inputs can be converted to field elements.
impl ToElements<BaseElement> for PublicInputs {
    fn to_elements(&self) -> Vec<BaseElement> {
        vec![self.start, self.updated, self.datahash]
    }
}

// For a specific instance of our computation, we'll keep track of the public inputs and
// the computation's context which we'll build in the constructor. The context is used
// internally by the Winterfell prover/verifier when interpreting this AIR.
pub struct WorkAir {
    context: AirContext<BaseElement>,
    start: BaseElement,
    result: BaseElement,
    datahash: BaseElement,
}

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
        let degrees = vec![TransitionConstraintDegree::new(3)];

        // We also need to specify the exact number of assertions we will place against the
        // execution trace. This number must be the same as the number of items in a vector
        // returned from the get_assertions() method below.
        let num_assertions = 3;

        WorkAir {
            context: AirContext::new(trace_info, degrees, num_assertions, options),
            start: pub_inputs.start,
            result: pub_inputs.updated,
            datahash: pub_inputs.datahash,
        }
    }


    // In this method we'll define our transition constraints; a computation is considered to
    // be valid, if for all valid state transitions, transition constraints evaluate to all
    // zeros, and for any invalid transition, at least one constraint evaluates to a non-zero
    // value. The `frame` parameter will contain current and next states of the computation.
    // TODO - how to handle this if start and result are vectors?
    fn evaluate_transition<E: FieldElement<BaseField = BaseElement> + From<Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E], // as vector?
    ) {
        // First, we'll read the current state, and use it to compute the expected next state.

        // Then, we'll subtract the expected next state from the actual next state; this will
        // evaluate to zero if and only if the expected and actual states are the same.

        let current_state = frame.current();
        let flag = current_state[0].base_element(0).as_int();
        let len_sample = usize::from(WrappedBaseElement::wrap(current_state[1].base_element(0)));
        let hidden_size = usize::from(WrappedBaseElement::wrap(current_state[2].base_element(0)));
        let learning_rate = current_state[3];
        let mut dataset_hash = current_state[4]; // Current dataset hash
        let mut w_b_hash = current_state[5]; // Hash of weights and biases
        let input = current_state.get(6..(6usize + len_sample)); // Input feature
        let expected = current_state[6usize + len_sample]; // Expected output
        let mut weights_ih = current_state.get((7usize + len_sample)..(7usize + len_sample + len_sample * hidden_size))
            .map(|slice| slice.iter().map(|&e| e.base_element(0)).collect::<Vec<_>>())
            .unwrap_or_default(); // Weight from input to hidden layer
        let mut weights_ho = current_state.get((7usize + len_sample + len_sample * hidden_size)..(7usize + len_sample + len_sample * hidden_size + hidden_size))
            .map(|slice| slice.iter().map(|&e| e.base_element(0)).collect::<Vec<_>>())
            .unwrap_or_default(); // Bias for hidden layer
        let mut biases_h = current_state.get((7usize + len_sample + len_sample * hidden_size + hidden_size)..(7usize + len_sample + len_sample * hidden_size + 2usize * hidden_size))
            .map(|slice| slice.iter().map(|&e| e.base_element(0)).collect::<Vec<_>>())
            .unwrap_or_default(); // Weight from hidden to output layer
        let mut bias_o = current_state[7usize + len_sample + len_sample * hidden_size + 2usize * hidden_size]; // Bias for output layer

        let mut hidden_activations = vec![BaseElement::ZERO; hidden_size];
        let mut hidden_errors = vec![BaseElement::ZERO; hidden_size];
        let mut hidden_gradients = vec![BaseElement::ZERO; hidden_size];
        let mut output = bias_o;

        if flag == 1 {
            dataset_hash = BaseElement::ZERO.into();
        } else {
            dataset_hash = update_hash(
                dataset_hash.base_element(0),
                &[
                    input.map(|slice| slice.iter().map(|&e| e.base_element(0)).collect::<Vec<_>>())
                        .unwrap_or_default(),
                    vec![expected.base_element(0)],
                ].concat(),
            ).into();
        };

        if flag == 0 {
            for j in 0..hidden_size {
                let mut activation = biases_h[j];
                for i in 0..len_sample {
                    activation += (input.unwrap()[i] * weights_ih[j * len_sample + i].into() / E4.into()).base_element(0);
                }
                hidden_activations[j] = sigmoid_approx(activation.base_element(0));
                output += (hidden_activations[j] * weights_ho[j].base_element(0) / E4).into();
            }

            let error = output - expected;
            let output_gradient = error.base_element(0) * sigmoid_approx(output.base_element(0)) * (E4 - sigmoid_approx(output.base_element(0))) / E8;
            for j in 0..hidden_size {
                hidden_errors[j] = output_gradient * weights_ho[j].base_element(0) / E4;
                hidden_gradients[j] = hidden_errors[j] * hidden_activations[j] * (E4 - hidden_activations[j]) / E8;

                weights_ho[j] += (learning_rate * output_gradient.into() * hidden_activations[j].into() / E8.into()).base_element(0);

                for i in 0..len_sample {
                    weights_ih[j * len_sample + i] += (learning_rate * hidden_gradients[j].into() * input.unwrap()[i] / E8.into()).base_element(0);
                }

                biases_h[j] += (learning_rate * hidden_gradients[j].into() / E4.into()).base_element(0);
            };
            bias_o += learning_rate * output_gradient.into() / E4.into();
        };
        
        w_b_hash = update_hash(w_b_hash.base_element(0), &[
            weights_ih.clone(),
            weights_ho.clone(),
            biases_h.clone(),
            vec![bias_o.base_element(0)]
        ].concat()).into();

        result[0] = (frame.next()[4] - dataset_hash) + (frame.next()[5] - w_b_hash);
        // TODO - Remove printing when debugging is complete
        println!("flag: {}, frame.next()[4]: {}, dataset_hash: {}, frame.next()[5]: {}, w_b_hash: {}, frame.next()[-1]: {}, bias_o: {}, result: {}", flag, frame.next()[4], dataset_hash, frame.next()[5], w_b_hash, frame.next()[7usize + len_sample + len_sample * hidden_size + 2usize * hidden_size], bias_o, result[0]);
    }


    // Here, we'll define a set of assertions about the execution trace which must be satisfied
    // for the computation to be valid. Essentially, this ties computation's execution trace
    // to the public inputs.
    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        // for our computation to be valid, value in column 0 at step 0 must be equal to the
        // starting value, and at the last step it must be equal to the result.
        let last_step = self.trace_length() - 1;
        vec![
            Assertion::single(4, 0, self.start), // hash of initial weights & biases
            Assertion::single(4, last_step, self.result), // hash of final weights & biases
            Assertion::single(5, last_step, self.datahash), // dataset hash
        ]
    }


    // This is just boilerplate which is used by the Winterfell prover/verifier to retrieve
    // the context of the computation.
    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }
}

// PROVER
// We'll use BLAKE3 as the hash function during proof generation.
type Blake3 = Blake3_256<BaseElement>;

// Our prover needs to hold STARK protocol parameters which are specified via ProofOptions
// struct.
pub struct WorkProver {
    options: ProofOptions,
}

impl WorkProver {
    pub fn new(options: ProofOptions) -> Self {
        Self { options }
    }
}

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
    type VC = MerkleTree<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = BaseElement>> = DefaultTraceLde<E, Blake3, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = BaseElement>> =
        DefaultConstraintEvaluator<'a, WorkAir, E>;
    type ConstraintCommitment<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;

    // Our public inputs consist of the first and last value in the execution trace.
    fn get_pub_inputs(&self, trace: &Self::Trace) -> PublicInputs {
        let last_step = trace.length() - 1;
        PublicInputs {
            start: trace.get(4, 0),
            updated: trace.get(4, last_step),
            datahash: trace.get(5, last_step),
        }
    }

    // We'll use the default trace low-degree extension.
    fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Self::BaseField>,
        domain: &StarkDomain<Self::BaseField>,
        partition_options: PartitionOptions,
    ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
        DefaultTraceLde::new(trace_info, main_trace, domain, partition_options)
    }

    // We'll use the default constraint evaluator to evaluate AIR constraints.
    fn new_evaluator<'a, E: FieldElement<BaseField = BaseElement>>(
        &self,
        air: &'a WorkAir,
        aux_rand_elements: Option<AuxRandElements<E>>,
        composition_coefficients: winterfell::ConstraintCompositionCoefficients<E>,
    ) -> Self::ConstraintEvaluator<'a, E> {
        DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
    }

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
    }

    fn options(&self) -> &ProofOptions {
        &self.options
    }
}
pub(crate) fn prove_work(
    num_epochs: usize,
    learning_rate: BaseElement,
    mut model: Model,
    dataset: Dataset,
) -> (Model, BaseElement, Proof) {
    // Build the execution trace and get the result from the last step.
    let trace = generate_nn_trace(num_epochs, learning_rate, &mut model, &dataset);
    let n = num_epochs * (dataset.input_matrix.len() + 1); //  number of rows
    let len_sample = dataset.input_matrix[0].len();
    let hidden_size = model.weights_input_hidden.len();
    let num_columns = (len_sample + 2) * (hidden_size + 1) + 5;
    let mut final_row = vec![BaseElement::ZERO; num_columns];
    trace.read_row_into(n - 1, &mut final_row);
    //let final_row = &trace[n - 1];
    let len_sample = dataset.input_matrix[0].len();
    let hidden_size = model.weights_input_hidden.len();

    let weights_input_hidden = final_row.get((7 + len_sample)..(7 + len_sample + len_sample * hidden_size));
    let weights_hidden_output = final_row.get((7 + len_sample + len_sample * hidden_size)..(7 + len_sample + len_sample * hidden_size + hidden_size));
    let bias_hidden = final_row.get((7 + len_sample + len_sample * hidden_size + hidden_size)..(7 + len_sample + len_sample * hidden_size + 2 * hidden_size));
    let bias_output = final_row.get(7 + len_sample + len_sample * hidden_size + 2 * hidden_size);
    let updated_model = Model {
        weights_input_hidden: WrappedBaseElement::wrap_vec(weights_input_hidden.map(|s| s.to_vec()).unwrap_or_else(Vec::new)),
        weights_hidden_output: WrappedBaseElement::wrap_vec(weights_hidden_output.map(|s| s.to_vec()).unwrap_or_else(Vec::new)),
        bias_hidden: WrappedBaseElement::wrap_vec(bias_hidden.map(|s| s.to_vec()).unwrap_or_else(Vec::new)),
        bias_output:  WrappedBaseElement::wrap(bias_output.copied().unwrap_or_else(BaseElement::default))
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
        BatchingMethod::Linear,
        BatchingMethod::Linear,
    );

    // Instantiate the prover and generate the proof.
    let prover = WorkProver::new(options);
    let proof = prover.prove(trace).unwrap();

    (updated_model, dataset_hash, proof)
}
