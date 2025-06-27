use winterfell::{
    crypto::{hashers::{Blake3_256, Rp64_256},
        DefaultRandomCoin, ElementHasher, Hasher, MerkleTree},
    math::{fields::f64::BaseElement, FieldElement, StarkField, ToElements},
    matrix::ColMatrix,
    Air, AirContext, Assertion, AuxRandElements, BatchingMethod,
    CompositionPoly, CompositionPolyTrace,
    DefaultConstraintCommitment, DefaultConstraintEvaluator, DefaultTraceLde,
    EvaluationFrame, FieldExtension, PartitionOptions,
    Proof, ProofOptions, Prover, StarkDomain,
    Trace, TraceInfo, TraceTable, TracePolyTable,
    TransitionConstraintDegree,
};
//use sha2::{Sha256, Digest};
use std::convert::TryInto;
use serde::{Serialize, Deserialize};
use candid::CandidType;



// TRACE

// Wrapped BaseElement for serialisation
#[derive(CandidType, Serialize, Deserialize, Clone, Copy, Debug)]
pub struct WrappedBaseElement(pub u64);

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
    pub weights: Vec<WrappedBaseElement>,
    pub bias: WrappedBaseElement,
}

#[derive(Deserialize, CandidType, Clone)]
pub struct Dataset {
    pub input_matrix: Vec<Vec<WrappedBaseElement>>, // Each row is an input vector
    pub expected_output: Vec<WrappedBaseElement>,
}

pub const E4: BaseElement = BaseElement::new(10_000);
const E8: BaseElement = BaseElement::new(100_000_000);

/*
// Taylor series approximation of sigmoid: sigmoid(x) ≈ 0.5 + x/4 - x^3/48
pub fn sigmoid_approx(x: BaseElement) -> BaseElement {
    BaseElement::new(5000) + x / BaseElement::new(4) - (x.exp(3) / BaseElement::new(4_800_000_000))
}
*/

fn is_power_of_2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

fn update_hash(
    current_hash: BaseElement,
    inputs: &[BaseElement],
) -> BaseElement {
    let digest = Rp64_256::hash_elements(
        &[current_hash].iter().chain(inputs.iter()).cloned().collect::<Vec<_>>()
    );
    digest.as_elements()[0] // Extracting the first field element from the hash digest
}

fn row_hash(inputs: &[BaseElement]) -> BaseElement {
    let digest = Rp64_256::hash_elements(inputs);
    digest.as_elements()[0]
}

fn update_hash_E<E: FieldElement<BaseField = BaseElement>>(
    current_hash: E,
    inputs: &[E],
) -> E {
    let digest = Rp64_256::hash_elements(
        &[current_hash].iter().chain(inputs.iter()).cloned().collect::<Vec<_>>()
    );
    digest.as_elements()[0].into() // Extracting the first field element from the hash digest
}

fn row_hash_E<E: FieldElement<BaseField = BaseElement>>(inputs: &[E]) -> E {
    let digest = Rp64_256::hash_elements(inputs);
    digest.as_elements()[0].into()
}

fn generate_nn_trace(
    num_epochs: usize,
    learning_rate: BaseElement,
    model: &mut Model,
    dataset: &Dataset,
) -> TraceTable<BaseElement> {
    let num_samples = dataset.input_matrix.len();
    let len_sample = dataset.input_matrix[0].len();
    assert!(is_power_of_2(num_epochs), "Number of epochs must be a power of 2");
    assert!(is_power_of_2(num_samples + 1), "Number of samples + 1 must be a power of 2");
    assert_eq!(len_sample, model.weights.len(), "Number of model weights must match number of features");
    let num_columns = len_sample * 2 + 9; // Includes flag, len_sample, learning_rate, dataset_hash, wb_hash, sample, expected, weights and bias
    println!("num_samples: {num_samples}, len_sample: {len_sample}, num_columns: {num_columns}");
    assert!(num_columns <= 255, "Too many columns");
    let mut trace = TraceTable::new(num_columns, num_epochs * (num_samples + 1));
    let mut dataset_hash: BaseElement;
    let mut w_b_hash: BaseElement;
    let mut output: BaseElement;
    let mut error: BaseElement;

    for epoch in 0..num_epochs {
        dataset_hash = BaseElement::ZERO;
        output = BaseElement::ZERO;
        error = BaseElement::ZERO;
        for sample_idx in 0..num_samples {
            let input = WrappedBaseElement::unwrap_vec(&dataset.input_matrix[sample_idx]);
            let expected = WrappedBaseElement::unwrap_vec(&dataset.expected_output)[sample_idx];
            w_b_hash = row_hash(&[
                WrappedBaseElement::unwrap_vec(&model.weights),
                vec![model.bias.clone().unwrap()]
            ].concat());
            trace.update_row(epoch * (num_samples + 1) + sample_idx, &[
                vec![BaseElement::new(1)],
                vec![BaseElement::new(len_sample.try_into().unwrap())],
                vec![learning_rate],
                vec![dataset_hash], // Current dataset hash (previous row)
                vec![w_b_hash], // Hash of weights and biases (current row)
                vec![output],
                vec![error],
                input.clone(), // Dataset sample
                vec![expected], // Expected output
                WrappedBaseElement::unwrap_vec(&model.weights),
                vec![model.bias.clone().unwrap()]
            ].concat());
            dataset_hash = update_hash(dataset_hash, &[input.clone(), vec![expected]].concat());
            
            output = model.bias.unwrap();
            for i in 0..input.len() {
                output += input[i] * WrappedBaseElement::unwrap_vec(&model.weights)[i] / E4;
            }
    
            // Compute gradient and update weights
            error = output - expected;
            let gradient = error;
    
            for i in 0..input.len() {
                let new_weight = WrappedBaseElement::unwrap_vec(&model.weights)[i]
                    - learning_rate * gradient * input[i] / E8;
                model.weights[i] = WrappedBaseElement::wrap(new_weight);
            }
    
            let new_bias = WrappedBaseElement::unwrap(model.bias)
                - learning_rate * gradient / E4;
            model.bias = WrappedBaseElement::wrap(new_bias);
        }

        // Append separator row (n+1) with zero sample data
        let zero_sample = vec![BaseElement::ZERO; dataset.input_matrix[0].len()];
        w_b_hash = row_hash(&[
            WrappedBaseElement::unwrap_vec(&model.weights),
            vec![model.bias.clone().unwrap()]
        ].concat());
        trace.update_row((epoch + 1) * (num_samples + 1) - 1, &[
            vec![BaseElement::new(0)],
            vec![BaseElement::new(len_sample.try_into().unwrap())],
            vec![learning_rate],
            vec![dataset_hash],
            vec![w_b_hash],
            vec![output],
            vec![error],
            zero_sample,
            vec![BaseElement::new(0)],
            WrappedBaseElement::unwrap_vec(&model.weights),
            vec![model.bias.clone().unwrap()]
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
    // starting model, final model, and the (private) dataset.
    fn new(trace_info: TraceInfo, pub_inputs: PublicInputs, options: ProofOptions) -> Self {
        // Our computation requires transition constraints. The constraints themselves
        // are defined in the evaluate_transition() method below, but here we need to specify
        // the expected degree of each constraint.
        let degrees = vec![
            TransitionConstraintDegree::new(7),
            TransitionConstraintDegree::new(12),
            TransitionConstraintDegree::new(3),
            TransitionConstraintDegree::new(3)
        ];

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
    fn evaluate_transition<E: FieldElement<BaseField = BaseElement> + From<Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E], // as vector?
    ) {
        // First, we'll read the current state, and use it to compute the expected next state.
        // Then, we'll subtract the expected next state (as hashes) from the actual next state;
        // this will evaluate to zero if and only if the expected and actual states are the same.

        let current = frame.current();
        let flag = current[0];
        let len_sample = usize::from(WrappedBaseElement::wrap(current[1].base_element(0)));
        let learning_rate = current[2];
        let mut dataset_hash = current[3]; // Current dataset hash
        // current[4] is hash of weights and biases from previous row
        // current[5] is output from previous row
        // current[6] is error from previous row
        let input = current.get(7..(7usize + len_sample)); // Input features
        let expected = current[7usize + len_sample]; // Expected output
        let mut weights = current.get((8usize + len_sample)..(8usize + len_sample * 2usize))
            .map(|slice| slice.iter().map(|&e| e).collect::<Vec<_>>())
            .unwrap_or_default(); // Weights
        let mut bias = current[8usize + len_sample * 2usize]; // Bias

        let mut output = bias;

        dataset_hash = update_hash_E(
            dataset_hash,
            &[
                input.map(|slice| slice.iter().map(|&e| e).collect::<Vec<_>>())
                    .unwrap_or_default(),
                vec![expected],
            ].concat(),
        ) * flag;

        for i in 0..len_sample {
            output += input.unwrap()[i] * weights[i] / E4.into();
        }

        output = output * flag;
        let error = output - expected;
        let gradient = error;
        for i in 0..len_sample {
            weights[i] -= (learning_rate * gradient * input.unwrap()[i] * flag / E8.into());
        }
        bias -= learning_rate * gradient * flag / E4.into();
        
        let w_b_hash = row_hash_E(&[
            weights.clone(),
            vec![bias]
        ].concat());

        result[0] = frame.next()[3] - dataset_hash;
        result[1] = frame.next()[4] - w_b_hash;
        result[2] = frame.next()[5] - output;
        result[3] = frame.next()[6] - error;
        // TODO - Remove printing when debugging is complete
        println!("--flag: {}, frame.next()[3]: {}, dataset_hash: {}, frame.next()[4]: {}, w_b_hash: {}, output: {}, error: {}, expected: {}, frame.next()[-1]: {}, bias: {}, result: {}, {}, {}, {}", flag, frame.next()[3], dataset_hash, frame.next()[4], w_b_hash, output, error, expected, frame.next()[8usize + len_sample * 2usize], bias, result[0], result[1], result[2], result[3]);
    }


    // Here, we'll define a set of assertions about the execution trace which must be satisfied
    // for the computation to be valid. Essentially, this ties computation's execution trace
    // to the public inputs.
    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        // For our computation to be valid, value in column 0 at step 0 must be equal to the
        // hash of initial weights & biases, and similarly at the last step.
        let last_step = self.trace_length() - 1;
        vec![
            Assertion::single(3, 0, self.start), // hash of initial weights & biases
            Assertion::single(3, last_step, self.result), // hash of final weights & biases
            Assertion::single(4, last_step, self.datahash), // dataset hash
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

    // Our public inputs consist of the hashes of the initial model, the final model and the dataset.
    fn get_pub_inputs(&self, trace: &Self::Trace) -> PublicInputs {
        let last_step = trace.length() - 1;
        PublicInputs {
            start: trace.get(3, 0),
            updated: trace.get(3, last_step),
            datahash: trace.get(4, last_step),
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
    let num_columns = len_sample * 2 + 7;
    let mut final_row = vec![BaseElement::ZERO; num_columns];
    trace.read_row_into(n - 1, &mut final_row);
    //let final_row = &trace[n - 1];
    let len_sample = dataset.input_matrix[0].len();

    let weights = final_row.get((6 + len_sample)..(6 + len_sample * 2));
    let bias = final_row.get(6 + len_sample * 2);
    let updated_model = Model {
        weights: WrappedBaseElement::wrap_vec(weights.map(|s| s.to_vec()).unwrap_or_else(Vec::new)),
        bias:  WrappedBaseElement::wrap(bias.copied().unwrap_or_else(BaseElement::default))
    };

    let dataset_hash = trace.get(4, n - 1);

    // Define proof options; these will be enough for ~96-bit security level.
    let options = ProofOptions::new(
        32, // number of queries
        32,  // blowup factor
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
