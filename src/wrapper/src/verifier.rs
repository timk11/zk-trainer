use winterfell::{
    crypto::{hashers::Blake3_256, DefaultRandomCoin},
    math::fields::f128::BaseElement,
    verify,
    AcceptableOptions, AirContext, Proof,
};
use sha2::{Sha256, Digest};
use std::convert::TryInto;

type Blake3 = Blake3_256<BaseElement>;

pub struct Model {
    pub weights_input_hidden: Vec<BaseElement>,
    pub weights_hidden_output: Vec<BaseElement>,
    pub bias_hidden: Vec<BaseElement>,
    pub bias_output: BaseElement,
}

pub struct PublicInputs {
    start: BaseElement,         // hash of initial weights and biases
    updated: BaseElement,       // hash of final weights and biases
    datahash: BaseElement,      // hash of dataset
}

pub struct WorkAir {
    context: AirContext<BaseElement>,
    start: BaseElement,
    result: Vec<BaseElement>,
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
    BaseElement::from(hash_value)
}

pub(crate) fn verify_work(initial_model: Model, updated_model: Model, datahash: BaseElement, proof: Proof) -> bool {
    // Convert initial and updated models into hashes
    let start = update_hash(BaseElement::ZERO, [
        initial_model.weights_input_hidden.clone(),
        initial_model.weights_hidden_output.clone(),
        initial_model.bias_hidden.clone(),
        vec![initial_model.bias_output]
    ].concat());
    let updated = update_hash(BaseElement::ZERO, [
        initial_model.weights_input_hidden.clone(),
        initial_model.weights_hidden_output.clone(),
        initial_model.bias_hidden.clone(),
        vec![initial_model.bias_output]
    ].concat());

    // The verifier will accept proofs with parameters which guarantee 95 bits or more of
    // conjectured security
    let min_opts = AcceptableOptions::MinConjecturedSecurity(95);

    // The number of steps and options are encoded in the proof itself, so we don't need to
    // pass them explicitly to the verifier.
    let pub_inputs = PublicInputs { start, updated, datahash };
    let outcome: bool = match verify::<WorkAir, Blake3, DefaultRandomCoin<Blake3>>(proof, pub_inputs, &min_opts) {
        Ok(_) => true,
        Err(_) => false,
    };
    outcome
}