use winterfell::{
    crypto::{hashers::{Blake3_256, Rp64_256}, DefaultRandomCoin, ElementHasher, MerkleTree},
    math::{fields::f64::BaseElement, FieldElement, StarkField},
    verify,
    AcceptableOptions, Proof,
};
use sha2::{Sha256, Digest};
use std::convert::TryInto;
use crate::prover::{Model, PublicInputs, WorkAir, WrappedBaseElement};

type Blake3 = Blake3_256<BaseElement>;
type VC = MerkleTree<Blake3>;

fn update_hash(
    current_hash: BaseElement,
    inputs: &[BaseElement],
) -> BaseElement {
    let hash_result = Rp64_256::hash_elements(
        &[current_hash].iter().chain(inputs.iter()).cloned().collect::<Vec<_>>()
    );
    hash_result.as_elements()[0] // Extracting the first field element from the hash digest
}

pub(crate) fn verify_work(initial_model: Model, updated_model: Model, datahash: BaseElement, proof: Proof) -> bool {
    // Convert initial and updated models into hashes
    let start = update_hash(BaseElement::ZERO, &[
        WrappedBaseElement::unwrap_vec(&initial_model.weights_input_hidden),
        WrappedBaseElement::unwrap_vec(&initial_model.weights_hidden_output),
        WrappedBaseElement::unwrap_vec(&initial_model.bias_hidden),
        vec![initial_model.bias_output.unwrap()]
    ].concat());
    let updated = update_hash(BaseElement::ZERO, &[
        WrappedBaseElement::unwrap_vec(&updated_model.weights_input_hidden),
        WrappedBaseElement::unwrap_vec(&updated_model.weights_hidden_output),
        WrappedBaseElement::unwrap_vec(&updated_model.bias_hidden),
        vec![updated_model.bias_output.unwrap()]
    ].concat());

    // The verifier will accept proofs with parameters which guarantee 95 bits or more of
    // conjectured security
    let min_opts = AcceptableOptions::MinConjecturedSecurity(95);

    // The number of steps and options are encoded in the proof itself, so we don't need to
    // pass them explicitly to the verifier.
    let pub_inputs = PublicInputs { start, updated, datahash };
    let outcome: bool = match verify::<WorkAir, Blake3, DefaultRandomCoin<Blake3>, VC>(proof, pub_inputs, &min_opts) {
        Ok(_) => true,
        Err(_) => false,
    };
    outcome
}