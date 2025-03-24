use winterfell::{
    math::{fields::f128::BaseElement, FieldElement},
    Proof,
};
use winter_utils::DeserializationError;
use sha2::{Sha256, Digest};
use serde::{Deserialize, Serialize};
use candid::CandidType;

mod prover;
mod verifier;

use crate::prover::{WrappedBaseElement, Model, Dataset, sigmoid_approx, prove_work};
use crate::verifier::verify_work;

// Wrapped Proof for serialisation
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct WrappedProof {
    pub proof_data: Vec<u8>, // Store proof as bytes
}

impl WrappedProof {
    pub fn wrap(proof: Proof) -> Self {
        WrappedProof {
            proof_data: proof.to_bytes(),
        }
    }

    pub fn unwrap(self) -> Result<Proof, DeserializationError> {
        winterfell::Proof::from_bytes(&self.proof_data)
    }
}

const MODULUS: u128 = 340282366920938463463374557953744961537;

#[ic_cdk::update]
fn initialise_model(
    len_sample: usize,
    hidden_size: usize,
) -> Model {
    // Helper function to generate deterministic values
    fn deterministic_value(seed: &[u8]) -> BaseElement {
        let mut hasher = Sha256::new();
        hasher.update(seed);
        let hash_bytes = hasher.finalize();
        let num = u64::from_le_bytes(hash_bytes[..8].try_into().unwrap());
        BaseElement::new(num as u128 % 20_000 + MODULUS - 10_000) // Convert to BaseElement
    }

    let weights_input_hidden = (0..(len_sample * hidden_size))
        .map(|i| deterministic_value(&i.to_le_bytes()))
        .collect();

    let weights_hidden_output = (0..hidden_size)
        .map(|i| deterministic_value(&i.to_le_bytes()))
        .collect();

    let bias_hidden = vec![BaseElement::ZERO; hidden_size];

    let bias_output = BaseElement::ZERO;

    let wrapped_weights_input_hidden = WrappedBaseElement::wrap_vec(weights_input_hidden);
    let wrapped_weights_hidden_output = WrappedBaseElement::wrap_vec(weights_hidden_output);
    let wrapped_bias_hidden = WrappedBaseElement::wrap_vec(bias_hidden);
    let wrapped_bias_output = WrappedBaseElement::wrap(bias_output);
    
    Model {
        weights_input_hidden: wrapped_weights_input_hidden,
        weights_hidden_output: wrapped_weights_hidden_output,
        bias_hidden: wrapped_bias_hidden,
        bias_output: wrapped_bias_output,
    }
}

#[ic_cdk::query]
fn train_and_prove(
    num_epochs: usize,
    learning_rate: WrappedBaseElement,
    model: Model,
    dataset: Dataset,
) -> (Model, WrappedBaseElement, WrappedProof) {
    let (updated_model, dataset_hash, proof) =
        prove_work(num_epochs, learning_rate.unwrap(), model, dataset);
    (updated_model, WrappedBaseElement::wrap(dataset_hash), WrappedProof::wrap(proof))
}

#[ic_cdk::update]
fn verify(
    initial_model: Model,
    updated_model: Model,
    datahash: WrappedBaseElement,
    proof: WrappedProof
) -> bool {
    verify_work(initial_model, updated_model, datahash.unwrap(), proof.unwrap().expect("_"))
}

#[ic_cdk::update]
fn predict(model: Model, input: Vec<WrappedBaseElement>) -> WrappedBaseElement {
    let len_sample = input.len();
    let hidden_size = model.bias_hidden.len();

    // Calculate hidden layer activations
    let mut hidden_activations = vec![BaseElement::ZERO; hidden_size];
    for h in 0..hidden_size {
        let mut sum = WrappedBaseElement::unwrap_vec(&model.bias_hidden)[h];
        for i in 0..len_sample {
            sum += input[i].unwrap() * WrappedBaseElement::unwrap_vec(&model.weights_input_hidden)[h * len_sample + i];
        }
        hidden_activations[h] = sigmoid_approx(sum);
    }

    // Calculate output
    let mut output_sum = model.bias_output.unwrap();
    for h in 0..hidden_size {
        output_sum += hidden_activations[h] * WrappedBaseElement::unwrap_vec(&model.weights_hidden_output)[h];
    }

    WrappedBaseElement::wrap(sigmoid_approx(output_sum)) // Final output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialise_model_works() {
        let model = initialise_model(
            10, // sample length
            9   // hidden layer size
        );
        assert_eq!(model.weights_input_hidden.len(), 10 * 9);
    }
}

ic_cdk::export_candid!();