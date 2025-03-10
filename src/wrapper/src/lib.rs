use winterfell::{
    math::{fields::f128::BaseElement, FieldElement, StarkField},
    Proof,
};
use winter_utils::DeserializationError;
use sha2::{Sha256, Digest};
use serde::{Deserialize, Serialize};
use candid::CandidType;

mod prover;
mod verifier;

use crate::prover::{Model, Dataset, sigmoid_approx, prove_work};
use crate::verifier::verify_work;

/// Wrapped BaseElement for serialisation
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct WrappedBaseElement(pub u128);

impl WrappedBaseElement {
    pub fn wrap(element: BaseElement) -> Self {
        WrappedBaseElement(element.as_int())
    }

    pub fn unwrap(self) -> BaseElement {
        BaseElement::new(self.0)
    }
}

/// Wrapped Proof for serialisation
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
        BaseElement::from(num % 20_000 - 10_000) // Convert to BaseElement
    }

    let weights_input_hidden = (0..(len_sample * hidden_size))
        .map(|i| deterministic_value(&i.to_le_bytes()))
        .collect();

    let weights_hidden_output = (0..hidden_size)
        .map(|i| deterministic_value(&i.to_le_bytes()))
        .collect();

    let bias_hidden = vec![BaseElement::ZERO; hidden_size];

    let bias_output = BaseElement::ZERO;

    Model {
        weights_input_hidden,
        weights_hidden_output,
        bias_hidden,
        bias_output,
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
        let mut sum = model.bias_hidden[h];
        for i in 0..len_sample {
            sum += input[i].unwrap() * model.weights_input_hidden[h * len_sample + i];
        }
        hidden_activations[h] = sigmoid_approx(sum);
    }

    // Calculate output
    let mut output_sum = model.bias_output;
    for h in 0..hidden_size {
        output_sum += hidden_activations[h] * model.weights_hidden_output[h];
    }

    WrappedBaseElement::wrap(sigmoid_approx(output_sum)) // Final output
}

ic_cdk::export_candid!();