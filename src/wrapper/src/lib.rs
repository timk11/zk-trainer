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

fn convert_dataset(
    input_matrix: Vec<Vec<i128>>, 
    expected: Vec<i128>
) -> (Dataset, Vec<(f64, f64)>) {
    assert!(
        input_matrix.len() == expected.len(),
        "The number of samples must equal the length of the expected output"
    );

    let mut standardised_data = Vec::new();
    let mut standardisation_params = Vec::new();

    // Define the new range [100, 9900]
    let new_min = 100.0;
    let new_max = 9900.0;

    // Standardise each row
    for row in &input_matrix {
        let min = *row.iter().min().unwrap() as f64;
        let max = *row.iter().max().unwrap() as f64;
        let range = (max - min).max(1.0); // Prevent division by zero

        let standardised_row: Vec<WrappedBaseElement> = row
            .iter()
            .map(|&x| {
                let standardised_value = (((x as f64 - min) / range) * (new_max - new_min) + new_min)
                    .round() as u128;
                WrappedBaseElement::wrap(BaseElement::new(standardised_value))
            })
            .collect();

        standardised_data.push(standardised_row);
        standardisation_params.push((min, max));
    }

    // Standardise the expected output using the same approach
    let min = *expected.iter().min().unwrap() as f64;
    let max = *expected.iter().max().unwrap() as f64;
    let range = (max - min).max(1.0); // Prevent division by zero

    let standardised_expected: Vec<WrappedBaseElement> = expected
        .iter()
        .map(|&x| {
            let standardised_value = (((x as f64 - min) / range) * (new_max - new_min) + new_min)
                .round() as u128;
            WrappedBaseElement::wrap(BaseElement::new(standardised_value))
        })
        .collect();

    standardisation_params.push((min, max));


    (Dataset {
        input_matrix: standardised_data,
        expected_output: standardised_expected,
    }, standardisation_params)
}

#[cfg(test)]
mod tests {
    use super::*;

    /*
    let sample_data: Vec<Vec<i128>> = vec![
        vec![23, 84, 0, 7, 73, 1, 0, 3],
        vec![19, 64, 0, 8, 59, 1, 2, 4],
        vec![24, 98, 1, 7, 91, 1, 2, 4],
        vec![29, 89, 1, 8, 98, 1, 1, 4],
        vec![19, 92, 1, 6, 65, 1, 3, 4],
        vec![19, 88, 1, 8, 89, 1, 3, 3],
        vec![29, 84, 1, 7, 68, 1, 1, 2],
        vec![25, 78, 1, 6, 50, 1, 1, 2],
        vec![17, 94, 0, 6, 80, 1, 0, 1],
        vec![23, 98, 1, 8, 71, 1, 0, 5],
        vec![17, 80, 0, 8, 88, 0, 4, 4],
        vec![17, 97, 1, 6, 87, 1, 2, 2],
        vec![21, 83, 1, 8, 97, 1, 2, 4],
        vec![9, 82, 1, 8, 72, 1, 2, 3],
        vec![10, 78, 1, 8, 74, 1, 1, 4]
    ];
    let sample_expected: Vec<i128> = vec![
        67, 61, 74, 71, 70, 71, 67, 66, 69, 72, 68, 71, 70, 66, 65
    ];
    let verification_data: Vec<i128> = vec![14, 60, 1, 10, 65, 1, 0, 3]; // expected output = 60
    */

    #[test]
    fn test_initialise_model() {
        let model = initialise_model(
            10, // sample length
            9   // hidden layer size
        );
        assert_eq!(model.weights_input_hidden.len(), 10 * 9);
        assert_eq!(model.weights_hidden_output.len(), 9);
        assert_eq!(model.bias_hidden.len(), 9);
    }

    #[test]
    fn test_train_and_prove() {
        let sample_data: Vec<Vec<i128>> = vec![
            vec![23, 84, 0, 7, 73, 1, 0, 3],
            vec![19, 64, 0, 8, 59, 1, 2, 4],
            vec![24, 98, 1, 7, 91, 1, 2, 4],
            vec![29, 89, 1, 8, 98, 1, 1, 4],
            vec![19, 92, 1, 6, 65, 1, 3, 4],
            vec![19, 88, 1, 8, 89, 1, 3, 3],
            vec![29, 84, 1, 7, 68, 1, 1, 2],/*
            vec![25, 78, 1, 6, 50, 1, 1, 2],
            vec![17, 94, 0, 6, 80, 1, 0, 1],
            vec![23, 98, 1, 8, 71, 1, 0, 5],
            vec![17, 80, 0, 8, 88, 0, 4, 4],
            vec![17, 97, 1, 6, 87, 1, 2, 2],
            vec![21, 83, 1, 8, 97, 1, 2, 4],
            vec![9, 82, 1, 8, 72, 1, 2, 3],
            vec![10, 78, 1, 8, 74, 1, 1, 4]*/
        ];
        let sample_expected: Vec<i128> = vec![
            67, 61, 74, 71, 70, 71, 67, //66, 69, 72, 68, 71, 70, 66, 65
        ];
        let num_epochs: usize = 8;
        let learning_rate = WrappedBaseElement(500);
        let model = initialise_model(8, 12);
        let dataset = convert_dataset(sample_data, sample_expected).0;
        let result = train_and_prove(num_epochs, learning_rate, model, dataset);
    }

    #[test]
    fn test_verify() {
        //
    }

    #[test]
    fn test_predict() {
        //
    }
}

ic_cdk::export_candid!();