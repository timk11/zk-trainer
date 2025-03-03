use winterfell::{
    math::fields::f128::BaseElement,
    Proof,
};
use sha2::Sha256;

mod prover;
mod verifier;

use crate::prover::prove_work;
use crate::verifier::verify_work;

struct Model {
    pub weights_input_hidden: Vec<BaseElement>,
    pub weights_hidden_output: Vec<BaseElement>,
    pub bias_hidden: Vec<BaseElement>,
    pub bias_output: BaseElement,
}

struct Dataset {
    pub input_matrix: Vec<Vec<BaseElement>>, // Each row is an input vector
    pub expected_output: Vec<BaseElement>,
}

struct PublicInputs {
    start: BaseElement,         // hash of initial weights and biases
    updated: BaseElement,       // hash of final weights and biases
    datahash: BaseElement,      // hash of dataset
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
        BaseElement::from(num) // Convert to BaseElement
    }

    let weights_input_hidden = (0..(len_sample * hidden_size))
        .map(|i| deterministic_value(&i.to_le_bytes()))
        .collect();

    let weights_hidden_output = (0..hidden_size)
        .map(|i| deterministic_value(&i.to_le_bytes()))
        .collect();

    let bias_hidden = (0..hidden_size)
        .map(|i| deterministic_value(&i.to_le_bytes()))
        .collect();

    let bias_output = deterministic_value(&[0u8]);

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
    learning_rate: BaseElement,
    mut model: prover::Model,
    dataset: prover::Dataset,
) -> (prover::Model, BaseElement, Proof) {
    prove_work(num_epochs, learning_rate, model, dataset)
}

#[ic_cdk::update]
fn verify(
    initial_model: verifier::Model,
    updated_model: verifier::Model,
    datahash: BaseElement,
    proof: Proof
) -> bool {
    verify_work(initial_model, updated_model, datahash, proof)
}

// Taylor series approximation of sigmoid: sigmoid(x) ≈ 0.5 + x/4 - x^3/48
fn sigmoid_approx(x: BaseElement) -> BaseElement {
    let x2 = x * x;
    let x3 = x2 * x;
    BaseElement::from(0.5) + x / BaseElement::from(4.0) - x3 / BaseElement::from(48.0)
}

#[ic_cdk::update]
fn predict(model: &Model, input: &[BaseElement]) -> BaseElement {
    let len_sample = input.len();
    let hidden_size = model.bias_hidden.len();

    // Calculate hidden layer activations
    let mut hidden_activations = vec![BaseElement::from(0.0); hidden_size];
    for h in 0..hidden_size {
        let mut sum = model.bias_hidden[h];
        for i in 0..len_sample {
            sum += input[i] * model.weights_input_hidden[h * len_sample + i];
        }
        hidden_activations[h] = sigmoid_approx(sum);
    }

    // Calculate output
    let mut output_sum = model.bias_output;
    for h in 0..hidden_size {
        output_sum += hidden_activations[h] * model.weights_hidden_output[h];
    }

    sigmoid_approx(output_sum) // Final output
}

ic_cdk::export_candid!();