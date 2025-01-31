use serde::{Deserialize, Serialize};
use candid::CandidType;
use winterfell::math::{fields::f128::BaseElement, FieldElement, StarkField}; // https://crates.io/crates/winterfell

// This wraps the BaseElement type from `winterfell`
pub struct WrappedBaseElement(pub BaseElement);

impl CandidType for WrappedBaseElement {
    fn _ty() -> candid::types::Type {
        // Treat the wrapper as a u128 for Candid purposes
        u128::_ty()
    }

    fn idl_serialize<S>(&self, serializer: S) -> Result<(), S::Error>
    where
        S: candid::types::Serializer,
    {
        // Serialise BaseElement as a u128 for Candid
        let value: u128 = self.0.as_int();
        serializer.serialize_u128(value)
    }
}

impl Serialize for WrappedBaseElement {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialise BaseElement as a u128
        let value: u128 = self.0.as_int();
        serializer.serialize_u128(value)
    }
}

impl<'de> Deserialize<'de> for WrappedBaseElement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Deserialise BaseElement from a u128
        let value = u128::deserialize(deserializer)?;
        Ok(WrappedBaseElement(BaseElement::new(value.into())))
    }
}

// This tests using the BaseElement `exp` method in a Rust canister
#[ic_cdk::query]
fn square(start: WrappedBaseElement) -> WrappedBaseElement {
    let mut result = start.0;
    result = result.exp(2);
    WrappedBaseElement(result)
}

// This exports the BaseElement `exp` method for use in a Motoko canister
#[ic_cdk::query]
fn exp(base: WrappedBaseElement, exponent: u128) -> WrappedBaseElement {
    let result = base.0.exp(exponent); // Using the `exp` method from winterfell
    WrappedBaseElement(result)
}

ic_cdk::export_candid!();
