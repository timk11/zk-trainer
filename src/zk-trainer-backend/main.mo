import Wrapper "canister:wrapper";
import Nat "mo:base/Nat";

actor {
  type BaseElement = Nat; // Type renaming is not essential, but will become more helpful as more types are added

  public func exp(base: BaseElement, exponent: Nat): async Nat {
    // Call the exp function in the Rust canister
    let result = await Wrapper.exp(base, exponent);
    return result;
  };
};
