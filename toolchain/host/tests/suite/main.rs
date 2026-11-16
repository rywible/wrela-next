//! The native host's tests, in one binary: loading, the first-light fixture, and GPU behavior.

// Every test but the fixture's loads a `Host`, which only a build with the `gpu` feature has.
#[cfg(feature = "gpu")]
#[cfg(feature = "gpu")]
mod common;
mod fixture;
#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
mod load;
