//! Minimal Rust <-> Core ML bridge over `objc2-core-ml`.
//!
//! Production status: this bridge is the Core ML entry point for the composite
//! ANE runtime — [`super::encoder_session::AneEncoderSession`] (one per pooled
//! production session) calls [`predict_f32`] on every ANE-path encoder run, and
//! [`super::runtime::AneRuntime`] calls [`compile_and_load`] once per bucket at
//! load time. It compiles + loads a per-bucket `.mlpackage`, runs it on the
//! Apple Neural Engine (`CPU_AND_NE`), and produces output that matches a Python
//! `coremltools` reference on the SAME package + input (verified by the
//! `#[ignore]` GO/NO-GO smoke test below). This is the only module in the crate
//! allowed to touch `objc2_core_ml` / `objc2_foundation` (the parent enforces
//! the isolation).
//!
//! ISOLATION: all `objc2_*` usage stays inside `runtime/coreml/`.
//! Gated `#[cfg(all(feature = "ane", target_os = "macos"))]`.
//!
//! Every `objc2` call is `unsafe` (Objective-C messaging); `unsafe` blocks are
//! kept tight and each carries a SAFETY note. Failures map to `RuntimeError`
//! variants — never `unwrap` on an objc2 result.

mod cache;
mod predict;

pub use cache::compile_and_load;
pub use predict::predict_f32;

/// Extract a human-readable message from an NSError without leaking it to clients
/// (used only for internal `RuntimeError` messages / test diagnostics).
pub(super) fn ns_error_message(err: &objc2_foundation::NSError) -> String {
    // `localizedDescription` is a safe getter in this objc2-foundation version;
    // it returns an owned NSString describing the error.
    err.localizedDescription().to_string()
}

#[cfg(test)]
mod tests;
