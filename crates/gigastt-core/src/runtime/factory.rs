use super::{error::RuntimeError, session::RuntimeSession};

/// Creates a `Runtime` configured for a specific execution provider.
pub trait RuntimeFactory: Send + Sync + 'static {
    fn create(&self, intra_threads: usize) -> Result<Box<dyn Runtime>, RuntimeError>;

    /// Returns a CPU-only factory suitable for small auxiliary models.
    fn cpu_fallback(&self) -> Box<dyn RuntimeFactory>;

    /// Whether [`Engine::load`] should refuse files whose SHA-256 does not
    /// match the pinned table. Production factories leave this `true`. The
    /// mock factory returns `false` so unit tests can load empty placeholder
    /// ONNX files.
    fn verify_on_disk_checksums(&self) -> bool {
        true
    }
}

/// Owns loaded sessions. One runtime per `Engine`.
pub trait Runtime: Send + Sync + 'static {
    fn load_session(
        &self,
        model_path: &std::path::Path,
        is_encoder: bool,
    ) -> Result<Box<dyn RuntimeSession>, RuntimeError>;
}
