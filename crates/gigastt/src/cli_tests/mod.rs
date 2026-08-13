//! CLI parse tests for `gigastt` subcommands. Homed next to `main.rs`.

pub(super) use super::serve::ServeProfile;
pub(super) use super::*;
pub(super) use gigastt::server;

// Serialize tests that mutate process env vars to avoid races under
// cargo test's default multi-threaded harness (used by tarpaulin).
pub(super) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// Restore a captured env value when dropped, so an env-mutating test never
// leaks `GIGASTT_ENCODER_INTRA_THREADS` to a sibling test (clap reads the
// process environment). Paired with `ENV_LOCK` to serialize these tests.
pub(super) struct EnvRestore(&'static str, Option<String>);
impl Drop for EnvRestore {
    fn drop(&mut self) {
        match &self.1 {
            Some(v) => unsafe { std::env::set_var(self.0, v) },
            None => unsafe { std::env::remove_var(self.0) },
        }
    }
}

mod batch;
mod download;
mod parse;
mod serve;
mod transcribe;
