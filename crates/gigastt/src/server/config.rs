//! Server configuration types, origin policy, and runtime limits.

use anyhow::Context;
use serde::Deserialize;

/// Supported input sample rates (Hz). Default is 48000 for backward
/// compatibility. Single source of truth for both the WebSocket `Ready`
/// payload and the REST `/v1/models` capabilities response.
pub(crate) const SUPPORTED_RATES: &[u32] = &[8000, 16000, 24000, 44100, 48000];
pub(crate) const DEFAULT_SAMPLE_RATE: u32 = 48000;

/// Derive the pool-backpressure retry hint from the configured checkout
/// timeout so the `Retry-After` header / `retry_after_ms` JSON field stay
/// consistent with the actual wait window.
pub(crate) fn pool_retry_after_ms(limits: &RuntimeLimits) -> u32 {
    limits
        .pool_checkout_timeout_secs
        .saturating_mul(1000)
        .min(u32::MAX as u64) as u32
}
pub(crate) fn pool_retry_after_secs(limits: &RuntimeLimits) -> u64 {
    limits.pool_checkout_timeout_secs
}

/// Origin policy for CORS + cross-origin deny middleware.
///
/// gigastt is a privacy-first local server: by default we deny cross-origin
/// requests outright so a malicious page cannot trigger transcription from a
/// logged-in user's microphone via a drive-by WebSocket. Loopback origins
/// (`localhost`, `127.0.0.1`, `[::1]`) are always permitted; additional origins
/// must be listed explicitly via `--allow-origin`, and the wildcard `*`
/// behavior is opt-in via `--cors-allow-any`.
#[derive(Debug, Clone, Default)]
pub struct OriginPolicy {
    /// When true, the server accepts ANY `Origin` and echoes `*` in the CORS
    /// response — matches the old v0.5.x behavior. Dangerous default-off.
    pub allow_any: bool,
    /// Exact-match allowlist (e.g. `https://app.example.com`). Case-insensitive.
    pub allowed_origins: Vec<String>,
}

impl OriginPolicy {
    /// Loopback-only default policy: cross-origin requests from non-local
    /// pages are denied until the operator adds explicit allowlist entries.
    pub fn loopback_only() -> Self {
        Self::default()
    }
}

#[derive(Debug)]
pub(crate) enum OriginVerdict {
    /// No `Origin` header — treat as a non-browser client (curl, native
    /// SDK). Opaque `null` is **not** this: sandboxed iframes and `data:`
    /// documents send it, and the default policy exists to stop that drive-by.
    AllowedNoEcho,
    /// Origin matches the policy; echo the exact string (or `*` if
    /// `allow_any` is on).
    Allowed(String),
    /// Origin present but not allowed — respond 403 before reaching the
    /// handler.
    Denied,
}

fn is_loopback_origin(origin: &str) -> bool {
    // Normalize once; compare lowercase prefixes. The prefix must be followed
    // by a port separator (`:`), a path (`/`), or end-of-string — otherwise
    // `http://localhost.evil.com` would be accepted as a DNS continuation of
    // the loopback hostname.
    let lowered = origin.to_ascii_lowercase();
    const HOST_PREFIXES: &[&str] = &[
        "http://localhost",
        "https://localhost",
        "http://127.0.0.1",
        "https://127.0.0.1",
        "http://[::1]",
        "https://[::1]",
    ];
    HOST_PREFIXES.iter().any(|p| match lowered.strip_prefix(p) {
        None => false,
        Some(rest) => rest.is_empty() || rest.starts_with(':') || rest.starts_with('/'),
    })
}

impl OriginPolicy {
    pub(crate) fn evaluate(&self, origin: Option<&str>) -> OriginVerdict {
        let Some(origin) = origin else {
            return OriginVerdict::AllowedNoEcho;
        };
        // Browsers send `Origin: null` from sandboxed iframes and `data:`
        // documents. That is still a cross-origin drive-by against a
        // loopback server — deny it. Native clients omit the header entirely.
        if origin.eq_ignore_ascii_case("null") {
            return OriginVerdict::Denied;
        }
        if self.allow_any || is_loopback_origin(origin) {
            return OriginVerdict::Allowed(origin.to_string());
        }
        if self
            .allowed_origins
            .iter()
            .any(|a| a.eq_ignore_ascii_case(origin))
        {
            return OriginVerdict::Allowed(origin.to_string());
        }
        OriginVerdict::Denied
    }
}

/// Runtime limits surfaced to per-request handlers. Separate from `ServerConfig`
/// because it needs to travel through `http::AppState` to the WebSocket handler.
#[derive(Debug, Clone)]
pub struct RuntimeLimits {
    /// WebSocket idle timeout. If no frame arrives within this window the
    /// server closes the connection. Default: 300s.
    pub idle_timeout_secs: u64,
    /// Maximum WebSocket frame / message size in bytes. Default: 512 KiB.
    pub ws_frame_max_bytes: usize,
    /// Maximum REST request body in bytes. Default: 50 MiB.
    pub body_limit_bytes: usize,
    /// Per-IP rate limit: requests-per-minute. `0` disables the limiter
    /// (default). Applies to /v1/* and /v1/ws; /health is exempt.
    pub rate_limit_per_minute: u32,
    /// Max burst size before the token bucket starts refilling.
    pub rate_limit_burst: u32,
    /// Maximum wall-clock duration of a single WebSocket session (seconds).
    /// `0` disables the cap entirely (not recommended — a silence-streaming
    /// client would hold a triplet forever). Default: 3600 (1 hour).
    pub max_session_secs: u64,
    /// Grace window (seconds) after the shutdown signal during which in-flight
    /// WebSocket / SSE tasks may emit their final frames and close cleanly.
    /// Values of `0` are clamped to `1` to avoid a no-op drain. Default: 10.
    pub shutdown_drain_secs: u64,
    /// Pool checkout timeout (seconds). REST and WebSocket handlers wait this
    /// long for a free session triplet before returning 503 / `timeout`.
    /// The `Retry-After` hint echoes the same value. Default: 30.
    pub pool_checkout_timeout_secs: u64,
    /// Per-request **no-progress** inference watchdog (seconds). The deadline
    /// resets every time a decode window completes, so a file that keeps making
    /// progress never trips — only a run that stalls for this many seconds
    /// without finishing a window returns a typed `inference_timeout` (504 on
    /// REST, a job failure on `/v1/jobs`). On a trip the run's abort flag is
    /// flipped, so the pooled triplet is released within one window instead of
    /// staying wedged for the rest of the file. `0` disables the watchdog.
    /// Default: 600.
    ///
    /// This was a *total* wall-clock cap in earlier releases; the flag name,
    /// the default, and the `inference_timeout` error code are unchanged, and a
    /// short file that genuinely hangs still trips at the same moment — but the
    /// hidden ceiling on audio duration (roughly `timeout ÷ RTF`) is gone, so
    /// long files are no longer capped by this limit.
    pub inference_timeout_secs: u64,
    /// Opt-in maximum decoded audio length in seconds for file transcription.
    /// `0` (the default) means **unlimited**: a file of any length transcribes,
    /// because the default path decodes in bounded windows so peak audio memory
    /// is O(one window). When > 0, audio longer than this is rejected with HTTP
    /// 413 and error code `audio_too_long`. The paths that must hold the whole
    /// decoded buffer in RAM — diarization, `channels=split` (including its
    /// per-channel Opus decode), and the raw telephony codecs — keep a
    /// fixed ~30-minute safety ceiling regardless of this value, so they refuse
    /// rather than OOM. The VAD file path, WAVE ingest, and streamed OGG/Opus
    /// decode in bounded windows like the default path, so no ceiling applies
    /// to them.
    pub max_audio_secs: u64,
    /// Whether the asynchronous `/v1/jobs` API is enabled. Off by default so
    /// existing single-user installs see no change.
    pub jobs_enabled: bool,
    /// TTL in seconds for completed/failed/cancelled jobs before eviction.
    /// Default: 3600 (1 hour).
    pub jobs_ttl_secs: u64,
    /// Maximum number of jobs kept in memory (queued + finished). When full,
    /// POST /v1/jobs returns 429 + Retry-After. Default: 100.
    pub jobs_max: usize,
    /// Maximum total bytes of buffered uploads held across all in-memory jobs
    /// (queued + processing; a terminal job releases its body). The count cap
    /// (`jobs_max`) alone can't bound RAM — each queued job holds its full
    /// upload as `Bytes`, so `jobs_max` × `body_limit_bytes` (default 100 × 50
    /// MiB ≈ 5 GiB) could sit in memory while the queue is "not full" by count.
    /// A submission that would push the live total over this budget gets the
    /// same 429 + `Retry-After` backpressure as a count-full queue. Default:
    /// 512 MiB.
    pub jobs_max_bytes: usize,
    /// Maximum retry attempts for a job that panics. Default: 3.
    pub jobs_retry: u32,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            idle_timeout_secs: 300,
            ws_frame_max_bytes: 512 * 1024,
            body_limit_bytes: 50 * 1024 * 1024,
            rate_limit_per_minute: 0,
            rate_limit_burst: 10,
            max_session_secs: 3600,
            shutdown_drain_secs: 10,
            pool_checkout_timeout_secs: 30,
            inference_timeout_secs: 600,
            max_audio_secs: 0,
            jobs_enabled: false,
            jobs_ttl_secs: 3600,
            jobs_max: 100,
            jobs_max_bytes: 512 * 1024 * 1024,
            jobs_retry: 3,
        }
    }
}

impl RuntimeLimits {
    /// The opt-in audio-length limit as the `Option<f64>` seconds the core
    /// request expects: `0` maps to `None` (unlimited), any positive value to
    /// `Some(secs)`.
    pub fn max_audio_secs_opt(&self) -> Option<f64> {
        (self.max_audio_secs != 0).then_some(self.max_audio_secs as f64)
    }
}

/// TOML-deserializable representation of `RuntimeLimits`. Fields default to
/// the same values as `RuntimeLimits::default()` so a partial config file
/// only overrides what the operator cares about.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RuntimeLimitsConfig {
    /// WebSocket idle timeout in seconds.
    pub idle_timeout_secs: u64,
    /// Maximum WebSocket frame size in bytes.
    pub ws_frame_max_bytes: usize,
    /// Maximum REST request body size in bytes.
    pub body_limit_bytes: usize,
    /// Per-IP rate limit in requests per minute (`0` = disabled).
    pub rate_limit_per_minute: u32,
    /// Token-bucket burst size for the rate limiter.
    pub rate_limit_burst: u32,
    /// Maximum wall-clock duration of a single WebSocket session in seconds.
    pub max_session_secs: u64,
    /// Graceful shutdown drain window in seconds.
    pub shutdown_drain_secs: u64,
    /// Pool checkout timeout in seconds before returning 503.
    pub pool_checkout_timeout_secs: u64,
    /// Per-request inference timeout in seconds (`0` disables).
    pub inference_timeout_secs: u64,
    /// Opt-in maximum decoded audio length in seconds (`0` = unlimited).
    pub max_audio_secs: u64,
    /// Enable the asynchronous `/v1/jobs` API.
    pub jobs_enabled: bool,
    /// TTL in seconds for completed/failed/cancelled jobs.
    pub jobs_ttl_secs: u64,
    /// Maximum number of jobs kept in memory.
    pub jobs_max: usize,
    /// Maximum total bytes of buffered job uploads kept in memory.
    pub jobs_max_bytes: usize,
    /// Maximum retry attempts for a panicking job.
    pub jobs_retry: u32,
}

impl Default for RuntimeLimitsConfig {
    fn default() -> Self {
        let d = RuntimeLimits::default();
        Self {
            idle_timeout_secs: d.idle_timeout_secs,
            ws_frame_max_bytes: d.ws_frame_max_bytes,
            body_limit_bytes: d.body_limit_bytes,
            rate_limit_per_minute: d.rate_limit_per_minute,
            rate_limit_burst: d.rate_limit_burst,
            max_session_secs: d.max_session_secs,
            shutdown_drain_secs: d.shutdown_drain_secs,
            pool_checkout_timeout_secs: d.pool_checkout_timeout_secs,
            inference_timeout_secs: d.inference_timeout_secs,
            max_audio_secs: d.max_audio_secs,
            jobs_enabled: d.jobs_enabled,
            jobs_ttl_secs: d.jobs_ttl_secs,
            jobs_max: d.jobs_max,
            jobs_max_bytes: d.jobs_max_bytes,
            jobs_retry: d.jobs_retry,
        }
    }
}

impl From<RuntimeLimitsConfig> for RuntimeLimits {
    fn from(cfg: RuntimeLimitsConfig) -> Self {
        Self {
            idle_timeout_secs: cfg.idle_timeout_secs,
            ws_frame_max_bytes: cfg.ws_frame_max_bytes,
            body_limit_bytes: cfg.body_limit_bytes,
            rate_limit_per_minute: cfg.rate_limit_per_minute,
            rate_limit_burst: cfg.rate_limit_burst,
            max_session_secs: cfg.max_session_secs,
            shutdown_drain_secs: cfg.shutdown_drain_secs,
            pool_checkout_timeout_secs: cfg.pool_checkout_timeout_secs,
            inference_timeout_secs: cfg.inference_timeout_secs,
            max_audio_secs: cfg.max_audio_secs,
            jobs_enabled: cfg.jobs_enabled,
            jobs_ttl_secs: cfg.jobs_ttl_secs,
            jobs_max: cfg.jobs_max,
            jobs_max_bytes: cfg.jobs_max_bytes,
            jobs_retry: cfg.jobs_retry,
        }
    }
}

/// Load runtime limits from a TOML config file.
pub fn load_config_file(path: &std::path::Path) -> anyhow::Result<RuntimeLimits> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;
    let cfg: RuntimeLimitsConfig = toml::from_str(&content)
        .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
    Ok(cfg.into())
}

/// Default bind address for the Prometheus `/metrics` listener — loopback
/// only. Metrics live on their own port so they are never behind the primary
/// CORS allowlist or per-IP rate limiter; operators expose this port
/// deliberately (e.g. to a Prometheus scraper).
pub fn default_metrics_listen() -> std::net::SocketAddr {
    std::net::SocketAddr::from(([127, 0, 0, 1], 9090))
}

/// Server runtime configuration. `run_with_config` is the canonical entry
/// point; `run` / `run_with_shutdown` remain as thin wrappers for callers
/// that only need the pre-0.6 positional parameters.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// TCP port to listen on.
    pub port: u16,
    /// Bind address (e.g. `"127.0.0.1"` for loopback-only or `"0.0.0.0"` for all interfaces).
    pub host: String,
    /// Cross-origin request policy (loopback-only by default).
    pub origin_policy: OriginPolicy,
    /// Runtime limits (timeouts, body sizes, rate-limiting parameters).
    pub limits: RuntimeLimits,
    /// Expose Prometheus metrics. Off by default — keeps the server quiet for
    /// single-user local installs. When on, `/metrics` is served on a
    /// *separate* listener ([`ServerConfig::metrics_listen`]), never on the
    /// primary port, so it is not gated by the CORS allowlist or rate limiter.
    pub metrics_enabled: bool,
    /// Bind address for the Prometheus `/metrics` listener — a separate
    /// loopback socket (default `127.0.0.1:9090`), not the primary port. Only
    /// consulted when `metrics_enabled` is true.
    pub metrics_listen: std::net::SocketAddr,
    /// Trust `X-Forwarded-For` / `X-Real-IP` for rate-limit IP extraction.
    pub trust_proxy: bool,
    /// Path to TOML config file for runtime limits (reloaded on SIGHUP).
    pub config_path: Option<std::path::PathBuf>,
    /// Size of the triplet pool reserved for batch REST / job transcription.
    /// Split off from the interactive pool so long files can't starve WS/SSE.
    pub batch_pool_size: usize,
}

impl ServerConfig {
    /// Sensible local-only default: listen on `127.0.0.1:9876`, deny
    /// non-loopback origins, default runtime limits, metrics off.
    pub fn local(port: u16) -> Self {
        Self {
            port,
            host: "127.0.0.1".to_string(),
            origin_policy: OriginPolicy::loopback_only(),
            limits: RuntimeLimits::default(),
            metrics_enabled: false,
            metrics_listen: default_metrics_listen(),
            trust_proxy: false,
            config_path: None,
            batch_pool_size: 0,
        }
    }
}

#[cfg(test)]
mod tests;
