//! Download progress reporting (human stderr + NDJSON).

use std::sync::atomic::{AtomicU8, Ordering};

/// Progress reporting mode for `gigastt download` (and any other caller that
/// sets it process-wide): `Human` keeps the interactive `\r` stderr reporter,
/// `Json` emits NDJSON events — one [`ProgressEvent`] per line — on stdout
/// for sidecar integrators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ProgressMode {
    /// Interactive human reporter: a `\r`-redrawn percentage line on stderr.
    #[default]
    Human,
    /// Machine-readable NDJSON events on stdout; nothing else may write there.
    Json,
}

impl ProgressMode {
    /// Stable token accepted by `--progress` / `GIGASTT_DOWNLOAD_PROGRESS`.
    pub fn as_str(self) -> &'static str {
        match self {
            ProgressMode::Human => "human",
            ProgressMode::Json => "json",
        }
    }
}

impl std::str::FromStr for ProgressMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "human" => Ok(ProgressMode::Human),
            "json" => Ok(ProgressMode::Json),
            other => Err(format!(
                "unknown progress mode '{other}' (expected 'human' or 'json')"
            )),
        }
    }
}

/// Failure category surfaced in the NDJSON `error` event and mapped to the
/// documented `gigastt download` exit-code taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProgressErrorKind {
    /// Network failure: unreachable host, TLS, broken stream, HTTP error status.
    Network,
    /// Local filesystem failure: create/write/rename of a staging or model file.
    Disk,
    /// SHA-256 mismatch on a staged download (corrupt or tampered artefact).
    Checksum,
    /// Cancelled by the operator (SIGINT / Ctrl-C).
    Interrupted,
    /// Anything else; keeps the four primary kinds stable to match on.
    Other,
}

impl ProgressErrorKind {
    /// `gigastt download` process exit code for this failure category. `0` is
    /// success; every category keeps the historical `!= 0` failure contract.
    pub fn exit_code(self) -> i32 {
        // BSD `sysexits`-flavored codes, deliberately avoiding 2: clap exits 2
        // on argument/usage errors (before any event can be emitted), and an
        // integrator keying retries off "network" must be able to tell the two
        // apart.
        match self {
            // EX_UNAVAILABLE: the remote end could not be reached / served.
            ProgressErrorKind::Network => 69,
            // EX_IOERR: local create/write/rename failure.
            ProgressErrorKind::Disk => 74,
            // EX_DATAERR: SHA-256 mismatch on a staged download.
            ProgressErrorKind::Checksum => 65,
            // Conventional 128 + SIGINT.
            ProgressErrorKind::Interrupted => 130,
            // Historical generic failure code (anyhow's `Termination`).
            ProgressErrorKind::Other => 1,
        }
    }
}

/// Machine-readable `gigastt download` progress event, serialized as a single
/// NDJSON line on stdout when [`ProgressMode::Json`] is active. One line = one
/// event; the `phase` tag is the discriminator integrators match on.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProgressEvent {
    /// Byte progress of one file's download (throttled to ~200 ms per file,
    /// plus an unconditional event at 100%).
    Download {
        file: String,
        bytes_done: u64,
        bytes_total: u64,
    },
    /// The on-device INT8 quantization pass started for `file` (~2 min, no
    /// byte progress — its presence tells a sidecar the CLI is busy, not hung).
    Quantize { file: String },
    /// SHA-256 verification of a staged download started.
    Verify { file: String },
    /// All requested artefacts are ready; emitted once, last.
    Done { model_dir: String },
    /// Fatal failure, emitted right before the non-zero exit.
    Error {
        kind: ProgressErrorKind,
        message: String,
    },
}

impl ProgressEvent {
    /// Serialize as one NDJSON line (no trailing newline). This POD enum
    /// cannot realistically fail serialization; a minimal valid `error`
    /// object is the fallback rather than panicking on a progress path.
    pub fn to_ndjson(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            "{\"phase\":\"error\",\"kind\":\"other\",\"message\":\"progress event serialization failed\"}"
                .to_string()
        })
    }
}

/// Process-wide download progress mode, set once by the CLI before any
/// download call. Library users get the `Human` default (identical to the
/// historical `\r` reporter) unless they opt into JSON events.
static PROGRESS_MODE: AtomicU8 = AtomicU8::new(ProgressMode::Human as u8);

/// Set the process-wide download progress mode. Call once at startup, before
/// any `ensure_*` download function runs.
pub fn set_progress_mode(mode: ProgressMode) {
    PROGRESS_MODE.store(mode as u8, Ordering::Relaxed);
}

/// The process-wide download progress mode (`Human` unless set).
pub fn progress_mode() -> ProgressMode {
    match PROGRESS_MODE.load(Ordering::Relaxed) {
        1 => ProgressMode::Json,
        _ => ProgressMode::Human,
    }
}

/// Emit `event` as one NDJSON line on stdout — but only in
/// [`ProgressMode::Json`]; in `Human` mode this is a no-op so call sites never
/// branch on the mode. Public because phases only the CLI can see (quantize
/// entry, terminal done/error) are emitted from the binary crate.
///
/// Write failures (e.g. the reader closed the pipe) are ignored: progress
/// reporting must never take down an in-flight download.
pub fn emit_progress_event(event: &ProgressEvent) {
    if progress_mode() != ProgressMode::Json {
        return;
    }
    use std::io::Write;
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = writeln!(lock, "{}", event.to_ndjson());
    let _ = lock.flush();
}

/// Classify a failed download for the NDJSON `error` event / exit-code
/// taxonomy. Typed root causes win (reqwest → network, io → disk); the
/// remaining cases are recognized by the stable messages of the two
/// `anyhow::bail!` sites in this module (SHA-256 mismatch, HTTP error status).
#[cfg(feature = "net")]
pub fn classify_download_error(err: &anyhow::Error) -> ProgressErrorKind {
    for cause in err.chain() {
        if cause.downcast_ref::<reqwest::Error>().is_some() {
            return ProgressErrorKind::Network;
        }
        if cause.downcast_ref::<std::io::Error>().is_some() {
            return ProgressErrorKind::Disk;
        }
    }
    let msg = format!("{err:#}");
    if msg.contains("SHA-256 mismatch") {
        return ProgressErrorKind::Checksum;
    }
    // `stream_to_partial_then_finalize` bails with "…: HTTP <status>" on a
    // non-2xx response; that is a network-class failure for integrators.
    if msg.contains("HTTP ") {
        return ProgressErrorKind::Network;
    }
    ProgressErrorKind::Other
}

/// Throttle for NDJSON `download` events: at most one per 200 ms per file,
/// plus an unconditional event at 100% so integrators always see completion.
#[cfg(feature = "net")]
pub(super) const JSON_PROGRESS_THROTTLE: std::time::Duration =
    std::time::Duration::from_millis(200);

/// Where progress output goes. `Human` renders the legacy `\r` stderr line;
/// `Json` routes [`ProgressEvent`]s to the emitter (process stdout in the CLI,
/// a captured buffer in tests).
#[cfg(feature = "net")]
pub(super) struct ProgressSink {
    pub(super) mode: ProgressMode,
    pub(super) emit: Box<dyn Fn(&ProgressEvent) + Send + Sync>,
}

#[cfg(feature = "net")]
impl ProgressSink {
    /// Sink honouring the process-wide [`ProgressMode`] set by the CLI.
    pub(super) fn global() -> Self {
        Self {
            mode: progress_mode(),
            emit: Box::new(emit_progress_event),
        }
    }

    /// Human-mode sink for tests that exercise the legacy renderer.
    #[cfg(test)]
    pub(super) fn human() -> Self {
        Self {
            mode: ProgressMode::Human,
            emit: Box::new(|_| {}),
        }
    }

    /// Capturing Json-mode sink: returns the sink plus the shared event log.
    #[cfg(test)]
    pub(super) fn capturing() -> (Self, std::sync::Arc<std::sync::Mutex<Vec<ProgressEvent>>>) {
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink_log = std::sync::Arc::clone(&log);
        (
            Self {
                mode: ProgressMode::Json,
                emit: Box::new(move |e| {
                    if let Ok(mut guard) = sink_log.lock() {
                        guard.push(e.clone());
                    }
                }),
            },
            log,
        )
    }

    pub(super) fn event(&self, event: &ProgressEvent) {
        (self.emit)(event);
    }
}

/// Per-file download progress reporter. Human mode keeps the historical
/// stderr `\r` rendering byte-for-byte; Json mode emits throttled NDJSON
/// `download` events through the sink (first chunk immediately, then at most
/// one per [`JSON_PROGRESS_THROTTLE`], and always exactly one at 100%).
#[cfg(feature = "net")]
pub(super) struct DownloadProgress {
    total: u64,
    pub(super) current: u64,
    pub(super) last_percent: u8,
    pub(super) last_json_emit: Option<std::time::Instant>,
    json_final_emitted: bool,
}

#[cfg(feature = "net")]
impl DownloadProgress {
    pub(super) fn new(total: u64) -> Self {
        Self {
            total,
            current: 0,
            last_percent: 0,
            last_json_emit: None,
            json_final_emitted: false,
        }
    }

    /// The legacy human line for the current state, or `None` when the
    /// percentage did not move (the historical throttle).
    pub(super) fn human_tick(&mut self) -> Option<String> {
        let percent = (self.current * 100)
            .checked_div(self.total)
            .map(|p| p as u8)
            .unwrap_or(0);
        if percent == self.last_percent {
            return None;
        }
        self.last_percent = percent;
        Some(format!(
            "\rDownloading... {percent}% ({:.1}MB / {:.1}MB)",
            self.current as f64 / 1_048_576.0,
            self.total as f64 / 1_048_576.0
        ))
    }

    /// The legacy human completion line (redraws over the progress line).
    pub(super) fn human_finish(&self) -> String {
        format!(
            "\rDownload complete ({:.1}MB)                    ",
            self.current as f64 / 1_048_576.0
        )
    }

    pub(super) fn update(&mut self, bytes: u64, sink: &ProgressSink, label: &str) {
        self.current += bytes;
        match sink.mode {
            ProgressMode::Human => {
                if let Some(line) = self.human_tick() {
                    eprint!("{line}");
                }
            }
            ProgressMode::Json => {
                let complete = self.total > 0 && self.current >= self.total;
                let due = self
                    .last_json_emit
                    .is_none_or(|t| t.elapsed() >= JSON_PROGRESS_THROTTLE);
                if (complete && !self.json_final_emitted) || (due && !complete) {
                    sink.event(&ProgressEvent::Download {
                        file: label.to_string(),
                        bytes_done: self.current,
                        bytes_total: self.total,
                    });
                    self.last_json_emit = Some(std::time::Instant::now());
                    if complete {
                        self.json_final_emitted = true;
                    }
                }
            }
        }
    }

    pub(super) fn finish(&mut self, sink: &ProgressSink, label: &str) {
        match sink.mode {
            ProgressMode::Human => eprintln!("{}", self.human_finish()),
            ProgressMode::Json => {
                // An unknown (chunked) total never hits the 100% branch in
                // `update`; close the file out with exactly one final event.
                if !self.json_final_emitted {
                    self.json_final_emitted = true;
                    sink.event(&ProgressEvent::Download {
                        file: label.to_string(),
                        bytes_done: self.current,
                        bytes_total: self.total,
                    });
                }
            }
        }
    }
}
