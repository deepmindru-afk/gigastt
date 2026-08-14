//! CLI punctuation / ITN modes and encoder thread budgeting.

use gigastt_core::model::ModelVariant;

// ---------------------------------------------------------------------------
// CLI mode enums (shared by clap and the recipe)
// ---------------------------------------------------------------------------

/// Whether to run the optional punctuation / casing restoration pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PunctuationMode {
    /// Always attempt to load + apply the punct model.
    On,
    /// Never apply punctuation (pass-through bare output).
    Off,
    /// Decide from the active model variant: on for `rnnt` (bare output),
    /// off for `e2e_rnnt` (punctuation already baked into the head).
    Auto,
}

impl std::str::FromStr for PunctuationMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "1" | "yes" => Ok(PunctuationMode::On),
            "off" | "false" | "0" | "no" => Ok(PunctuationMode::Off),
            "auto" => Ok(PunctuationMode::Auto),
            other => Err(format!(
                "unknown punctuation mode '{other}' (expected 'on', 'off', or 'auto')"
            )),
        }
    }
}

/// clap value parser for `--punctuation`.
pub fn parse_punctuation_mode(s: &str) -> Result<PunctuationMode, String> {
    s.parse()
}

/// Whether to run the optional inverse text normalization pass
/// (Russian number-words → digits). Mirrors [`PunctuationMode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItnMode {
    /// Always apply ITN.
    On,
    /// Never apply ITN (pass-through number-words).
    Off,
    /// Decide from the active model variant: on for `rnnt` (spells numbers as
    /// words), off for `e2e_rnnt` (ITN already baked into the head).
    Auto,
}

impl std::str::FromStr for ItnMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "1" | "yes" => Ok(ItnMode::On),
            "off" | "false" | "0" | "no" => Ok(ItnMode::Off),
            "auto" => Ok(ItnMode::Auto),
            other => Err(format!(
                "unknown ITN mode '{other}' (expected 'on', 'off', or 'auto')"
            )),
        }
    }
}

/// clap value parser for `--itn`.
pub fn parse_itn_mode(s: &str) -> Result<ItnMode, String> {
    s.parse()
}

/// Resolve `--itn` against the active model variant: `auto` enables ITN only
/// for the bare `rnnt` head (the `e2e_rnnt` head already digitizes numbers).
pub fn resolve_itn(mode: ItnMode, variant: ModelVariant) -> bool {
    match mode {
        ItnMode::On => true,
        ItnMode::Off => false,
        ItnMode::Auto => variant == ModelVariant::Rnnt,
    }
}

/// Resolve `--punctuation` against the active model variant: `auto` enables the
/// pass only for the bare `rnnt` head (`e2e_rnnt` already punctuates).
pub fn resolve_punctuation(mode: PunctuationMode, variant: ModelVariant) -> bool {
    match mode {
        PunctuationMode::On => true,
        PunctuationMode::Off => false,
        // e2e_rnnt already emits punctuation/casing, so only the bare rnnt head
        // benefits from the restoration pass.
        PunctuationMode::Auto => variant == ModelVariant::Rnnt,
    }
}

// ---------------------------------------------------------------------------
// Thread budgeting
// ---------------------------------------------------------------------------

/// Resolve the encoder intra-op thread count when the operator left the flag /
/// env unset. `requested == Some(v)` (an explicit flag/env value, including `1`)
/// is honoured verbatim and only passes through the engine's oversubscription
/// clamp downstream. `None` (unset) spreads the logical CPUs across the
/// concurrently-running pool triplets: `max(1, budget / total_pool_slots)`,
/// so a default install uses nearly every core instead of one.
///
/// When unset, spreads the logical CPUs across concurrent pool slots:
/// `max(1, logical_cpus / total_pool_slots)`. A default install uses every core
/// instead of one (critical for stretch RTF on encoder-bound work). Explicit
/// values still pass through for debug. `total_pool_slots` is the effective
/// number of triplets that can run at once (serve: `pool_size + batch_pool_size`;
/// offline transcribe: `1`).
///
/// Pure and total so the budgeting math is unit-tested without touching ORT or
/// the real CPU count.
pub fn resolve_encoder_intra_threads(
    requested: Option<usize>,
    total_pool_slots: usize,
    logical_cpus: usize,
) -> usize {
    match requested {
        Some(explicit) => explicit,
        None => {
            let slots = total_pool_slots.max(1);
            let cpus = logical_cpus.max(1);
            (cpus / slots).max(1)
        }
    }
}
