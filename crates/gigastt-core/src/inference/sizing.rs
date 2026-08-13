//! Pure pool / thread / RAM budgeting helpers for engine load.
//!
//! Free of ONNX sessions so unit tests can drive the math with plain inputs.
//! Called from [`super::engine::Engine`] load cascade.

use anyhow::Context;

/// Approximate resident bytes a single pooled encoder triplet costs, as a
/// multiple of the encoder file size on disk. Measured at ~1.9x the INT8
/// encoder file (225 MB file → ~0.4 GB resident per extra pooled slot, dynamic
/// INT8 graph, CPU EP, release). Used by [`cap_pool_size_for_ram`] to keep
/// `pool_size * encoder_file_bytes * this` under a fraction of total RAM.
pub(crate) const ENCODER_RESIDENT_MULTIPLIER: u64 = 2;

/// Fraction (denominator) of total system RAM the pooled encoder sessions are
/// allowed to occupy before [`cap_pool_size_for_ram`] clamps the pool.
/// `2` = at most half of total RAM budgeted to encoder slots, leaving headroom
/// for the decoder/joiner sessions, audio buffers, inference arenas, and the
/// rest of the system.
pub(crate) const POOL_RAM_FRACTION_DENOM: u64 = 2;

/// Parse a cgroup memory limit file body (`memory.max` v2 or
/// `memory.limit_in_bytes` v1). Returns `None` for missing/unbounded/`max`.
/// Pure so unit tests can feed strings without a real cgroup mount.
#[cfg(any(test, target_os = "linux"))]
pub(crate) fn parse_cgroup_memory_limit(raw: &str) -> Option<u64> {
    let s = raw.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("max") {
        return None;
    }
    let bytes: u64 = s.parse().ok()?;
    // Kernel v1 often reports a huge sentinel (~2^63-1) when unlimited.
    if bytes == 0 || bytes >= (1u64 << 62) {
        return None;
    }
    Some(bytes)
}

/// Read Linux cgroup memory limit (v2 then v1). `None` on non-Linux or when
/// unlimited / unreadable. Does not panic on missing files.
pub(crate) fn cgroup_memory_limit_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        const CANDIDATES: &[&str] = &[
            "/sys/fs/cgroup/memory.max",                   // cgroup v2 unified
            "/sys/fs/cgroup/memory/memory.limit_in_bytes", // cgroup v1
        ];
        for path in CANDIDATES {
            if let Ok(raw) = std::fs::read_to_string(path)
                && let Some(bytes) = parse_cgroup_memory_limit(&raw)
            {
                return Some(bytes);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Total physical RAM in bytes, or `0` if it can't be determined (in which case
/// the pool RAM cap is a no-op). macOS: `sysctl HW_MEMSIZE`; Linux/other unix:
/// `sysconf(_SC_PHYS_PAGES) * _SC_PAGESIZE`.
pub(crate) fn total_ram_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut mem: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let mib = [libc::CTL_HW, libc::HW_MEMSIZE];
        // SAFETY: `mib`/`mem`/`len` are valid for the duration of the call;
        // sysctl writes at most `len` bytes into `mem`.
        let rc = unsafe {
            libc::sysctl(
                mib.as_ptr() as *mut libc::c_int,
                mib.len() as libc::c_uint,
                &mut mem as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 { mem } else { 0 }
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        // SAFETY: sysconf has no side effects and returns -1 on error.
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if pages > 0 && page_size > 0 {
            (pages as u64).saturating_mul(page_size as u64)
        } else {
            0
        }
    }
    #[cfg(not(unix))]
    {
        0
    }
}

/// Effective RAM budget for pool sizing: **min(host RAM, cgroup memory.max)**
/// when a cgroup limit is present (Docker/k8s). Falls back to host-only when
/// cgroup files are missing (macOS, bare metal without limits).
pub(crate) fn effective_ram_bytes() -> u64 {
    let host = total_ram_bytes();
    match cgroup_memory_limit_bytes() {
        Some(limit) if limit > 0 => {
            if host == 0 {
                limit
            } else {
                host.min(limit)
            }
        }
        _ => host,
    }
}

/// Number of triplets to reserve for the batch pool given `n` loaded and a
/// requested `batch_pool_size`, always leaving at least one for the
/// interactive pool (so `n <= 1` or `batch_pool_size == 0` yields 0).
pub(crate) fn batch_split_count(n: usize, batch_pool_size: usize) -> usize {
    batch_pool_size.min(n.saturating_sub(1))
}

/// Clamp the requested `pool_size` so the pooled encoder sessions can't
/// exceed [`POOL_RAM_FRACTION_DENOM`]⁻¹ of total RAM. Each triplet costs
/// about `encoder_bytes * ENCODER_RESIDENT_MULTIPLIER` resident (the encoder
/// dominates; decoder/joiner are small), so the max safe pool is
/// `(total_ram / denom) / per_triplet`, never below 1. Logs a warning when
/// it clamps. A no-op (returns `requested`) when RAM or encoder size is
/// unknown (`0`) — we never *raise* the requested size, only lower it.
pub(crate) fn cap_pool_size_for_ram(requested: usize, encoder_bytes: u64, total_ram: u64) -> usize {
    if requested <= 1 || encoder_bytes == 0 || total_ram == 0 {
        return requested.max(1);
    }
    let per_triplet = encoder_bytes.saturating_mul(ENCODER_RESIDENT_MULTIPLIER);
    let budget = total_ram / POOL_RAM_FRACTION_DENOM;
    // At least one slot always allowed even if a single triplet exceeds the
    // budget — the pool can't be empty, and partial-load tolerance
    // (`min_size`) handles a genuine OOM at load time.
    let max_slots = (budget / per_triplet.max(1)).max(1) as usize;
    if max_slots < requested {
        tracing::warn!(
            "Capping pool size {requested} -> {max_slots}: \
             {requested} encoder slots (~{} MiB each) would exceed half of \
             {} MiB total RAM. Concurrency is reduced; add RAM or lower \
             --pool-size to silence this.",
            per_triplet / (1024 * 1024),
            total_ram / (1024 * 1024),
        );
        max_slots
    } else {
        requested
    }
}

/// Clamp the requested encoder intra-op thread count so the pooled encoder
/// sessions can't oversubscribe the CPU. Each of `pool_size` triplets can
/// run concurrently, so the total intra-op parallelism is
/// `pool_size * threads`; capping that at `logical_cpus` keeps the machine
/// from thrashing on context switches. The effective per-encoder count is
/// therefore `clamp(requested, 1, logical_cpus / pool_size)`, never below 1.
/// Logs a warning when it lowers the request. The default `requested == 1`
/// always returns `1`, so the built sessions are unchanged.
pub(crate) fn clamp_encoder_intra_threads(
    pool_size: usize,
    requested: usize,
    logical_cpus: usize,
) -> usize {
    let requested = requested.max(1);
    let pool_size = pool_size.max(1);
    let logical_cpus = logical_cpus.max(1);
    // Leave at least one thread per encoder even on a machine with fewer
    // logical CPUs than pooled triplets.
    let max_per_encoder = (logical_cpus / pool_size).max(1);
    if requested > max_per_encoder {
        tracing::warn!(
            "Capping encoder intra-op threads {requested} -> {max_per_encoder}: \
             {pool_size} pooled encoder(s) x {requested} threads would exceed \
             the {logical_cpus} logical CPU(s) available. Lower --pool-size or \
             --encoder-intra-threads to silence this."
        );
        max_per_encoder
    } else {
        requested
    }
}

/// Partition `items` into an interactive pool and an optional batch pool of
/// `batch_pool_size` items, always leaving at least one item interactive.
/// `batch_pool_size == 0` (or too few items to split) yields no batch pool.
pub(crate) fn split_pool_items<T: Send>(
    mut items: Vec<T>,
    batch_pool_size: usize,
) -> (Vec<T>, Option<Vec<T>>) {
    let n = items.len();
    let batch = batch_split_count(n, batch_pool_size);
    if batch == 0 {
        return (items, None);
    }
    let batch_items = items.split_off(n - batch);
    (items, Some(batch_items))
}

/// Probe a freshly-built state; on failure, rebuild it once and re-probe.
///
/// `probe` is a runtime self-check, `rebuild` converts the failed state into
/// a replacement (receiving the probe error so it can log the cause). A
/// rebuilt state that still fails the probe is a hard error — there is no
/// second fallback level.
///
/// Production call site is CoreML runtime-fallback; unit tests exercise the
/// pure decision logic without a session.
#[cfg_attr(not(any(test, feature = "coreml")), allow(dead_code))]
pub(crate) fn probe_or_rebuild<S>(
    state: S,
    probe: impl Fn(&S) -> anyhow::Result<()>,
    rebuild: impl FnOnce(S, anyhow::Error) -> anyhow::Result<S>,
) -> anyhow::Result<S> {
    match probe(&state) {
        Ok(()) => Ok(state),
        Err(probe_err) => {
            let rebuilt = rebuild(state, probe_err)?;
            probe(&rebuilt).context("state failed probe even after rebuild")?;
            Ok(rebuilt)
        }
    }
}

/// Decide the final pool from per-triplet load results: returns the
/// successfully loaded triplets when at least `min_size` loaded (warning
/// when the pool is degraded below `pool_size`), or an error describing the
/// shortfall. `min_size` is clamped to `1..=pool_size`.
pub(crate) fn finalize_pool_load<T>(
    results: Vec<anyhow::Result<T>>,
    pool_size: usize,
    min_size: usize,
) -> anyhow::Result<Vec<T>> {
    let min_size = min_size.clamp(1, pool_size.max(1));
    let mut loaded = Vec::with_capacity(results.len());
    let mut first_err: Option<anyhow::Error> = None;
    for r in results {
        match r {
            Ok(t) => loaded.push(t),
            Err(e) => {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
    }
    let n = loaded.len();
    if n >= min_size {
        if n < pool_size {
            let detail = first_err
                .map(|e| format!("; first error: {e:#}"))
                .unwrap_or_default();
            tracing::warn!(
                "degraded pool: loaded {n}/{pool_size} session triplets ({} failed){detail}",
                pool_size - n
            );
        }
        Ok(loaded)
    } else {
        let detail = first_err.map(|e| format!(": {e:#}")).unwrap_or_default();
        Err(anyhow::anyhow!(
            "loaded only {n}/{pool_size} session triplets, need at least {min_size}{detail}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cgroup_memory_limit_max_and_sentinel() {
        assert_eq!(parse_cgroup_memory_limit("max"), None);
        assert_eq!(parse_cgroup_memory_limit("MAX\n"), None);
        assert_eq!(parse_cgroup_memory_limit(""), None);
        assert_eq!(parse_cgroup_memory_limit("0"), None);
        // v1 unlimited sentinel
        assert_eq!(parse_cgroup_memory_limit("9223372036854771712"), None);
        assert_eq!(
            parse_cgroup_memory_limit("1073741824"),
            Some(1024 * 1024 * 1024)
        );
        assert_eq!(
            parse_cgroup_memory_limit("  536870912\n"),
            Some(512 * 1024 * 1024)
        );
    }

    #[test]
    fn test_effective_ram_prefers_tighter_cgroup() {
        // Pure composition: min(host, cgroup) — exercised via cap_pool_size with
        // a 1 GiB "cgroup" budget vs larger host-class numbers.
        let enc = 225 * 1024 * 1024;
        let one_gib = 1024u64 * 1024 * 1024;
        // Half of 1 GiB = 512 MiB budget; ~450 MiB/triplet → max 1 slot.
        assert_eq!(cap_pool_size_for_ram(2, enc, one_gib), 1);
    }

    #[test]
    fn test_cap_pool_size_for_ram_clamps_on_low_memory() {
        // 225 MiB encoder, 2x resident => ~450 MiB/triplet. Half of 2 GiB =
        // 1 GiB budget => floor(1024/450) = 2 slots; a request for 4 clamps.
        let enc = 225 * 1024 * 1024;
        let two_gib = 2 * 1024 * 1024 * 1024;
        assert_eq!(cap_pool_size_for_ram(4, enc, two_gib), 2);
    }

    #[test]
    fn test_cap_pool_size_for_ram_no_clamp_with_ample_ram() {
        // 64 GiB host easily fits a pool of 4 of the same encoder.
        let enc = 225 * 1024 * 1024;
        let sixty_four_gib = 64u64 * 1024 * 1024 * 1024;
        assert_eq!(cap_pool_size_for_ram(4, enc, sixty_four_gib), 4);
    }

    #[test]
    fn test_cap_pool_size_for_ram_never_below_one() {
        // Even a single triplet larger than the whole budget still yields 1 —
        // the pool can't be empty; partial-load tolerance handles real OOM.
        let huge_enc = 8 * 1024 * 1024 * 1024;
        let small_ram = 1024 * 1024 * 1024;
        assert_eq!(cap_pool_size_for_ram(4, huge_enc, small_ram), 1);
    }

    #[test]
    fn test_cap_pool_size_for_ram_noop_on_unknown_inputs() {
        // Unknown RAM or encoder size (0) => never lower the request.
        assert_eq!(cap_pool_size_for_ram(4, 0, 8 << 30), 4);
        assert_eq!(cap_pool_size_for_ram(4, 200 << 20, 0), 4);
        // pool_size <= 1 is returned as-is (min 1).
        assert_eq!(cap_pool_size_for_ram(1, 200 << 20, 1 << 30), 1);
        assert_eq!(cap_pool_size_for_ram(0, 200 << 20, 1 << 30), 1);
    }

    #[test]
    fn test_clamp_encoder_intra_threads() {
        // Default request of 1 is always returned unchanged (no behavior change).
        assert_eq!(clamp_encoder_intra_threads(2, 1, 10), 1);
        assert_eq!(clamp_encoder_intra_threads(4, 1, 4), 1);

        // Fits within budget: pool 2 x 4 threads = 8 <= 16 CPUs -> 4.
        assert_eq!(clamp_encoder_intra_threads(2, 4, 16), 4);

        // Over budget: pool 4 x 4 = 16 > 10 CPUs -> floor(10/4) = 2 per encoder.
        assert_eq!(clamp_encoder_intra_threads(4, 4, 10), 2);

        // More pooled encoders than CPUs still leaves at least 1 thread each.
        assert_eq!(clamp_encoder_intra_threads(8, 4, 4), 1);

        // Zero inputs are floored to 1 (total function, never panics/divides-by-0).
        assert_eq!(clamp_encoder_intra_threads(0, 4, 8), 4);
        assert_eq!(clamp_encoder_intra_threads(2, 0, 8), 1);
        assert_eq!(clamp_encoder_intra_threads(2, 4, 0), 1);
    }

    #[test]
    fn test_batch_split_count_clamps() {
        assert_eq!(batch_split_count(4, 1), 1); // typical: 1 batch, 3 stream
        assert_eq!(batch_split_count(4, 0), 0); // split disabled
        assert_eq!(batch_split_count(4, 10), 3); // clamped: leave 1 interactive
        assert_eq!(batch_split_count(1, 1), 0); // can't split a single triplet
        assert_eq!(batch_split_count(0, 1), 0); // empty pool
        assert_eq!(batch_split_count(2, 1), 1);
    }

    #[test]
    fn test_finalize_pool_load_full() {
        let r: Vec<anyhow::Result<u32>> = vec![Ok(1), Ok(2), Ok(3)];
        assert_eq!(finalize_pool_load(r, 3, 3).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_finalize_pool_load_degraded_boots() {
        // 2 of 4 loaded with min_size 1 → degraded pool is accepted.
        let r: Vec<anyhow::Result<u32>> = vec![
            Ok(1),
            Err(anyhow::anyhow!("boom")),
            Ok(3),
            Err(anyhow::anyhow!("boom2")),
        ];
        assert_eq!(finalize_pool_load(r, 4, 1).unwrap(), vec![1, 3]);
    }

    #[test]
    fn test_finalize_pool_load_below_min_errors() {
        // Only 1 loaded but min_size 2 → error naming the shortfall.
        let r: Vec<anyhow::Result<u32>> = vec![
            Ok(1),
            Err(anyhow::anyhow!("boom")),
            Err(anyhow::anyhow!("boom2")),
        ];
        let err = finalize_pool_load(r, 3, 2).unwrap_err().to_string();
        assert!(err.contains("loaded only 1/3"), "got: {err}");
        assert!(err.contains("need at least 2"), "got: {err}");
    }

    #[test]
    fn test_finalize_pool_load_all_fail_errors() {
        let r: Vec<anyhow::Result<u32>> =
            vec![Err(anyhow::anyhow!("a")), Err(anyhow::anyhow!("b"))];
        assert!(finalize_pool_load(r, 2, 1).is_err());
    }

    #[test]
    fn test_finalize_pool_load_min_clamped_to_pool() {
        // min_size > pool_size is clamped down; a full load still succeeds.
        let r: Vec<anyhow::Result<u32>> = vec![Ok(1), Ok(2)];
        assert_eq!(finalize_pool_load(r, 2, 99).unwrap(), vec![1, 2]);
    }

    #[test]
    fn test_probe_or_rebuild_keeps_state_when_probe_passes() {
        let rebuilt = std::cell::Cell::new(false);
        let result = probe_or_rebuild(
            7u32,
            |v| {
                assert_eq!(*v, 7);
                Ok(())
            },
            |_, _| {
                rebuilt.set(true);
                Ok(99)
            },
        )
        .expect("healthy state must survive unchanged");
        assert_eq!(result, 7);
        assert!(!rebuilt.get(), "rebuild must not run when the probe passes");
    }

    #[test]
    fn test_probe_or_rebuild_rebuilds_when_probe_fails() {
        let result = probe_or_rebuild(
            1u32,
            |v| {
                if *v == 1 {
                    Err(anyhow::anyhow!("first probe failed"))
                } else {
                    Ok(())
                }
            },
            |old, probe_err| {
                assert_eq!(old, 1, "rebuild receives the failed state");
                assert!(
                    probe_err.to_string().contains("first probe failed"),
                    "rebuild receives the probe error for logging"
                );
                Ok(2)
            },
        )
        .expect("rebuilt state passing the probe is OK");
        assert_eq!(result, 2);
    }

    #[test]
    fn test_probe_or_rebuild_propagates_rebuild_error() {
        let result = probe_or_rebuild(
            1u32,
            |_| Err(anyhow::anyhow!("probe failed")),
            |_, _| Err(anyhow::anyhow!("rebuild failed")),
        );
        let err = result.expect_err("rebuild failure must be fatal");
        assert!(err.to_string().contains("rebuild failed"));
    }

    #[test]
    fn test_probe_or_rebuild_fails_when_rebuilt_state_fails_probe() {
        let result = probe_or_rebuild(
            1u32,
            |_| Err(anyhow::anyhow!("always fails")),
            |_, _| Ok(2u32),
        );
        assert!(
            result.is_err(),
            "a rebuilt state that still fails the probe must be a hard error"
        );
    }

    #[test]
    fn test_finalize_pool_load_degraded_includes_error_detail() {
        // The degraded-pool branch logs the first error; exercise the
        // `first_err` formatting path (the loaded triplets are still returned).
        let r: Vec<anyhow::Result<u32>> = vec![Ok(1), Err(anyhow::anyhow!("first failure cause"))];
        assert_eq!(finalize_pool_load(r, 2, 1).unwrap(), vec![1]);
    }

    #[test]
    fn test_finalize_pool_load_below_min_no_errors_when_all_ok_but_short() {
        // No Err entries, but fewer results than pool_size with min above the
        // loaded count → still errors (loaded count is what matters).
        let r: Vec<anyhow::Result<u32>> = vec![Ok(1)];
        let err = finalize_pool_load(r, 3, 2).unwrap_err().to_string();
        assert!(err.contains("loaded only 1/3"), "got: {err}");
    }
}
