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
    let r: Vec<anyhow::Result<u32>> = vec![Err(anyhow::anyhow!("a")), Err(anyhow::anyhow!("b"))];
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
