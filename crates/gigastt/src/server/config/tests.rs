use super::*;

#[test]
fn test_runtime_limits_default_rate_limit_disabled() {
    let limits = RuntimeLimits::default();
    assert_eq!(
        limits.rate_limit_per_minute, 0,
        "rate limiting must be off by default (privacy-first)"
    );
    assert_eq!(limits.rate_limit_burst, 10, "default burst size must be 10");
}

#[test]
fn test_runtime_limits_default_jobs() {
    let limits = RuntimeLimits::default();
    assert!(!limits.jobs_enabled);
    assert_eq!(limits.jobs_ttl_secs, 3600);
    assert_eq!(limits.jobs_max, 100);
    assert_eq!(limits.jobs_max_bytes, 512 * 1024 * 1024);
    assert_eq!(limits.jobs_retry, 3);
}

#[test]
fn test_max_audio_secs_default_unlimited_and_opt_mapping() {
    let limits = RuntimeLimits::default();
    assert_eq!(
        limits.max_audio_secs, 0,
        "default must be 0 = unlimited (owner's decision); files of any length transcribe"
    );
    assert_eq!(
        limits.max_audio_secs_opt(),
        None,
        "0 maps to None (unlimited)"
    );
    let capped = RuntimeLimits {
        max_audio_secs: 600,
        ..Default::default()
    };
    assert_eq!(capped.max_audio_secs_opt(), Some(600.0));
}

#[test]
fn test_runtime_limits_default_session_and_drain() {
    // Locks in the documented defaults so a silent change
    // can't quietly disable the shutdown drain or the session cap.
    let limits = RuntimeLimits::default();
    assert_eq!(
        limits.max_session_secs, 3600,
        "default session cap must be 1 hour to stop silence-streamers from \
         holding a triplet forever"
    );
    assert_eq!(
        limits.shutdown_drain_secs, 10,
        "default shutdown drain must be 10 s — comfortably inside the usual \
         k8s terminationGracePeriodSeconds = 30"
    );
}

#[test]
fn test_supported_rates_contains_common() {
    assert!(
        SUPPORTED_RATES.contains(&8000),
        "SUPPORTED_RATES must include 8000 Hz"
    );
    assert!(
        SUPPORTED_RATES.contains(&16000),
        "SUPPORTED_RATES must include 16000 Hz"
    );
    assert!(
        SUPPORTED_RATES.contains(&48000),
        "SUPPORTED_RATES must include 48000 Hz"
    );
}

#[test]
fn test_default_sample_rate_in_supported() {
    assert!(
        SUPPORTED_RATES.contains(&DEFAULT_SAMPLE_RATE),
        "DEFAULT_SAMPLE_RATE ({DEFAULT_SAMPLE_RATE}) must be present in SUPPORTED_RATES"
    );
}

#[test]
fn test_loopback_origin_matcher() {
    assert!(is_loopback_origin("http://localhost"));
    assert!(is_loopback_origin("https://localhost:3000"));
    assert!(is_loopback_origin("http://127.0.0.1:9876"));
    assert!(is_loopback_origin("HTTPS://127.0.0.1")); // case-insensitive
    assert!(is_loopback_origin("http://[::1]:9876"));
    assert!(!is_loopback_origin("https://evil.example.com"));
    assert!(!is_loopback_origin("http://192.168.1.10"));
    // Foiled prefix spoof: host must be exactly localhost / 127.0.0.1 / [::1]
    assert!(!is_loopback_origin("http://localhost.evil.example.com"));
}

#[test]
fn test_origin_policy_default_denies_third_party() {
    let policy = OriginPolicy::loopback_only();
    assert!(matches!(
        policy.evaluate(Some("https://evil.example.com")),
        OriginVerdict::Denied
    ));
}

#[test]
fn test_origin_policy_allows_loopback_by_default() {
    let policy = OriginPolicy::loopback_only();
    assert!(matches!(
        policy.evaluate(Some("http://localhost:3000")),
        OriginVerdict::Allowed(_)
    ));
}

#[test]
fn test_origin_policy_allows_listed_origin() {
    let policy = OriginPolicy {
        allow_any: false,
        allowed_origins: vec!["https://app.example.com".into()],
    };
    assert!(matches!(
        policy.evaluate(Some("https://app.example.com")),
        OriginVerdict::Allowed(_)
    ));
    // Trailing-path mutations are not a match — allowlist is exact origin only.
    assert!(matches!(
        policy.evaluate(Some("https://app.example.com.evil.com")),
        OriginVerdict::Denied
    ));
}

#[test]
fn test_origin_policy_allow_any_short_circuits() {
    let policy = OriginPolicy {
        allow_any: true,
        allowed_origins: vec![],
    };
    assert!(matches!(
        policy.evaluate(Some("https://anything.example.com")),
        OriginVerdict::Allowed(_)
    ));
}

#[test]
fn test_runtime_limits_from_toml() {
    let toml_str = r#"
        idle_timeout_secs = 600
        rate_limit_per_minute = 120
    "#;
    let cfg: RuntimeLimitsConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(cfg.idle_timeout_secs, 600);
    assert_eq!(cfg.rate_limit_per_minute, 120);
    assert_eq!(cfg.max_session_secs, 3600);
}

#[test]
fn test_runtime_limits_config_to_limits() {
    let cfg = RuntimeLimitsConfig::default();
    let limits: RuntimeLimits = cfg.into();
    let defaults = RuntimeLimits::default();
    assert_eq!(limits.idle_timeout_secs, defaults.idle_timeout_secs);
    assert_eq!(limits.max_session_secs, defaults.max_session_secs);
}

#[test]
fn test_origin_policy_no_header_allowed() {
    let policy = OriginPolicy::loopback_only();
    assert!(matches!(
        policy.evaluate(None),
        OriginVerdict::AllowedNoEcho
    ));
}

#[test]
fn test_origin_policy_null_origin_denied() {
    let policy = OriginPolicy::loopback_only();
    assert!(matches!(
        policy.evaluate(Some("null")),
        OriginVerdict::Denied
    ));
    assert!(matches!(
        policy.evaluate(Some("NULL")),
        OriginVerdict::Denied
    ));
}

#[test]
fn test_pool_retry_after_ms_saturation() {
    let limits = RuntimeLimits {
        pool_checkout_timeout_secs: u32::MAX as u64,
        ..Default::default()
    };
    let ms = pool_retry_after_ms(&limits);
    assert_eq!(ms, u32::MAX);
}

#[test]
fn test_load_config_file_not_found() {
    let result = load_config_file(std::path::Path::new("/nonexistent/config.toml"));
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("Failed to read config file"));
}

#[test]
fn test_load_config_file_bad_toml() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), b"not valid toml {{{").unwrap();
    let result = load_config_file(tmp.path());
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("Failed to parse config file"));
}

#[test]
fn test_load_config_file_valid() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        tmp.path(),
        b"idle_timeout_secs = 123\nws_frame_max_bytes = 1024\n",
    )
    .unwrap();
    let limits = load_config_file(tmp.path()).unwrap();
    assert_eq!(limits.idle_timeout_secs, 123);
    assert_eq!(limits.ws_frame_max_bytes, 1024);
    assert_eq!(limits.max_session_secs, 3600);
}

#[test]
fn test_server_config_local_defaults() {
    let cfg = ServerConfig::local(9876);
    assert_eq!(cfg.port, 9876);
    assert_eq!(cfg.host, "127.0.0.1");
    assert!(!cfg.metrics_enabled);
    assert!(!cfg.trust_proxy);
    assert!(cfg.config_path.is_none());
}
