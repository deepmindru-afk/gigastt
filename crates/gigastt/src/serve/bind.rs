//! Loopback bind gates for the API and metrics listeners.

use std::net::IpAddr;

/// Guard non-loopback binds. Privacy-first default: the server will only
/// listen on 127.0.0.1 / ::1 / localhost unless the operator opts in via
/// `--bind-all` or `GIGASTT_ALLOW_BIND_ANY=1`. Mirrors the intent of Docker's
/// `--host 0.0.0.0` — explicit consent to expose a local STT service.
pub(crate) fn ensure_bind_allowed(host: &str, bind_all_flag: bool) -> anyhow::Result<()> {
    if is_loopback_host(host) {
        return Ok(());
    }
    let env_opt_in = std::env::var("GIGASTT_ALLOW_BIND_ANY")
        .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);
    if bind_all_flag || env_opt_in {
        tracing::warn!(
            host = %host,
            "binding to non-loopback address — anyone on the network can reach this server"
        );
        return Ok(());
    }
    anyhow::bail!(
        "refusing to bind to '{host}': non-loopback addresses require \
         `--bind-all` (or env GIGASTT_ALLOW_BIND_ANY=1) to prevent accidental \
         public exposure of local transcription"
    )
}

/// Consent gate for the separate metrics listener. That listener serves
/// Prometheus `/metrics` with no CORS allowlist or rate limiter, so a
/// non-loopback `--metrics-listen` requires the same explicit `--bind-all`
/// (or `GIGASTT_ALLOW_BIND_ANY=1`) opt-in as the primary port — keeps the
/// loopback-by-default invariant symmetric instead of letting telemetry leak
/// network-wide silently. No-op when metrics are disabled: nothing is bound.
pub(crate) fn ensure_metrics_bind_allowed(
    metrics_enabled: bool,
    metrics_listen: &std::net::SocketAddr,
    bind_all_flag: bool,
) -> anyhow::Result<()> {
    if !metrics_enabled {
        return Ok(());
    }
    ensure_bind_allowed(&metrics_listen.ip().to_string(), bind_all_flag)
}

pub(crate) fn is_loopback_host(host: &str) -> bool {
    // Accept the common human forms first.
    let lowered = host.trim().to_ascii_lowercase();
    if lowered == "localhost" || lowered == "::1" {
        return true;
    }
    // Strip optional brackets around IPv6 literals.
    let stripped = lowered.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = stripped.parse::<IpAddr>() {
        return ip.is_loopback();
    }
    false
}
