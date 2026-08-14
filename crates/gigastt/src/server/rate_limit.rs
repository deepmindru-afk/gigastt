//! Per-IP token-bucket rate limiter.
//!
//! Replaces the `tower_governor` crate (which pulled `governor`, `dashmap`,
//! `quanta`, `parking_lot`, and `forwarded-header-value`) with a focused
//! ~150-line implementation tailored to gigastt's single middleware hook.
//!
//! Refill formula: `refill_per_ms = rpm / 60_000.0`, so
//! `--rate-limit-per-minute 30` allows one token every 2 s with a configurable
//! burst. When the bucket is empty the caller gets a 429 with `Retry-After: 60`.
//!
//! IP extraction mirrors the old `SmartIpKeyExtractor`:
//! - first hop of `X-Forwarded-For` (trimmed), then
//! - `X-Real-IP`, then
//! - `ConnectInfo<SocketAddr>::ip()`.
//!
//! The rate-limiter & X-Forwarded-For trust boundary is documented in
//! `docs/deployment.md` — the reverse proxy must **overwrite** the
//! header with the real peer address, never append.

use axum::extract::{ConnectInfo, Request};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Json, Response};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Requests per minute. Invariant: `0 < rpm <= MAX_RPM`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rpm(u32);

impl Rpm {
    /// { rpm > 0 && rpm <= MAX_RPM }
    /// fn new(rpm: u32) -> Result<Rpm, String>
    /// { ret.as_ref().map(|r| r.0 > 0 && r.0 <= MAX_RPM).unwrap_or(true) }
    pub fn new(rpm: u32) -> Result<Self, String> {
        if rpm == 0 {
            return Err("rpm must be > 0".into());
        }
        if rpm > MAX_RPM {
            return Err(format!("rpm must be <= {MAX_RPM}"));
        }
        Ok(Rpm(rpm))
    }

    /// { true }
    /// fn get(self) -> u32
    /// { ret > 0 }
    pub fn get(self) -> u32 {
        self.0
    }

    /// Construct without validation. Caller must guarantee `0 < rpm <= MAX_RPM`.
    ///
    /// { rpm > 0 && rpm <= MAX_RPM }
    /// fn from_raw(rpm: u32) -> Rpm
    /// { ret.0 > 0 && ret.0 <= MAX_RPM }
    pub(crate) fn from_raw(rpm: u32) -> Self {
        debug_assert!(rpm > 0 && rpm <= MAX_RPM);
        Rpm(rpm)
    }
}

/// Burst size (max concurrent tokens). Invariant: `burst >= 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Burst(u32);

impl Burst {
    /// { burst >= 1 }
    /// fn new(burst: u32) -> Result<Burst, String>
    /// { ret.as_ref().map(|b| b.0 >= 1).unwrap_or(true) }
    pub fn new(burst: u32) -> Result<Self, String> {
        if burst < 1 {
            return Err("burst must be >= 1".into());
        }
        Ok(Burst(burst))
    }

    /// { true }
    /// fn get(self) -> u32
    /// { ret >= 1 }
    pub fn get(self) -> u32 {
        self.0
    }

    /// Construct without validation. Caller must guarantee `burst >= 1`.
    ///
    /// { burst >= 1 }
    /// fn from_raw(burst: u32) -> Burst
    /// { ret.0 >= 1 }
    pub(crate) fn from_raw(burst: u32) -> Self {
        debug_assert!(burst >= 1);
        Burst(burst)
    }
}

/// Single per-IP bucket. Fractional tokens (`f64`) let us express arbitrary
/// refill rates below 1 token/ms without losing precision — matches the
/// `per_millisecond(60_000 / rpm)` semantics of `tower_governor` 0.7.
#[derive(Debug)]
pub struct TokenBucket {
    capacity: f64,
    refill_per_ms: f64,
    tokens: f64,
    last_refill: Instant,
    /// Wall-clock timestamp of the last refill (milliseconds since the epoch)
    /// used by `RateLimiter::evict_stale` to bound memory. Stored as a plain
    /// `u64` rather than a second `Instant` because eviction is driven off a
    /// single global "now" without needing per-bucket monotonic comparison.
    last_seen_ms: u64,
}

impl TokenBucket {
    /// { refill_per_ms >= 0.0 }
    /// fn new(capacity: u32, refill_per_ms: f64, now: Instant, now_ms: u64) -> TokenBucket
    /// { ret.tokens == ret.capacity && ret.capacity == capacity as f64 && ret.refill_per_ms == refill_per_ms }
    pub fn new(capacity: u32, refill_per_ms: f64, now: Instant, now_ms: u64) -> Self {
        Self {
            capacity: capacity as f64,
            refill_per_ms,
            tokens: capacity as f64,
            last_refill: now,
            last_seen_ms: now_ms,
        }
    }

    /// Refill the bucket based on elapsed time and try to consume one token.
    /// Returns `true` when the request is allowed.
    ///
    /// { refill_per_ms >= 0.0 }
    /// fn try_consume(&mut self, now: Instant, now_ms: u64) -> bool
    /// { ret == (self.tokens >= 1.0) }
    pub fn try_consume(&mut self, now: Instant, now_ms: u64) -> bool {
        let elapsed_ms = now
            .saturating_duration_since(self.last_refill)
            .as_secs_f64()
            * 1000.0;
        if elapsed_ms > 0.0 {
            self.tokens = (self.tokens + elapsed_ms * self.refill_per_ms).min(self.capacity);
            self.last_refill = now;
        }
        self.last_seen_ms = now_ms;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Upper bound on `rpm` accepted by [`RateLimiter::new`]. Beyond this the
/// 1 ms refill interval would truncate to zero and the bucket would saturate.
pub const MAX_RPM: u32 = 60_000;

/// Hard cap on the number of per-IP buckets to prevent unbounded memory
/// growth under a rotating-IP botnet. When the cap is hit the oldest bucket
/// is evicted before the new one is inserted.
const MAX_BUCKETS: usize = 100_000;

/// Per-IP buckets behind a single mutex. The critical section is a handful of
/// float operations on one bucket, so sharding would buy nothing measurable.
pub struct RateLimiter {
    buckets: Mutex<HashMap<IpAddr, TokenBucket>>,
    capacity: Burst,
    refill_per_ms: f64,
    effective_rpm: Rpm,
    max_entries: usize,
}

impl RateLimiter {
    /// Construct from the same `(rpm, burst)` pair the CLI exposes.
    ///
    /// `rpm` is clamped to the [`MAX_RPM`] maximum (the
    /// interval hits 1 ms precision there; anything higher would truncate to
    /// zero and saturate the bucket). Emits a `warn!` once when clamping.
    ///
    /// { rpm > 0 }
    /// fn new(rpm: u32, burst: u32) -> RateLimiter
    /// { ret.effective_rpm.0 > 0 && ret.capacity.0 >= 1 }
    pub fn new(rpm: u32, burst: u32) -> Self {
        if rpm > MAX_RPM {
            tracing::warn!(
                rpm,
                max_rpm = MAX_RPM,
                "rate_limit_per_minute exceeds {MAX_RPM}; clamped to {MAX_RPM} (1 ms minimum interval)"
            );
        }
        let effective_rpm = rpm.clamp(1, MAX_RPM);
        let refill_per_ms = effective_rpm as f64 / 60_000.0;
        Self {
            buckets: Mutex::new(HashMap::new()),
            capacity: Burst::from_raw(burst.max(1)),
            refill_per_ms,
            effective_rpm: Rpm::from_raw(effective_rpm),
            max_entries: MAX_BUCKETS,
        }
    }

    /// Minimum interval between successful requests for the effective (clamped)
    /// rpm, in milliseconds. Used for the startup log line.
    ///
    /// ```text
    /// { self.effective_rpm.0 > 0 }
    /// fn interval_ms(&self) -> u64
    /// { ret >= 1 }
    /// ```
    pub fn interval_ms(&self) -> u64 {
        (60_000u64 / self.effective_rpm.0.max(1) as u64).max(1)
    }

    /// Check a request from `ip`. Returns `true` when the bucket had a token,
    /// `false` when the caller should be 429'd. Inserts a fresh bucket for
    /// first-time callers.
    ///
    /// ```text
    /// { self.capacity.0 >= 1 }
    /// fn check(&self, ip: IpAddr) -> bool
    /// { ret == (self.buckets[&ip].tokens >= 1.0 after refill) }
    /// ```
    pub fn check(&self, ip: IpAddr) -> bool {
        let now = Instant::now();
        let now_ms = unix_ms();

        let mut buckets = self.buckets.lock();

        // Fast path: existing bucket.
        if let Some(bucket) = buckets.get_mut(&ip) {
            return bucket.try_consume(now, now_ms);
        }

        // Slow path: new IP. Evict the stalest of a bounded sample if at capacity.
        if buckets.len() >= self.max_entries {
            Self::evict_one(&mut buckets);
        }

        let mut bucket = TokenBucket::new(self.capacity.0, self.refill_per_ms, now, now_ms);
        let allowed = bucket.try_consume(now, now_ms);
        buckets.insert(ip, bucket);
        allowed
    }

    /// Evict the stalest bucket from a bounded sample of 100 entries. Takes the
    /// already-held map: `check` calls this while holding the lock, and
    /// re-locking would deadlock.
    fn evict_one(buckets: &mut HashMap<IpAddr, TokenBucket>) {
        let oldest = buckets
            .iter()
            .take(100)
            .min_by_key(|(_, bucket)| bucket.last_seen_ms)
            .map(|(ip, _)| *ip);
        if let Some(key) = oldest {
            buckets.remove(&key);
        }
    }

    /// Drop buckets whose `last_seen_ms` is older than `older_than`. Called
    /// from the background tokio task in `run_with_config` to bound memory
    /// under sustained single-visitor traffic.
    ///
    /// ```text
    /// { true }
    /// fn evict_stale(&self, older_than: Duration)
    /// { self.buckets.len() <= old(self.buckets.len()) }
    /// ```
    pub fn evict_stale(&self, older_than: Duration) {
        let cutoff = unix_ms().saturating_sub(older_than.as_millis() as u64);
        self.buckets
            .lock()
            .retain(|_, bucket| bucket.last_seen_ms >= cutoff);
    }

    #[cfg(test)]
    #[allow(clippy::len_without_is_empty)]
    /// ```text
    /// { true }
    /// fn len(&self) -> usize
    /// { ret == self.buckets.len() }
    /// ```
    pub fn len(&self) -> usize {
        self.buckets.lock().len()
    }
}

/// ```text
/// { true }
/// fn unix_ms() -> u64
/// { ret >= 0 }
/// ```
fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Extract the client IP from `X-Forwarded-For` (first hop), `X-Real-IP`, or
/// the TCP `ConnectInfo`, in that order. Mirrors `SmartIpKeyExtractor` from
/// `tower_governor`. The proxy must overwrite (not append) `X-Forwarded-For`
/// — see `docs/deployment.md`.
///
/// When `trust_proxy` is `false`, forwarded headers are ignored entirely and
/// only `ConnectInfo` is used. When `true`, the headers are consulted only
/// if the direct peer IP is loopback or a trusted proxy hop (RFC1918 /
/// IPv6 unique-local / IPv6 link-local).
///
/// ```text
/// { true }
/// fn extract_client_ip(req: &Request, trust_proxy: bool) -> Option<IpAddr>
/// { ret.is_some() == (!trust_proxy || req.extensions().get::<ConnectInfo<SocketAddr>>().is_some() || has_forwarded_headers) }
/// ```
pub fn extract_client_ip(req: &Request, trust_proxy: bool) -> Option<IpAddr> {
    let direct_ip = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip());

    if !trust_proxy {
        return direct_ip;
    }

    // Trust proxy mode: only read forwarded headers when the direct peer
    // is a known private proxy subnet.
    if let Some(connect_ip) = direct_ip
        && !connect_ip.is_loopback()
        && !is_trusted_proxy_hop(connect_ip)
    {
        return Some(connect_ip);
    }

    let headers = req.headers();
    if let Some(value) = headers.get("x-forwarded-for")
        && let Ok(s) = value.to_str()
    {
        let first = s.split(',').next().unwrap_or("").trim();
        if let Ok(ip) = first.parse::<IpAddr>() {
            return Some(ip);
        }
    }
    if let Some(value) = headers.get("x-real-ip")
        && let Ok(s) = value.to_str()
        && let Ok(ip) = s.trim().parse::<IpAddr>()
    {
        return Some(ip);
    }
    direct_ip
}

/// Return true for addresses that may sit in front of this process as a
/// reverse proxy: IPv4 RFC1918 (10/8, 172.16/12, 192.168/16), IPv6 unique
/// local (`fc00::/7`), and IPv6 link-local (`fe80::/10`). Loopback is
/// handled separately by the caller.
fn is_trusted_proxy_hop(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            o[0] == 10 || (o[0] == 172 && (o[1] & 0xF0) == 16) || (o[0] == 192 && o[1] == 168)
        }
        IpAddr::V6(v6) => v6.is_unique_local() || v6.is_unicast_link_local(),
    }
}

/// Build a per-request middleware that consults `limiter` before forwarding
/// to the next layer. Emits the same `429 Too Many Requests` +
/// `Retry-After: 60` contract the previous `tower_governor` layer produced.
///
/// ```text
/// { true }
/// async fn rate_limit_middleware(limiter: Arc<RateLimiter>, trust_proxy: bool, req: Request, next: Next) -> Response
/// { ret.status() == 429 || ret.status() == next.run(req).await.status() }
/// ```
/// Paths exempt from per-IP rate limiting. These are either handled by a
/// separate mechanism (loopback-only enforcement in the handler itself) or
/// are already outside the main request budget.
fn is_rate_limit_exempt(path: &str) -> bool {
    path == "/v1/admin/reload"
}

pub async fn rate_limit_middleware(
    limiter: Arc<RateLimiter>,
    trust_proxy: bool,
    metrics: Option<Arc<super::metrics::MetricsRegistry>>,
    req: Request,
    next: Next,
) -> Response {
    if is_rate_limit_exempt(req.uri().path()) {
        return next.run(req).await;
    }
    let Some(ip) = extract_client_ip(&req, trust_proxy) else {
        tracing::debug!("rate limit: could not determine client IP");
        return next.run(req).await;
    };
    if limiter.check(ip) {
        next.run(req).await
    } else {
        tracing::debug!(client_ip = %ip, "rate limit rejected request");
        if let Some(ref reg) = metrics {
            reg.counter_inc("gigastt_rate_limit_rejections_total", &[], 1);
        }
        let retry_after_ms = limiter.interval_ms();
        let retry_after_secs = retry_after_ms.div_ceil(1000).max(1);
        (
            StatusCode::TOO_MANY_REQUESTS,
            [(
                axum::http::header::RETRY_AFTER,
                retry_after_secs.to_string(),
            )],
            Json(serde_json::json!({
                "error": "Too many requests",
                "code": "rate_limited",
                "retry_after_ms": retry_after_ms,
            })),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests;
