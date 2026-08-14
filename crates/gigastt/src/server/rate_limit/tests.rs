use super::*;
use axum::body::Body;
use axum::http::{HeaderValue, Request as HttpRequest};
use std::net::{Ipv4Addr, Ipv6Addr};

#[test]
fn test_token_bucket_allows_within_capacity() {
    // Burst = 5, refill irrelevant for this test — we consume under the
    // cap without waiting, every call must succeed.
    let now = Instant::now();
    let mut bucket = TokenBucket::new(5, 0.0, now, unix_ms());
    for i in 0..5 {
        assert!(bucket.try_consume(now, unix_ms()), "call {i} must succeed");
    }
    // 6th consumption without refill must fail — bucket is empty.
    assert!(
        !bucket.try_consume(now, unix_ms()),
        "6th call must be rate-limited"
    );
}

#[test]
fn test_token_bucket_refills_over_time() {
    // Refill rate = 1 token / ms. Drain the capacity, advance the clock,
    // verify the bucket refills.
    let start = Instant::now();
    let mut bucket = TokenBucket::new(2, 1.0, start, unix_ms());
    assert!(bucket.try_consume(start, unix_ms()));
    assert!(bucket.try_consume(start, unix_ms()));
    assert!(
        !bucket.try_consume(start, unix_ms()),
        "should be drained after 2 consumes"
    );
    let later = start + Duration::from_millis(3);
    assert!(
        bucket.try_consume(later, unix_ms()),
        "should refill after 3 ms"
    );
}

#[test]
fn test_rate_limiter_per_ip_isolation() {
    // Two IPs each with a burst of 1 — one consuming must not drain the
    // other.
    let limiter = RateLimiter::new(1, 1);
    let a: IpAddr = "10.0.0.1".parse().unwrap();
    let b: IpAddr = "10.0.0.2".parse().unwrap();
    assert!(limiter.check(a), "A first call allowed");
    assert!(
        limiter.check(b),
        "B first call allowed (independent bucket)"
    );
    assert!(!limiter.check(a), "A second call rate-limited");
    assert!(!limiter.check(b), "B second call rate-limited");
}

#[test]
fn test_rate_limiter_refill_formula_matches_v1_06() {
    // Mirrors `test_rate_limit_interval_formula` in src/server/mod.rs:
    // `refill_per_ms = rpm / 60_000` must equal `1 / interval_ms` for every
    // `interval_ms = (60_000 / rpm).max(1)` pairing. Concretely: draining
    // the bucket then waiting `interval_ms` must refill exactly 1 token.
    for &rpm in &[1u32, 10, 30, 60, 600, 60_000] {
        let limiter = RateLimiter::new(rpm, 1);
        let ip: IpAddr = "10.0.0.3".parse().unwrap();
        assert!(limiter.check(ip), "rpm={rpm}: initial burst allowed");
        assert!(
            !limiter.check(ip),
            "rpm={rpm}: second immediate call blocked"
        );
        // Advance the bucket's last_refill manually by draining, waiting,
        // and re-checking. Real tests use sleeps; here we inject the
        // refill via `try_consume` with a later instant.
        let mut buckets = limiter.buckets.lock();
        let bucket = buckets.get_mut(&ip).expect("bucket exists");
        let interval_ms = (60_000u64 / rpm as u64).max(1);
        let later = bucket.last_refill + Duration::from_millis(interval_ms);
        // `later - last_refill = interval_ms`, so the refill should be
        // `elapsed_ms * refill_per_ms = interval_ms * (rpm / 60_000) >= 1`.
        assert!(
            bucket.try_consume(later, unix_ms()),
            "rpm={rpm}: 1 token must refill after {interval_ms} ms",
        );
        drop(buckets);
    }
}

#[test]
fn test_extract_ip_prefers_forwarded_for_when_trusted() {
    // First hop (trimmed) wins over X-Real-IP and ConnectInfo.
    let mut req = HttpRequest::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    req.headers_mut().insert(
        "x-forwarded-for",
        HeaderValue::from_static("203.0.113.42 , 10.0.0.1"),
    );
    req.headers_mut()
        .insert("x-real-ip", HeaderValue::from_static("198.51.100.7"));
    req.extensions_mut().insert(ConnectInfo(SocketAddr::new(
        IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
        12345,
    )));
    let got = extract_client_ip(&req, true).expect("XFF must be parsed");
    assert_eq!(got, "203.0.113.42".parse::<IpAddr>().unwrap());
}

#[test]
fn test_extract_ip_ignores_forwarded_when_not_trusted() {
    // trust_proxy=false: headers ignored, ConnectInfo wins.
    let mut req = HttpRequest::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    req.headers_mut().insert(
        "x-forwarded-for",
        HeaderValue::from_static("203.0.113.42 , 10.0.0.1"),
    );
    req.extensions_mut().insert(ConnectInfo(SocketAddr::new(
        IpAddr::V4(Ipv4Addr::new(198, 51, 100, 7)),
        12345,
    )));
    let got = extract_client_ip(&req, false).expect("ConnectInfo must be used");
    assert_eq!(got, IpAddr::V4(Ipv4Addr::new(198, 51, 100, 7)));
}

#[test]
fn test_extract_ip_falls_back_to_connect_info() {
    // No proxy headers — must fall back to the ConnectInfo peer.
    let mut req = HttpRequest::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(SocketAddr::new(
        IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
        55555,
    )));
    let got = extract_client_ip(&req, true).expect("ConnectInfo fallback");
    assert_eq!(got, IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));
}

#[test]
fn test_extract_ip_uses_real_ip_when_forwarded_for_garbage() {
    // X-Forwarded-For is unparseable; X-Real-IP wins.
    let mut req = HttpRequest::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    req.headers_mut()
        .insert("x-forwarded-for", HeaderValue::from_static("not-an-ip"));
    req.headers_mut()
        .insert("x-real-ip", HeaderValue::from_static("198.51.100.7"));
    req.extensions_mut().insert(ConnectInfo(SocketAddr::new(
        IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
        12345,
    )));
    let got = extract_client_ip(&req, true).expect("X-Real-IP fallback");
    assert_eq!(got, "198.51.100.7".parse::<IpAddr>().unwrap());
}

#[test]
fn test_extract_ip_trusts_ipv6_ula_proxy() {
    let mut req = HttpRequest::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    req.headers_mut()
        .insert("x-forwarded-for", HeaderValue::from_static("203.0.113.42"));
    req.extensions_mut().insert(ConnectInfo(SocketAddr::new(
        IpAddr::V6(Ipv6Addr::new(0xfd00, 0, 0, 0, 0, 0, 0, 1)),
        12345,
    )));
    let got = extract_client_ip(&req, true).expect("ULA proxy must trust XFF");
    assert_eq!(got, "203.0.113.42".parse::<IpAddr>().unwrap());
}

#[test]
fn test_extract_ip_ignores_forwarded_from_public_ipv6() {
    let mut req = HttpRequest::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    req.headers_mut()
        .insert("x-forwarded-for", HeaderValue::from_static("203.0.113.42"));
    let public = Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1);
    req.extensions_mut()
        .insert(ConnectInfo(SocketAddr::new(IpAddr::V6(public), 12345)));
    let got = extract_client_ip(&req, true).expect("public IPv6 is not a trusted hop");
    assert_eq!(got, IpAddr::V6(public));
}

#[test]
fn test_extract_ip_skips_headers_when_direct_peer_is_public() {
    // trust_proxy=true but ConnectInfo is a public IP → ignore headers.
    let mut req = HttpRequest::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    req.headers_mut()
        .insert("x-forwarded-for", HeaderValue::from_static("203.0.113.42"));
    req.extensions_mut().insert(ConnectInfo(SocketAddr::new(
        IpAddr::V4(Ipv4Addr::new(198, 51, 100, 7)),
        12345,
    )));
    let got = extract_client_ip(&req, true).expect("ConnectInfo used");
    assert_eq!(got, IpAddr::V4(Ipv4Addr::new(198, 51, 100, 7)));
}

#[test]
fn test_eviction_removes_stale() {
    // Populate the limiter with two IPs, artificially age one, confirm
    // eviction drops only the stale bucket.
    let limiter = RateLimiter::new(60, 1);
    let fresh: IpAddr = "10.0.0.4".parse().unwrap();
    let stale: IpAddr = "10.0.0.5".parse().unwrap();
    assert!(limiter.check(fresh));
    assert!(limiter.check(stale));
    // Hand-roll an "old" last_seen_ms on the stale bucket.
    {
        let mut buckets = limiter.buckets.lock();
        let bucket = buckets.get_mut(&stale).expect("stale bucket");
        bucket.last_seen_ms = unix_ms().saturating_sub(10 * 60_000); // 10 min old
    }
    limiter.evict_stale(Duration::from_secs(60));
    assert_eq!(limiter.len(), 1, "stale bucket should be evicted");
    assert!(
        limiter.buckets.lock().contains_key(&fresh),
        "fresh bucket must survive eviction"
    );
}

#[test]
fn test_rate_limiter_random_eviction_at_cap() {
    // Create a tiny-capacity limiter so eviction fires immediately.
    let limiter = RateLimiter::new(60, 1);
    let mut ips = Vec::new();
    for i in 0..=limiter.max_entries {
        let ip = IpAddr::V4(Ipv4Addr::from(i as u32));
        ips.push(ip);
        assert!(limiter.check(ip), "IP {i} must be allowed on first visit");
    }
    // At this point we are at (or slightly over) capacity.  Adding one
    // more distinct IP must succeed because random eviction makes room.
    let extra = IpAddr::V4(Ipv4Addr::new(10, 1, 0, 1));
    limiter.check(extra);
    // Total count must stay bounded by max_entries.
    assert!(
        limiter.len() <= limiter.max_entries,
        "len={} must not exceed cap={}",
        limiter.len(),
        limiter.max_entries
    );
}

#[test]
fn test_admin_reload_is_rate_limit_exempt() {
    assert!(
        is_rate_limit_exempt("/v1/admin/reload"),
        "/v1/admin/reload must be exempt from rate limiting"
    );
    assert!(
        !is_rate_limit_exempt("/v1/transcribe"),
        "/v1/transcribe must NOT be exempt"
    );
    assert!(
        !is_rate_limit_exempt("/health"),
        "/health must NOT be exempt (it is outside the rate-limit layer entirely)"
    );
}

#[test]
fn test_rpm_new_rejects_zero() {
    assert!(Rpm::new(0).is_err());
}

#[test]
fn test_rpm_new_rejects_too_high() {
    assert!(Rpm::new(MAX_RPM + 1).is_err());
}

#[test]
fn test_rpm_new_accepts_valid() {
    let r = Rpm::new(30).unwrap();
    assert_eq!(r.get(), 30);
}

#[test]
fn test_burst_new_rejects_zero() {
    assert!(Burst::new(0).is_err());
}

#[test]
fn test_burst_new_accepts_valid() {
    let b = Burst::new(5).unwrap();
    assert_eq!(b.get(), 5);
}
