use super::*;

fn registry() -> MetricsRegistry {
    let r = MetricsRegistry::new();
    r.register_counter(
        "gigastt_http_requests_total",
        "Total HTTP requests processed",
    );
    r.register_histogram(
        "gigastt_http_request_duration_seconds",
        "HTTP request duration",
        DEFAULT_BUCKETS,
    );
    r
}

#[test]
fn test_register_server_metrics_exports_expected_families() {
    let text = register_server_metrics().render_prometheus();
    assert!(text.contains("# TYPE gigastt_http_requests_total counter"));
    assert!(text.contains("# TYPE gigastt_http_request_duration_seconds histogram"));
    assert!(text.contains("# TYPE gigastt_pool_available gauge"));
    assert!(text.contains("# TYPE gigastt_ws_active_connections gauge"));
    assert!(text.contains("# TYPE gigastt_inference_timeouts_total counter"));
}

#[test]
fn test_render_empty_registry() {
    let r = MetricsRegistry::new();
    assert_eq!(r.render_prometheus(), "");
}

#[test]
fn test_counter_increment_and_render() {
    let r = registry();
    r.counter_inc(
        "gigastt_http_requests_total",
        &[("method", "GET"), ("path", "/health"), ("status", "200")],
        1,
    );
    r.counter_inc(
        "gigastt_http_requests_total",
        &[("method", "GET"), ("path", "/health"), ("status", "200")],
        2,
    );
    let text = r.render_prometheus();
    assert!(text.contains("# HELP gigastt_http_requests_total Total HTTP requests processed"));
    assert!(text.contains("# TYPE gigastt_http_requests_total counter"));
    assert!(
        text.contains(
            "gigastt_http_requests_total{method=\"GET\",path=\"/health\",status=\"200\"} 3"
        )
    );
}

#[test]
fn test_histogram_bucket_cumulative() {
    let r = registry();
    let labels = [("method", "GET")];
    for v in [0.001, 0.03, 0.3, 1.5] {
        r.histogram_record("gigastt_http_request_duration_seconds", &labels, v);
    }
    let text = r.render_prometheus();
    // 0.001 ≤ 0.005 → contributes to every bucket including 0.005+
    // 0.03  ≤ 0.05  → contributes to 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    // 0.3   ≤ 0.5   → contributes to 0.5, 1.0, 2.5, 5.0, 10.0
    // 1.5   ≤ 2.5   → contributes to 2.5, 5.0, 10.0
    assert!(
        text.contains(
            "gigastt_http_request_duration_seconds_bucket{method=\"GET\",le=\"0.005\"} 1"
        )
    );
    assert!(
        text.contains("gigastt_http_request_duration_seconds_bucket{method=\"GET\",le=\"0.05\"} 2")
    );
    assert!(
        text.contains("gigastt_http_request_duration_seconds_bucket{method=\"GET\",le=\"0.5\"} 3")
    );
    assert!(
        text.contains("gigastt_http_request_duration_seconds_bucket{method=\"GET\",le=\"+Inf\"} 4")
    );
    assert!(text.contains("gigastt_http_request_duration_seconds_count{method=\"GET\"} 4"));
}

#[test]
fn test_label_ordering_stable() {
    let r = MetricsRegistry::new();
    r.counter_inc("c", &[("b", "1"), ("a", "2")], 1);
    r.counter_inc("c", &[("a", "2"), ("b", "1")], 4);
    let text = r.render_prometheus();
    // Same counter despite different insert order — totals to 5.
    assert!(text.contains("c{a=\"2\",b=\"1\"} 5"));
}

#[test]
fn test_label_escaping() {
    let r = MetricsRegistry::new();
    r.counter_inc("c", &[("l", "a\"b\\c\nd")], 1);
    let text = r.render_prometheus();
    assert!(
        text.contains("c{l=\"a\\\"b\\\\c\\nd\"} 1"),
        "escape failed: {text}"
    );
}

#[test]
fn test_empty_labels_render() {
    let r = MetricsRegistry::new();
    r.counter_inc("c", &[], 7);
    let text = r.render_prometheus();
    assert!(text.contains("c 7"));
}

#[test]
fn test_gauge_set_inc_and_render() {
    let r = MetricsRegistry::new();
    r.register_gauge("g", "A gauge");
    r.gauge_set("g", &[], 5);
    let text = r.render_prometheus();
    assert!(text.contains("# HELP g A gauge"));
    assert!(text.contains("# TYPE g gauge"));
    assert!(text.contains("g 5"));

    r.gauge_inc("g", &[], -2);
    let text = r.render_prometheus();
    assert!(text.contains("g 3"));
}

#[test]
fn test_histogram_sum_tracks_observations() {
    let r = MetricsRegistry::new();
    r.register_histogram("h", "H", &[1.0, 2.0]);
    r.histogram_record("h", &[], 0.5);
    r.histogram_record("h", &[], 1.5);
    r.histogram_record("h", &[], 2.5);
    let text = r.render_prometheus();
    assert!(text.contains("h_sum 4.5"));
    assert!(text.contains("h_count 3"));
}

#[test]
fn test_histogram_bucket_resize_on_reregister() {
    let r = MetricsRegistry::new();
    // First registration with 2 buckets
    r.register_histogram("h", "H", &[1.0, 2.0]);
    r.histogram_record("h", &[], 0.5);
    // Re-register with 4 buckets — existing series should resize
    r.register_histogram("h", "H", &[0.5, 1.0, 2.0, 5.0]);
    r.histogram_record("h", &[], 3.0);
    let text = r.render_prometheus();
    // Both observations should be counted
    assert!(text.contains("h_count 2"));
    assert!(text.contains("h_sum 3.5"));
    // 3.0 falls into the new 5.0 bucket (counts are non-cumulative in this impl)
    assert!(text.contains("h_bucket{le=\"5\"} 1"));
    assert!(text.contains("h_bucket{le=\"+Inf\"} 2"));
}

#[test]
fn test_fmt_f64_prom_special_values() {
    assert_eq!(fmt_f64_prom(f64::INFINITY), "+Inf");
    assert_eq!(fmt_f64_prom(f64::NEG_INFINITY), "-Inf");
    assert_eq!(fmt_f64_prom(f64::NAN), "NaN");
    assert_eq!(fmt_f64_prom(std::f64::consts::PI), "3.141592653589793");
}

#[test]
fn test_trim_outer_braces() {
    assert_eq!(trim_outer_braces(""), "");
    assert_eq!(trim_outer_braces("{a=\"b\"}"), "a=\"b\"");
    assert_eq!(trim_outer_braces("abc"), "abc");
    assert_eq!(trim_outer_braces("{}"), "");
}
