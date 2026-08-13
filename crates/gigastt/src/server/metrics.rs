//! Minimal Prometheus text-exposition registry (replaces
//! `metrics-exporter-prometheus`).
//!
//! We only need counters and histograms, keyed by a small set of fixed
//! labels. Rolling our own drops ~40 transitive crates from `Cargo.lock`
//! (the `metrics`/`metrics-util`/`indexmap`/`atomic-waker`/…  stack) and
//! keeps the `/metrics` contract entirely in-tree. The emitted text
//! matches the Prometheus 0.0.4 exposition format documented at
//! <https://prometheus.io/docs/instrumenting/exposition_formats/>.
//!
//! ## Concurrency
//! `RwLock<HashMap<..>>` is fine for our workload — counters/histograms
//! are hit on every HTTP request (per-handler middleware), but the scrape
//! endpoint is typically polled every 15 s so reader contention is low.
//! When we need lock-free update later, swap `RwLock` for a sharded map.
//!
//! ## Default histogram buckets
//! Matches `metrics-exporter-prometheus`'s defaults, which themselves come
//! from the Prometheus Go client library. Tweakable per-metric via
//! `register_histogram_with_buckets`.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;

/// Default histogram bucket bounds (seconds-scaled). Upper bound `f64::INFINITY`
/// is appended implicitly when rendering — consumers do not need to supply it.
pub const DEFAULT_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// Sorted label set keyed by ASCII name. Sorting keeps the serialised
/// label string stable regardless of the insertion order so the same
/// counter + label combination always maps to the same storage slot.
pub type Labels = Vec<(String, String)>;

fn sort_labels(labels: &[(&str, &str)]) -> Labels {
    let mut labels: Labels = labels
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    labels.sort_by(|a, b| a.0.cmp(&b.0));
    labels
}

fn format_labels(labels: &Labels) -> String {
    if labels.is_empty() {
        return String::new();
    }
    let mut out = String::from("{");
    for (i, (k, v)) in labels.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(k);
        out.push_str("=\"");
        // Escape the label value — per the Prometheus text format, the
        // characters `\`, `"`, and `\n` must be escaped. Rare in practice
        // but cheap to do and prevents crafted label values from breaking
        // the exposition output.
        for ch in v.chars() {
            match ch {
                '\\' => out.push_str("\\\\"),
                '"' => out.push_str("\\\""),
                '\n' => out.push_str("\\n"),
                c => out.push(c),
            }
        }
        out.push('"');
    }
    out.push('}');
    out
}

#[derive(Debug, Default)]
struct CounterFamily {
    help: String,
    values: HashMap<Labels, u64>,
}

#[derive(Debug, Default)]
struct GaugeFamily {
    help: String,
    values: HashMap<Labels, i64>,
}

#[derive(Debug, Default)]
struct HistogramFamily {
    help: String,
    buckets: Vec<f64>,
    series: HashMap<Labels, HistogramSeries>,
}

#[derive(Debug, Default, Clone)]
struct HistogramSeries {
    /// Cumulative bucket counts; index `i` is observations ≤ `buckets[i]`.
    /// Trailing `+Inf` bucket is the grand total (`count`), not stored here.
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

/// Prometheus-compatible registry used by the server. Typically wrapped in
/// an `Arc` and stashed on `AppState` so every handler can record into it.
#[derive(Debug, Default)]
pub struct MetricsRegistry {
    counters: RwLock<HashMap<Arc<str>, CounterFamily>>,
    gauges: RwLock<HashMap<Arc<str>, GaugeFamily>>,
    histograms: RwLock<HashMap<Arc<str>, HistogramFamily>>,
}

impl MetricsRegistry {
    /// Create an empty registry. Families are declared lazily on first use
    /// via `counter_inc` / `histogram_record` — separate `register_*`
    /// methods exist for setting help text ahead of time.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the `# HELP` text for a gauge family.
    pub fn register_gauge(&self, name: &str, help: &str) {
        let mut map = self.gauges.write();
        map.entry(Arc::from(name)).or_default().help = help.to_string();
    }

    /// Set a gauge to an absolute value.
    pub fn gauge_set(&self, name: &str, labels: &[(&str, &str)], value: i64) {
        let labels = sort_labels(labels);
        let mut map = self.gauges.write();
        let family = map.entry(Arc::from(name)).or_default();
        *family.values.entry(labels).or_insert(0) = value;
    }

    /// Increment a gauge by `delta` (may be negative).
    pub fn gauge_inc(&self, name: &str, labels: &[(&str, &str)], delta: i64) {
        let labels = sort_labels(labels);
        let mut map = self.gauges.write();
        let family = map.entry(Arc::from(name)).or_default();
        *family.values.entry(labels).or_insert(0) += delta;
    }

    /// Set the `# HELP` text for a counter family. Called during startup;
    /// overwrites any previously registered help text for the same name.
    pub fn register_counter(&self, name: &str, help: &str) {
        let mut map = self.counters.write();
        map.entry(Arc::from(name)).or_default().help = help.to_string();
    }

    /// Set the `# HELP` text and bucket bounds for a histogram family.
    /// Buckets are sorted and deduplicated; callers may pass
    /// [`DEFAULT_BUCKETS`] for the Prometheus client default.
    pub fn register_histogram(&self, name: &str, help: &str, buckets: &[f64]) {
        let mut normalised: Vec<f64> = buckets.to_vec();
        normalised.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        normalised.dedup();
        let mut map = self.histograms.write();
        let family = map.entry(Arc::from(name)).or_default();
        family.help = help.to_string();
        family.buckets = normalised;
    }

    /// Increment a counter. Lazily creates the family if it didn't exist.
    pub fn counter_inc(&self, name: &str, labels: &[(&str, &str)], delta: u64) {
        let labels = sort_labels(labels);
        let mut map = self.counters.write();
        let family = map.entry(Arc::from(name)).or_default();
        *family.values.entry(labels).or_insert(0) += delta;
    }

    /// Record one observation into a histogram. Lazily creates the family
    /// with [`DEFAULT_BUCKETS`] if it didn't exist.
    pub fn histogram_record(&self, name: &str, labels: &[(&str, &str)], value: f64) {
        let labels = sort_labels(labels);
        let mut map = self.histograms.write();
        let family = map.entry(Arc::from(name)).or_default();
        if family.buckets.is_empty() {
            family.buckets = DEFAULT_BUCKETS.to_vec();
        }
        let series = family
            .series
            .entry(labels)
            .or_insert_with(|| HistogramSeries {
                counts: vec![0; family.buckets.len()],
                sum: 0.0,
                count: 0,
            });
        // Keep the cumulative-counts vector in sync with the (possibly
        // re-registered) bucket list. Extending with zeros is correct
        // because extra buckets haven't seen observations yet.
        if series.counts.len() < family.buckets.len() {
            series.counts.resize(family.buckets.len(), 0);
        }
        for (i, &upper) in family.buckets.iter().enumerate() {
            if value <= upper {
                series.counts[i] += 1;
            }
        }
        series.sum += value;
        series.count += 1;
    }

    /// Render the current snapshot as Prometheus text. Formatting follows
    /// the `0.0.4; charset=utf-8` content type: `# HELP` and `# TYPE`
    /// comments per family, then one sample per (name, labels) pair.
    pub fn render_prometheus(&self) -> String {
        let mut out = String::new();

        // Counters first — stable alphabetical order for reproducible
        // scrape output across invocations.
        let counters = self.counters.read();
        let mut names: Vec<&Arc<str>> = counters.keys().collect();
        names.sort();
        for name in names {
            let family = &counters[name];
            if !family.help.is_empty() {
                let _ = writeln!(out, "# HELP {name} {}", family.help);
            }
            let _ = writeln!(out, "# TYPE {name} counter");
            let mut label_keys: Vec<&Labels> = family.values.keys().collect();
            label_keys.sort();
            for labels in label_keys {
                let _ = writeln!(
                    out,
                    "{name}{} {}",
                    format_labels(labels),
                    family.values[labels]
                );
            }
            out.push('\n');
        }
        drop(counters);

        let gauges = self.gauges.read();
        let mut names: Vec<&Arc<str>> = gauges.keys().collect();
        names.sort();
        for name in names {
            let family = &gauges[name];
            if !family.help.is_empty() {
                let _ = writeln!(out, "# HELP {name} {}", family.help);
            }
            let _ = writeln!(out, "# TYPE {name} gauge");
            let mut label_keys: Vec<&Labels> = family.values.keys().collect();
            label_keys.sort();
            for labels in label_keys {
                let _ = writeln!(
                    out,
                    "{name}{} {}",
                    format_labels(labels),
                    family.values[labels]
                );
            }
            out.push('\n');
        }
        drop(gauges);

        let histograms = self.histograms.read();
        let mut names: Vec<&Arc<str>> = histograms.keys().collect();
        names.sort();
        for name in names {
            let family = &histograms[name];
            if !family.help.is_empty() {
                let _ = writeln!(out, "# HELP {name} {}", family.help);
            }
            let _ = writeln!(out, "# TYPE {name} histogram");
            let mut label_keys: Vec<&Labels> = family.series.keys().collect();
            label_keys.sort();
            for labels in label_keys {
                let series = &family.series[labels];
                // Emit one `_bucket{le="<upper>"}` line per boundary plus
                // the implicit `+Inf` line carrying the grand total. When
                // the series has pre-existing labels we splice `le=` in as
                // another comma-separated entry; when it doesn't we emit
                // the `le=` label alone.
                let base = format_labels(labels);
                let inner = trim_outer_braces(&base);
                let le_prefix: &str = if inner.is_empty() { "" } else { "," };
                for (i, &upper) in family.buckets.iter().enumerate() {
                    let _ = writeln!(
                        out,
                        "{name}_bucket{{{inner}{le_prefix}le=\"{}\"}} {}",
                        fmt_f64_prom(upper),
                        series.counts[i],
                    );
                }
                let _ = writeln!(
                    out,
                    "{name}_bucket{{{inner}{le_prefix}le=\"+Inf\"}} {}",
                    series.count
                );
                let _ = writeln!(out, "{name}_sum{} {}", base, fmt_f64_prom(series.sum),);
                let _ = writeln!(out, "{name}_count{} {}", base, series.count,);
            }
            out.push('\n');
        }

        out
    }
}

/// Strip the surrounding `{ … }` from a pre-formatted label block so we
/// can splice the `le=…` sample label into the same comma-separated
/// sequence. Returns `""` when the input has no labels.
fn trim_outer_braces(formatted: &str) -> &str {
    if formatted.is_empty() {
        return "";
    }
    let inner = formatted
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .unwrap_or(formatted);
    if inner.is_empty() { "" } else { inner }
}

/// Format a float the way the Prometheus go client does: `+Inf` for
/// infinity, `NaN` for NaN, default `{}` otherwise.
fn fmt_f64_prom(v: f64) -> String {
    if v.is_infinite() {
        return if v.is_sign_positive() {
            "+Inf".into()
        } else {
            "-Inf".into()
        };
    }
    if v.is_nan() {
        return "NaN".into();
    }
    format!("{v}")
}

/// Register the server-wide Prometheus families used by HTTP / WS / pool
/// middleware. Called once when `--metrics` is on; the returned registry is
/// shared by the primary app and the loopback `/metrics` listener.
pub(crate) fn register_server_metrics() -> MetricsRegistry {
    let reg = MetricsRegistry::new();
    reg.register_counter(
        "gigastt_http_requests_total",
        "Total HTTP requests processed",
    );
    reg.register_histogram(
        "gigastt_http_request_duration_seconds",
        "HTTP request duration in seconds",
        DEFAULT_BUCKETS,
    );
    reg.register_gauge(
        "gigastt_pool_available",
        "Number of session triplets currently available in the pool",
    );
    reg.register_gauge(
        "gigastt_pool_waiters",
        "Number of tasks currently waiting for a pool checkout",
    );
    reg.register_gauge(
        "gigastt_batch_pool_available",
        "Number of session triplets currently available in the batch pool \
             (only populated when --batch-pool-size > 0)",
    );
    reg.register_gauge(
        "gigastt_batch_pool_waiters",
        "Number of tasks currently waiting for a batch-pool checkout",
    );
    reg.register_histogram(
        "gigastt_pool_checkout_duration_seconds",
        "Time spent waiting for a pool checkout",
        DEFAULT_BUCKETS,
    );
    reg.register_counter(
        "gigastt_pool_timeouts_total",
        "Total pool checkout timeouts",
    );
    reg.register_gauge(
        "gigastt_ws_active_connections",
        "Number of active WebSocket connections",
    );
    reg.register_histogram(
        "gigastt_inference_duration_seconds",
        "Inference duration in seconds",
        DEFAULT_BUCKETS,
    );
    reg.register_counter(
        "gigastt_rate_limit_rejections_total",
        "Total requests rejected by rate limiter",
    );
    reg.register_counter(
        "gigastt_inference_timeouts_total",
        "Total inference runs aborted by the per-request inference timeout",
    );
    reg
}

#[cfg(test)]
mod tests;
