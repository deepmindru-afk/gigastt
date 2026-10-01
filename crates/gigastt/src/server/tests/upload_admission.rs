//! Separate-process upload pressure probe: server RSS excludes client buffers.

use super::mock_engine;
use crate::server::{ServerConfig, run_with_config_listener};
use futures_util::{StreamExt, future::join_all};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const INPUT_BYTES: usize = 8 * 1024 * 1024;

struct ProbeServer(Child);
impl Drop for ProbeServer {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn memory_kib(pid: u32, field: &str) -> u64 {
    std::fs::read_to_string(format!("/proc/{pid}/status"))
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix(field))
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

#[test]
fn test_upload_memory_probe_server() {
    let Some(directory) = std::env::var_os("GIGASTT_UPLOAD_PROBE_DIR") else {
        return;
    };
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let (engine, _models) = mock_engine();
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let metrics_probe = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let metrics_address = metrics_probe.local_addr().unwrap();
            drop(metrics_probe);
            let mut config = ServerConfig::local(address.port());
            config.limits.body_limit_bytes = INPUT_BYTES;
            config.limits.pool_checkout_timeout_secs = 30;
            config.limits.jobs_enabled = true;
            config.metrics_enabled = true;
            config.metrics_listen = metrics_address;
            std::fs::write(
                Path::new(&directory).join("address"),
                format!("{address}\n{metrics_address}\n"),
            )
            .unwrap();
            run_with_config_listener(engine, config, None, listener)
                .await
                .unwrap();
        });
}

#[tokio::test]
async fn test_upload_admission_bounds_server_memory_at_pool_saturation() {
    let directory = tempfile::tempdir().unwrap();
    let child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "server::tests::upload_admission::test_upload_memory_probe_server",
            "--nocapture",
        ])
        .env("GIGASTT_UPLOAD_PROBE_DIR", directory.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let server = ProbeServer(child);
    let addresses = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Ok(address) = std::fs::read_to_string(directory.path().join("address")) {
                break address;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let mut addresses = addresses.lines();
    let address = addresses.next().unwrap();
    let metrics = addresses.next().unwrap();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if client
                .get(format!("http://{address}/health"))
                .send()
                .await
                .is_ok()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    // A live WebSocket owns the real pool's sole slot after boot warmup.
    let (mut websocket, _) = tokio_tungstenite::connect_async(format!("ws://{address}/v1/ws"))
        .await
        .unwrap();
    let ready = websocket
        .next()
        .await
        .unwrap()
        .unwrap()
        .into_text()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&ready).unwrap()["type"],
        "ready"
    );
    let baseline_kib = memory_kib(server.0.id(), "VmRSS:");
    let pending_client = reqwest::Client::new();
    let pending_url = format!("http://{address}/v1/transcribe");
    let pending = tokio::spawn(async move {
        pending_client
            .post(pending_url)
            .body(vec![0u8; INPUT_BYTES])
            .send()
            .await
    });
    // Readiness samples the real pool gauge. One waiter proves the accepted
    // body was collected and is retained while inference has no free slot.
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let _ = client
                .get(format!("http://{address}/ready"))
                .send()
                .await
                .unwrap();
            let text = client
                .get(format!("http://{metrics}/metrics"))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap();
            if text.lines().any(|line| line == "gigastt_pool_waiters 1") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();

    let mut attempts = Vec::new();
    for _ in 0..4 {
        for path in [
            "/v1/transcribe",
            "/v1/transcribe/stream",
            "/v1/audio/transcriptions",
            "/v1/jobs",
        ] {
            attempts.push(async move {
                let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
                socket.write_all(format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {INPUT_BYTES}\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
                let mut response = vec![0; 4096];
                let bytes = tokio::time::timeout(Duration::from_secs(2), socket.read(&mut response)).await.unwrap().unwrap();
                assert!(String::from_utf8_lossy(&response[..bytes]).starts_with("HTTP/1.1 503"));
            });
        }
    }
    join_all(attempts).await;
    assert_eq!(
        client
            .get(format!("http://{address}/health"))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    assert_eq!(
        client
            .get(format!("http://{address}/ready"))
            .send()
            .await
            .unwrap()
            .status(),
        503
    );
    let peak_kib = memory_kib(server.0.id(), "VmHWM:");
    eprintln!(
        "upload pressure: server baseline RSS={baseline_kib} KiB, server peak RSS={peak_kib} KiB, retained input={} KiB, rejected concurrent uploads=16; client buffers are in parent pid {}",
        INPUT_BYTES / 1024,
        std::process::id()
    );
    // Room for allocator and HTTP/runtime overhead; sixteen full collected
    // inputs would exceed this by over 100 MiB even with a saturated pool.
    assert!(
        peak_kib <= baseline_kib + 64 * 1024,
        "unexpected server memory growth"
    );
    pending.abort();
}
