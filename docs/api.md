# API

gigastt exposes WebSocket (streaming), REST, and SSE on a single port (default `9876`).
Machine-readable specs: [`docs/asyncapi.yaml`](asyncapi.yaml) (WebSocket) and
[`docs/openapi.yaml`](openapi.yaml) (REST).

## WebSocket — real-time streaming

Connect to `ws://127.0.0.1:9876/v1/ws`, send PCM16 audio frames, receive transcription
in real time.

```
Client                            Server
  |-------- connect --------------> |
  | <------- ready ----------------- |  {type:"ready", version:"1.0"}
  |------- configure (optional) --> |  {type:"configure", sample_rate:16000}
  |-------- binary PCM16 ---------> |
  | <------- partial --------------- |  {type:"partial", text:"привет"}
  | <------- final ----------------- |  {type:"final", text:"Привет, как дела?"}
```

**Supported sample rates:** 8, 16, 24, 44.1, 48 kHz (default 48 kHz, resampled to
16 kHz internally). Protocol messages are versioned via the `type` field; new fields
are additive only.

## REST

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check. Reports the loaded head + effective punctuation/ITN policy (`{"status":"ok","model":"gigaam-v3-rnnt","variant":"rnnt","punctuation":true,"itn":true,...}`). During first-run model download it stays up with `model:"loading"`. |
| `/ready` | GET | Readiness probe (200 when the engine pool is ready; 503 `initializing` while the model loads, `pool_exhausted` when saturated) |
| `/v1/models` | GET | Model info (encoder type, pool size, capabilities) |
| `/v1/transcribe` | POST | File transcription, full JSON response or export format |
| `/v1/transcribe/stream` | POST | File transcription with SSE streaming |
| `/v1/ws` | GET | WebSocket upgrade for real-time streaming |
| `/metrics` | GET | Prometheus metrics (enabled with `--metrics`). Served on the separate `--metrics-listen` port (default `127.0.0.1:9090`), not the main API port. |

```sh
# Full JSON
curl -X POST http://127.0.0.1:9876/v1/transcribe \
  -H "Content-Type: application/octet-stream" --data-binary @recording.wav
# {"text":"Привет, как дела?","words":[{"word":"Привет,","start":0.5,"end":0.9,"confidence":0.97}, ...],"duration":3.5}

# SSE streaming
curl -X POST http://127.0.0.1:9876/v1/transcribe/stream \
  -H "Content-Type: application/octet-stream" --data-binary @recording.wav
# data: {"type":"partial","text":"привет как"}
# data: {"type":"final","text":"Привет, как дела?"}
```

The default `rnnt` head emits bare lowercase; punctuation, casing, and Russian ITN are
applied per server configuration (`--punctuation` / `--itn`). The `e2e_rnnt` head bakes
them in.

### Query parameters

`/v1/transcribe` accepts the following query parameters:

- `channels` (optional, string) — use `split` to transcribe the left and right channels
  as separate speakers (`speaker_0`, `speaker_1`). Defaults to mono mix.
- `diarization` (optional, boolean) — request polyvoice speaker diarization. Mutually
  exclusive with `channels=split`; returns `400` with code `conflicting_modes` if both
  are set.

When either channel split or diarization produces speaker labels, each word object
includes a `speaker` integer:

```json
{
  "text": "привет да как дела",
  "words": [
    {"word": "привет", "start": 0.0, "end": 0.4, "confidence": 0.95, "speaker": 0},
    {"word": "да", "start": 0.5, "end": 0.8, "confidence": 0.91, "speaker": 1},
    {"word": "как", "start": 1.0, "end": 1.3, "confidence": 0.93, "speaker": 0},
    {"word": "дела", "start": 1.4, "end": 1.8, "confidence": 0.94, "speaker": 1}
  ],
  "duration": 2.0
}
```

### Export formats

`/v1/transcribe` supports alternative output formats via the `format` query
parameter:

| Format | Query value | Content-Type | Notes |
|---|---|---|---|
| JSON (default) | `json` | `application/json` | Same response as before |
| Plain text | `txt` | `text/plain` | Transcript text only |
| SubRip | `srt` | `application/x-subrip` | Speaker-aware cues |
| WebVTT | `vtt` | `text/vtt` | Speaker-aware cues |
| Markdown | `md` | `text/markdown` | YAML frontmatter + transcript |

```sh
# SubRip subtitles
curl -X POST "http://127.0.0.1:9876/v1/transcribe?format=srt" \
  -H "Content-Type: application/octet-stream" --data-binary @recording.wav

# Markdown with word timings
curl -X POST "http://127.0.0.1:9876/v1/transcribe?format=md&word_timestamps=true" \
  -H "Content-Type: application/octet-stream" --data-binary @recording.wav

# Force download as attachment
curl -X POST "http://127.0.0.1:9876/v1/transcribe?format=vtt&download=recording.vtt" \
  -H "Content-Type: application/octet-stream" --data-binary @recording.wav
```

Optional formatter controls:

- `max_chars_per_line` — subtitle line length limit (default 80, 0 = unlimited)
- `max_words_per_line` — subtitle word count limit (default 14, 0 = unlimited)
- `word_timestamps` — include per-word table in Markdown (default false)

### Long recordings — send the whole file

gigastt chunks long audio **internally** (overlapping windows, de-duplicated and
stitched with monotonic timestamps), so you do **not** need to pre-segment with
ffmpeg. POST the whole recording and let the server return real, media-relative
timestamps — every `word` in the JSON response carries `start`/`end` in seconds,
and the `srt`/`vtt`/`md` exports key off those, so there's no need to fabricate
per-chunk offsets client-side.

```sh
# One long recording → Markdown transcript with real per-word timings
curl -X POST "http://127.0.0.1:9876/v1/transcribe?format=md&word_timestamps=true" \
  --data-binary @meeting.m4a -o meeting.md
```

Content-Type is ignored — the container format (WAV/MP3/M4A/OGG/FLAC) is sniffed
from the bytes, so `--data-binary @file` is enough; multipart form uploads are
not accepted. The practical ceiling is the body limit (`--body-limit-bytes`,
default 50 MiB ≈ 26 min of 16 kHz mono WAV) and the per-request inference cap
(`--inference-timeout-secs`, default 600 s); raise **both** together for longer
single files. A batch worker should gate on `GET /ready` (not just `/health`) so
it backs off on `503` pool saturation instead of failing mid-job.

### Error responses

| HTTP | Code | When |
|---|---|---|
| 400 | `empty_body` | Request body is empty |
| 400 | `invalid_format` | Unsupported `format` query value |
| 400 | `conflicting_modes` | Both `channels=split` and `diarization=true` were requested |
| 413 | `payload_too_large` | Body exceeds `--body-limit-bytes` (default 50 MiB) |
| 422 | `invalid_audio` | Audio could not be decoded (unsupported/corrupt format) |
| 422 | `transcription_error` | Audio decoded but inference failed |
| 429 | `rate_limited` | Per-IP token bucket exhausted; `Retry-After` header included |
| 503 | `timeout` | All inference sessions busy; `Retry-After` + `retry_after_ms` |
| 503 | `pool_closed` | Server is shutting down, pool closed to new checkouts |

```
HTTP/1.1 503 Service Unavailable
Retry-After: 30

{"error":"Server busy, try again later","code":"timeout","retry_after_ms":30000}
```

## Client libraries

Ready-to-use WebSocket clients live in [`examples/`](../examples/):

```sh
pip install websockets && python examples/python_client.py recording.wav   # Python
bun examples/bun_client.ts recording.wav                                    # Bun / TypeScript
go run examples/go_client.go recording.wav                                  # Go (gorilla/websocket)
kotlinc examples/KotlinClient.kt -include-runtime -d client.jar && java -jar client.jar recording.wav  # Kotlin
```
