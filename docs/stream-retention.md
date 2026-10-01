# Streaming audio retention

`--stream-max-window-secs` is a threshold at which the engine attempts to
commit and slide. With stable-prefix commits enabled, it is **not a hard
retained-audio or encoder-work limit**. A nonempty hypothesis whose words
remain near the moving right edge can prevent sliding indefinitely. This is
a deterministic policy counterexample, not evidence that a particular model
or recorded conversation produces that sequence.

The policy excludes words ending in the final 1.0 second of the retained
window. After three unsuccessful cap attempts it relaxes agreement, but
still excludes those edge words. If every hypothesis places its only word
0.1 second before the current edge, zero words commit on every attempt.
The engine retains their audio to avoid silently losing uncommitted speech.
With the default 2.5-second trigger and 120 increments of 0.8 second, this
sequence retains 98.5 seconds: 1,576,000 f32 samples, or 6,304,000 logical PCM
bytes. Vec capacity and model activations are additional memory. Repeating
the sequence yields no finite bound derived from the configured trigger.
Agreement alone is insufficient if the word's ending timestamp keeps moving.

The model-free tests in `inference/engine/tests/mock.rs` exercise the actual
commit and slide methods for moving edge words, timestamp drift, stationary
words without agreement, silence, and manual endpoint mode. Stationary words
eventually leave the horizon and commit; an empty silent tail slides to the
1.5-second left context. Manual mode intentionally ignores automatic endpoint
signals. These tests characterize the policy; they do not substitute for
real-speech quality measurements.

Reproduce with:

```sh
cargo test -p gigastt-core --lib retention
cargo test -p gigastt-core --lib test_moving_edge_hypotheses
```

## Limits and a bounded correction contract

The server's nonzero wall-clock session cap limits connection lifetime, but
it does not establish a PCM bound: input can arrive faster than real time,
and a core-library caller has no server deadline. Frame size bounds one
message, not accumulated audio. A no-progress timeout cannot interrupt an
already-running native encoder call.

A separate hard retained-sample limit should be enforced **before appending
input and before invoking the encoder**, using checked/saturating arithmetic
and counting retained left context. Its breach should stop that transcription
with a typed resource-limit error, preserve the last readable provisional
text, and release the inference reservation after any active synchronous call
returns. It must not silently discard the live tail, report normal completion,
or force uncertain edge words into committed text. A new session/request can
then be started explicitly by the caller.

That policy needs an independently named, documented limit, a default selected
from measured long-speech workloads, consistent handling across core/FFI and
WS/SSE surfaces, and explicit tests for oversized single chunks and repeated
small chunks. Before enabling it, compare real-speech transcripts, commitment
stability and endpoint timing across default/manual/assistant modes and both
RNN-T heads and applicable CTC paths, including legitimate long words and unstable timestamps. The
research change described here leaves runtime commitment and endpoint behavior
unchanged; it removes the unsupported boundedness claim rather than selecting
an unmeasured cutoff.
