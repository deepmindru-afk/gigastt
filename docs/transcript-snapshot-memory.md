# Transcript snapshot allocation

File decoders keep a readable provisional transcript for cancellation and
status readers. Window publication now copies only the suffix replaced by
stitching. Immutable word chunks share the unchanged prefix; split channels
retain independent histories. Growing VAD mappings republish any previously
clamped tail whose timestamp can change. The stitching policy is unchanged.

`TranscriptSnapshot::get()` still returns an owned `TranscriptSegment` with
exact words, timestamps, speaker labels and provisional flags. It captures
shared heads under the mutex, then builds text and words outside the lock.
Single-channel reads copy O(total words); split-channel reads also sort by
start time and channel index. Frequent reads and full streaming wire payloads
therefore still incur full-transcript work. Snapshot publication alone no
longer repeatedly copies the unchanged prefix. Each decode owns its publication
history; sharing a snapshot between concurrent requests preserves complete
last-writer results without mixing their transcripts.

A private truncated chunk view retains at most twice its visible word storage;
smaller prefixes are compacted. Older readers can keep prior chunk versions
alive until their reads finish. Chain destruction is iterative, including
cancellation/reset after long recordings.

## Reproduce

```sh
cargo bench -p gigastt-core --features __internals --bench transcript_snapshot
```

This synthetic benchmark uses 16 new words per window and revisits four words
at each seam. Input construction is outside the measured region. It reports
cumulative allocation count/bytes for publication, bytes still retained by the
publisher and snapshot, and allocation bytes for owned readers separately. Two channels and
reads every 16 windows are included. Allocation counts can also be collected
with `--profile dev`; they are not wall-clock performance measurements.

On Linux x86_64, Rust 1.98.1, the original publication chain and the updated
single-channel path produced the following cumulative allocation bytes without
intermediate readers:

| Windows | Words | Previous publication | Updated publication | Updated retained snapshot |
|---:|---:|---:|---:|---:|
| 16 | 256 | 739,664 | 39,622 | 20,966 |
| 64 | 1,024 | 11,440,752 | 160,418 | 84,930 |
| 256 | 4,096 | 184,959,984 | 647,330 | 344,514 |
| 1,024 | 16,384 | 2,994,728,688 | 2,602,954 | 1,390,826 |

At 1,024 windows, publication allocation count falls from 33,608,919 to 25,596.
Retained storage stays comparable (previously 1,359,308 bytes). With two channels,
updated publication allocates 5,214,068 bytes and retains 2,781,620 bytes for
32,768 words. Increasing window count fourfold increases publication allocation
by approximately fourfold, rather than sixteenfold.

Owned reads remain a separate cost: a single final read allocates 1,375,668
bytes for the single-channel case. Reading every 16 windows plus the final read
allocates 45,776,172 bytes cumulatively; the two-channel counterpart allocates
299,788,250 bytes. These results demonstrate reduced copying during publication,
not an inference speedup or reduced total transcript size.
