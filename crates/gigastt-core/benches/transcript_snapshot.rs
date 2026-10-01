//! Allocation scaling of window publication, separately from owned reads.
//! cargo bench -p gigastt-core --features __internals --bench transcript_snapshot
use gigastt_core::inference::WordInfo;
use gigastt_core::test_support::SnapshotBenchPublisher;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct CountingAllocator;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every allocation operation is forwarded unchanged to System;
// counters only observe sizes and never access or modify allocated memory.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            ALLOCATIONS.fetch_add(1, Relaxed);
            ALLOCATED.fetch_add(layout.size(), Relaxed);
            LIVE.fetch_add(layout.size(), Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let next = unsafe { System.realloc(ptr, layout, new_size) };
        if !next.is_null() {
            ALLOCATIONS.fetch_add(1, Relaxed);
            ALLOCATED.fetch_add(new_size, Relaxed);
            LIVE.fetch_add(new_size, Relaxed);
            LIVE.fetch_sub(layout.size(), Relaxed);
        }
        next
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn main() {
    println!(
        "channels,windows,words,read_every,publication_allocations,publication_bytes,retained_bytes,reader_bytes"
    );
    for channels in [1, 2] {
        for read_every in [0, 16] {
            for windows in [16usize, 64, 256, 1024] {
                let words: Vec<_> = (0..windows * 16)
                    .map(|i| WordInfo::new(format!("word{i}"), i as f64, i as f64 + 0.5, 0.9, None))
                    .collect();
                let mut snapshot = SnapshotBenchPublisher::default();
                let before_live = LIVE.load(Relaxed);
                let (mut allocations, mut bytes, mut reader_bytes) = (0, 0, 0);
                for channel in 0..channels {
                    for window in 1..=windows {
                        let end = window * 16;
                        let retained = end.saturating_sub(20);
                        ALLOCATIONS.store(0, Relaxed);
                        ALLOCATED.store(0, Relaxed);
                        snapshot.publish(
                            &words[..end],
                            retained,
                            (channels > 1).then_some(channel),
                        );
                        allocations += ALLOCATIONS.load(Relaxed);
                        bytes += ALLOCATED.load(Relaxed);
                        if read_every > 0 && window % read_every == 0 {
                            ALLOCATED.store(0, Relaxed);
                            let result = snapshot.get().unwrap();
                            assert_eq!(result.words.len(), channel * words.len() + end);
                            reader_bytes += ALLOCATED.load(Relaxed);
                        }
                    }
                }
                let retained_bytes = LIVE.load(Relaxed) - before_live;
                ALLOCATED.store(0, Relaxed);
                let result = snapshot.get().unwrap();
                assert_eq!(result.words.len(), channels * words.len());
                reader_bytes += ALLOCATED.load(Relaxed);
                println!(
                    "{channels},{windows},{},{read_every},{allocations},{bytes},{retained_bytes},{reader_bytes}",
                    channels * words.len()
                );
            }
        }
    }
}
