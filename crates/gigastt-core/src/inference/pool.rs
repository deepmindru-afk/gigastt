//! Session pool: checkout/checkin of inference session triplets.

use std::ops::{Deref, DerefMut};

use crate::runtime::session::RuntimeSession;
use crate::runtime::tensor::Tensor;

/// A set of ONNX sessions for one inference pipeline (encoder + decoder + joiner).
///
/// Moved out of the pool on checkout and returned on checkin.
/// Each triplet is independent and can run inference concurrently with others.
///
/// The RNN-T heads populate all three sessions. The encoder-only CTC heads leave
/// `decoder` / `joiner` as `None` — the CTC branch in `run_inference` decodes
/// straight from the encoder output and never touches them, so loading them would
/// only waste encoder-sized RAM.
pub struct SessionTriplet {
    pub(crate) encoder: Box<dyn RuntimeSession>,
    pub(crate) decoder: Option<Box<dyn RuntimeSession>>,
    pub(crate) joiner: Option<Box<dyn RuntimeSession>>,
    /// Reusable encoder input tensors: `[audio_signal [1, N_MELS, num_frames], length [1]]`.
    /// Resized and overwritten in `run_inference` to avoid per-call allocations.
    pub(crate) encoder_inputs: Vec<Tensor>,
}

/// Errors returned by [`Pool::checkout`].
#[derive(Debug)]
pub enum PoolError {
    /// The pool was closed (graceful shutdown). All current and future
    /// waiters resolve to this variant; the caller should respond with a
    /// 503 / `pool_closed` to the client.
    Closed,
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::Closed => write!(f, "session pool is closed"),
        }
    }
}

impl std::error::Error for PoolError {}

/// Pool of pre-loaded items of type `T`.
///
/// `SessionPool = Pool<SessionTriplet>` is the only public instantiation
/// outside this module. Generic `T` exists so the pool semantics can be
/// unit-tested without ONNX models.
///
/// Checkout = pop from the queue, checkin = push back via the
/// [`PoolGuard`] returned by [`checkout`](Self::checkout). The pool size acts
/// as the concurrency limit — no separate semaphore needed. FIFO ordering is
/// preserved because waiters are stored in a queue and served in order.
pub struct Pool<T> {
    inner: std::sync::Arc<PoolInner<T>>,
}

struct PoolInner<T> {
    items: parking_lot::Mutex<std::collections::VecDeque<T>>,
    waiters: parking_lot::Mutex<std::collections::VecDeque<Waiter<T>>>,
    closed: std::sync::atomic::AtomicBool,
    total: usize,
}

enum Waiter<T> {
    #[cfg(feature = "async-pool")]
    Async(tokio::sync::oneshot::Sender<T>),
    Blocking(std::sync::mpsc::Sender<T>),
}

/// Public alias for the production pool: holds [`SessionTriplet`] instances.
pub type SessionPool = Pool<SessionTriplet>;

impl<T: Send> Pool<T> {
    /// Create a pool pre-filled with the given items.
    pub fn new(items: Vec<T>) -> Self {
        let total = items.len();
        Self {
            inner: std::sync::Arc::new(PoolInner {
                items: parking_lot::Mutex::new(std::collections::VecDeque::from(items)),
                waiters: parking_lot::Mutex::new(std::collections::VecDeque::new()),
                closed: std::sync::atomic::AtomicBool::new(false),
                total,
            }),
        }
    }

    /// Checkout an item from the pool. Awaits FIFO if none available.
    ///
    /// Returns [`PoolError::Closed`] if the pool was shut down via
    /// [`close`](Self::close) before an item became available.
    #[cfg(feature = "async-pool")]
    pub async fn checkout(&self) -> Result<PoolGuard<T>, PoolError> {
        // Fast path
        {
            let mut items = self.inner.items.lock();
            if self.inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(PoolError::Closed);
            }
            if let Some(item) = items.pop_front() {
                return Ok(PoolGuard::new(self.inner.clone(), item));
            }
        }

        // Slow path: register as an async waiter
        let (tx, rx) = tokio::sync::oneshot::channel();
        {
            let mut waiters = self.inner.waiters.lock();
            if self.inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(PoolError::Closed);
            }
            // Re-check items under the waiters lock to prevent the lost-wakeup
            // race: between releasing items.lock() and acquiring waiters.lock(),
            // another thread may have checked in an item and pushed it back to
            // items because there were no waiters yet.
            let mut items = self.inner.items.lock();
            if let Some(item) = items.pop_front() {
                drop(items);
                drop(waiters);
                return Ok(PoolGuard::new(self.inner.clone(), item));
            }
            waiters.push_back(Waiter::Async(tx));
        }

        match rx.await {
            Ok(item) => Ok(PoolGuard::new(self.inner.clone(), item)),
            Err(_) => Err(PoolError::Closed),
        }
    }

    /// Synchronous (blocking) checkout. Used by FFI and other synchronous callers.
    pub fn checkout_blocking(&self) -> Result<PoolGuard<T>, PoolError> {
        // Fast path
        {
            let mut items = self.inner.items.lock();
            if self.inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(PoolError::Closed);
            }
            if let Some(item) = items.pop_front() {
                return Ok(PoolGuard::new(self.inner.clone(), item));
            }
        }

        // Slow path: register as a blocking waiter
        let (tx, rx) = std::sync::mpsc::channel();
        {
            let mut waiters = self.inner.waiters.lock();
            if self.inner.closed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(PoolError::Closed);
            }
            // Same lost-wakeup guard as the async variant.
            let mut items = self.inner.items.lock();
            if let Some(item) = items.pop_front() {
                drop(items);
                drop(waiters);
                return Ok(PoolGuard::new(self.inner.clone(), item));
            }
            waiters.push_back(Waiter::Blocking(tx));
        }

        match rx.recv() {
            Ok(item) => Ok(PoolGuard::new(self.inner.clone(), item)),
            Err(_) => Err(PoolError::Closed),
        }
    }

    /// Close the pool: all current and future [`checkout`](Self::checkout)
    /// callers resolve to [`PoolError::Closed`]. Used by graceful shutdown.
    /// Idempotent.
    pub fn close(&self) {
        self.inner
            .closed
            .store(true, std::sync::atomic::Ordering::SeqCst);
        // Drain all pending waiters so their receivers get Canceled / RecvError.
        let mut waiters = self.inner.waiters.lock();
        waiters.clear();
    }

    /// Total number of items the pool was created with.
    pub fn total(&self) -> usize {
        self.inner.total
    }

    /// Number of currently available (not checked-out) items. O(1).
    pub fn available(&self) -> usize {
        let items = self.inner.items.lock();
        items.len()
    }

    /// Number of waiters currently blocked on checkout. O(1).
    pub fn waiters(&self) -> usize {
        let waiters = self.inner.waiters.lock();
        waiters.len()
    }
}

impl<T> PoolInner<T> {
    fn checkin(&self, mut item: T) {
        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        // Retry loop: if the waiter at the front of the queue was abandoned
        // (its receiver was dropped because the checkout future was cancelled
        // via timeout, select!, or abort), we must skip it and try the next
        // one, or return the item to the pool. Without this retry a cancelled
        // waiter permanently leaks a pool slot.
        loop {
            let mut waiters = self.waiters.lock();
            if let Some(waiter) = waiters.pop_front() {
                drop(waiters);
                match waiter {
                    #[cfg(feature = "async-pool")]
                    Waiter::Async(tx) => {
                        if let Err(returned_item) = tx.send(item) {
                            item = returned_item;
                            continue;
                        }
                    }
                    Waiter::Blocking(tx) => {
                        if let Err(std::sync::mpsc::SendError(returned_item)) = tx.send(item) {
                            item = returned_item;
                            continue;
                        }
                    }
                }
            } else {
                drop(waiters);
                let mut items = self.items.lock();
                items.push_back(item);
            }
            break;
        }
    }
}

/// RAII guard that auto-checks-in an item when dropped.
///
/// Returned by [`Pool::checkout`]. Deref to access the inner item.
/// On drop (including panic unwind) the item is returned to the pool;
/// if the pool was closed in the meantime the item is silently dropped.
pub struct PoolGuard<T> {
    inner: Option<std::sync::Arc<PoolInner<T>>>,
    item: Option<T>,
}

impl<T> PoolGuard<T> {
    fn new(inner: std::sync::Arc<PoolInner<T>>, item: T) -> Self {
        Self {
            inner: Some(inner),
            item: Some(item),
        }
    }

    /// Strip the lifetime so the guard can be moved into a `'static`
    /// context (e.g. `tokio::task::spawn_blocking`). Returns an
    /// [`OwnedReservation`] that owns the item and automatically returns it
    /// to the pool on drop. Call [`OwnedReservation::checkin`] to return the
    /// item explicitly before the reservation is dropped.
    pub fn into_owned(mut self) -> OwnedReservation<T> {
        let item = self
            .item
            .take()
            .unwrap_or_else(|| unreachable!("PoolGuard::into_owned called after drop"));
        let inner = self.inner.take().unwrap();
        OwnedReservation {
            inner,
            item: Some(item),
        }
    }
}

impl<T> Deref for PoolGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.item
            .as_ref()
            .unwrap_or_else(|| unreachable!("PoolGuard accessed after item taken"))
    }
}

impl<T> DerefMut for PoolGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item
            .as_mut()
            .unwrap_or_else(|| unreachable!("PoolGuard accessed after item taken"))
    }
}

impl<T> Drop for PoolGuard<T> {
    fn drop(&mut self) {
        if let (Some(inner), Some(item)) = (self.inner.take(), self.item.take()) {
            inner.checkin(item);
        }
    }
}

/// Owned counterpart to [`PoolGuard`] for `'static` contexts (e.g.
/// `spawn_blocking`). The item is returned to the pool automatically on drop.
///
/// Call [`Self::checkin`] to return the item explicitly and invalidate the
/// guard. If the reservation is dropped without calling `checkin`, the item
/// is still returned to the pool via the [`Drop`] impl. This guarantees that
/// the pool does not leak slots when a `spawn_blocking` task panics or is
/// cancelled.
pub struct OwnedReservation<T> {
    inner: std::sync::Arc<PoolInner<T>>,
    item: Option<T>,
}

impl<T> std::ops::Deref for OwnedReservation<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.item
            .as_ref()
            .unwrap_or_else(|| unreachable!("OwnedReservation accessed after checkin"))
    }
}

impl<T> std::ops::DerefMut for OwnedReservation<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item
            .as_mut()
            .unwrap_or_else(|| unreachable!("OwnedReservation accessed after checkin"))
    }
}

impl<T> OwnedReservation<T> {
    /// Return the item to the pool explicitly. After this call the reservation
    /// is empty and its [`Drop`] is a no-op.
    pub fn checkin(mut self) {
        if let Some(item) = self.item.take() {
            self.inner.checkin(item);
        }
    }
}

impl<T> Drop for OwnedReservation<T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            self.inner.checkin(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_checkout_blocking_fast_path() {
        let pool = Pool::new(vec![42u32]);
        let guard = pool.checkout_blocking().expect("checkout_blocking");
        assert_eq!(*guard, 42);
        drop(guard);
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn test_pool_checkout_blocking_closed() {
        let pool = Pool::<u32>::new(vec![]);
        pool.close();
        assert!(matches!(pool.checkout_blocking(), Err(PoolError::Closed)));
    }

    #[test]
    fn test_pool_checkout_blocking_slow_path() {
        let pool = std::sync::Arc::new(Pool::new(vec![42u32]));
        let primary = pool.checkout_blocking().unwrap();

        let handle = std::thread::spawn({
            let pool = pool.clone();
            move || pool.checkout_blocking()
        });

        std::thread::sleep(std::time::Duration::from_millis(50));
        drop(primary);

        let guard = handle.join().expect("join").expect("checkout");
        assert_eq!(*guard, 42);
        drop(guard);
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn test_pool_error_display() {
        assert_eq!(format!("{}", PoolError::Closed), "session pool is closed");
    }

    // ---- Pool tests (B.7) ---------------------------------------------------
    //
    // These exercise `Pool<T>` with synthetic items so the contract is
    // observable without loading ONNX models. `SessionPool = Pool<SessionTriplet>`
    // is just an alias, so any property proven here also holds for the real
    // pool.

    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_pool_guard_returns_triplet_on_normal_drop() {
        let pool = Pool::new(vec![1u32, 2, 3]);
        assert_eq!(pool.available(), 3);
        {
            let _guard = pool.checkout().await.expect("checkout");
            assert_eq!(pool.available(), 2);
        }
        // Dropping the guard returns the item.
        assert_eq!(pool.available(), 3);
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_pool_guard_returns_triplet_on_panic_unwind() {
        // The guard's Drop impl runs during unwind, so a panic between
        // checkout and the natural end of scope still restores capacity.
        let pool = std::sync::Arc::new(Pool::new(vec![1u32]));
        assert_eq!(pool.available(), 1);

        let pool_clone = pool.clone();
        let result = tokio::spawn(async move {
            let _guard = pool_clone.checkout().await.expect("checkout");
            assert_eq!(pool_clone.available(), 0);
            panic!("synthetic inference panic");
        })
        .await;
        assert!(result.is_err(), "spawned task must report the panic");

        // Capacity is restored thanks to PoolGuard::drop running on unwind.
        assert_eq!(pool.available(), 1);
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_pool_close_wakes_waiters_with_closed() {
        // A waiter blocked in `checkout` after the inventory is exhausted
        // must resolve to PoolError::Closed when `close()` fires. Map the
        // borrowed guard to the `()` success path so the spawn doesn't
        // need to carry the pool's lifetime.
        let pool = std::sync::Arc::new(Pool::<u32>::new(vec![]));
        let waiter = tokio::spawn({
            let pool = pool.clone();
            async move { pool.checkout().await.map(|_g| ()) }
        });

        // Give the waiter a moment to park on the channel.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        pool.close();

        let res = waiter.await.expect("join");
        assert!(matches!(res, Err(PoolError::Closed)));
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_pool_fifo_under_contention() {
        // With a single-slot pool and three queued waiters, the order of
        // wake-ups must match the order in which `checkout` was called.
        // The mpsc channel itself is FIFO; the Mutex serializes waiters
        // so ordering is preserved under normal contention.
        let pool = std::sync::Arc::new(Pool::new(vec![0u32]));

        let primary = pool.checkout().await.expect("primary checkout");
        assert_eq!(pool.available(), 0);

        let waker_log = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for id in 0u32..3 {
            let pool = pool.clone();
            let log = waker_log.clone();
            handles.push(tokio::spawn(async move {
                let g = pool.checkout().await.expect("checkout");
                log.lock().await.push(id);
                drop(g);
            }));
            // Stagger spawns so each waiter is parked before the next one
            // is registered with the channel.
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        // Release the only inventory slot so the queued waiters can run.
        drop(primary);
        for h in handles {
            h.await.expect("join");
        }

        let log = waker_log.lock().await.clone();
        assert_eq!(log, vec![0, 1, 2], "waiters must wake in FIFO order");
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_into_owned_for_spawn_blocking() {
        // `into_owned` strips the lifetime so the item can be moved into a
        // blocking thread, then `OwnedReservation::checkin` returns it.
        let pool = std::sync::Arc::new(Pool::new(vec![String::from("triplet")]));
        let guard = pool.checkout().await.expect("checkout");
        let reservation = guard.into_owned();

        let result = tokio::task::spawn_blocking(move || {
            // Pretend we're running blocking inference.
            assert_eq!(*reservation, "triplet");
            reservation.checkin();
            "done"
        })
        .await
        .expect("join");

        // After the blocking task returns the item, the pool is full again.
        assert_eq!(pool.available(), 1);
        assert_eq!(result, "done");
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_owned_reservation_returns_on_spawn_blocking_panic() {
        // If the blocking task panics, the reservation's Drop must still
        // return the item so the pool does not leak capacity.
        let pool = std::sync::Arc::new(Pool::new(vec![String::from("triplet")]));
        let guard = pool.checkout().await.expect("checkout");
        let reservation = guard.into_owned();

        let result = tokio::task::spawn_blocking(move || {
            let _reservation = reservation;
            panic!("simulated inference panic");
        })
        .await;

        assert!(result.is_err(), "spawn_blocking must report the panic");
        assert_eq!(
            pool.available(),
            1,
            "reservation must be returned after panic"
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_owned_reservation_drop_returns_item() {
        // Dropping an unchecked-in reservation still returns the item.
        let pool = std::sync::Arc::new(Pool::new(vec![String::from("triplet")]));
        let guard = pool.checkout().await.expect("checkout");
        let reservation = guard.into_owned();

        tokio::task::spawn_blocking(move || {
            let _reservation = reservation;
            // reservation dropped here
        })
        .await
        .expect("join");

        assert_eq!(pool.available(), 1);
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_pool_close_is_idempotent() {
        // `pool.close()` is wired into the shutdown hook; calling it twice
        // (e.g. shutdown signal + Drop) must not panic.
        let pool = Pool::<u32>::new(vec![]);
        pool.close();
        pool.close();
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_pool_waiters_count() {
        let pool = std::sync::Arc::new(Pool::<u32>::new(vec![]));
        let w1 = tokio::spawn({
            let p = pool.clone();
            async move { p.checkout().await.map(|_| ()) }
        });
        let w2 = tokio::spawn({
            let p = pool.clone();
            async move { p.checkout().await.map(|_| ()) }
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(pool.waiters(), 2, "both blocked tasks must be waiters");
        pool.close();
        let _ = w1.await;
        let _ = w2.await;
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_owned_reservation_round_trip_through_option() {
        // Mirrors the pattern used by `handle_binary_frame` in ws.rs:
        // the reservation is temporarily moved out of an Option into
        // spawn_blocking and then placed back on success.
        let pool = std::sync::Arc::new(Pool::new(vec![42u32]));
        let guard = pool.checkout().await.expect("checkout");
        let mut reservation: Option<OwnedReservation<u32>> = Some(guard.into_owned());

        let (res_back, val) = tokio::task::spawn_blocking(move || {
            let mut r = reservation.take().unwrap();
            *r += 1;
            let v = *r;
            (r, v)
        })
        .await
        .expect("join");

        reservation = Some(res_back);
        assert_eq!(val, 43);
        drop(reservation);
        assert_eq!(pool.available(), 1);
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_pool_slot_not_leaked_on_cancelled_checkout() {
        // If a checkout future is cancelled after registering a waiter but
        // before receiving an item, the oneshot receiver is dropped while the
        // sender remains in the waiters queue.  When another item is checked
        // in, the dead waiter must be skipped and the item returned to the
        // pool — otherwise the slot is leaked forever.
        let pool = std::sync::Arc::new(Pool::new(vec![42u32]));
        let primary = pool.checkout().await.expect("checkout");

        let aborted = tokio::spawn({
            let pool = pool.clone();
            async move { pool.checkout().await }
        });
        // Let the spawned task register as a waiter.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        aborted.abort();
        let _ = aborted.await;

        // The abandoned waiter is still queued.
        assert_eq!(pool.waiters(), 1);

        // Return the primary item.  Without the retry loop in checkin this
        // would silently drop the item because tx.send fails.
        drop(primary);

        assert_eq!(pool.available(), 1, "item must return to pool, not leak");
        assert_eq!(pool.waiters(), 0, "dead waiter must be removed");
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_pool_slot_not_leaked_on_timeout_checkout() {
        // Same scenario as above, but using tokio::time::timeout instead of
        // abort — this is the exact path hit by the REST and WS handlers.
        let pool = std::sync::Arc::new(Pool::new(vec![42u32]));
        let primary = pool.checkout().await.expect("checkout");

        let result =
            tokio::time::timeout(std::time::Duration::from_millis(10), pool.checkout()).await;
        assert!(result.is_err(), "checkout must time out");

        assert_eq!(pool.waiters(), 1);

        drop(primary);

        assert_eq!(
            pool.available(),
            1,
            "item must return to pool after timeout"
        );
        assert_eq!(pool.waiters(), 0, "dead waiter must be removed");
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
    async fn test_pool_multiple_dead_waiters_are_skipped() {
        // Several cancelled waiters in a row should all be skipped in one
        // checkin pass.
        let pool = std::sync::Arc::new(Pool::new(vec![0u32]));
        let primary = pool.checkout().await.expect("checkout");

        let mut handles = Vec::new();
        for _ in 0..3 {
            handles.push(tokio::spawn({
                let pool = pool.clone();
                async move { pool.checkout().await }
            }));
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        for h in handles {
            h.abort();
            let _ = h.await;
        }

        assert_eq!(pool.waiters(), 3);

        drop(primary);

        assert_eq!(
            pool.available(),
            1,
            "item returned after skipping 3 dead waiters"
        );
        assert_eq!(pool.waiters(), 0);
    }

    #[test]
    fn test_pool_sequential_checkouts_visit_every_item() {
        // Engine::warmup relies on this FIFO property: `total()` sequential
        // checkout/checkin cycles touch every pooled item exactly once.
        let pool = Pool::new(vec![1u32, 2, 3]);
        let mut seen = Vec::new();
        for _ in 0..pool.total() {
            let guard = pool.checkout_blocking().expect("checkout");
            seen.push(*guard);
            // guard drops here — the item returns to the back of the queue
        }
        seen.sort_unstable();
        assert_eq!(seen, vec![1, 2, 3]);
    }
}
