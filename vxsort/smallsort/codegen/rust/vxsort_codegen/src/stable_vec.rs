use std::{
    cell::UnsafeCell,
    fmt,
    mem::MaybeUninit,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

type StableSlot<T> = UnsafeCell<MaybeUninit<T>>;
type StableChunk<T> = Box<[StableSlot<T>]>;
type StableChunkLock<T> = OnceLock<StableChunk<T>>;

pub struct StableVec<T, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> {
    chunks: [StableChunkLock<T>; MAX_CHUNKS],
    len: AtomicUsize,
    write_lock: Mutex<()>,
}

#[derive(Clone)]
pub struct StableVecSnapshot<T, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> {
    vec: Arc<StableVec<T, FIRST_CHUNK, MAX_CHUNKS>>,
    len: usize,
}

pub struct StableVecIter<'a, T, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> {
    snapshot: &'a StableVecSnapshot<T, FIRST_CHUNK, MAX_CHUNKS>,
    index: usize,
}

unsafe impl<T: Send, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> Send
    for StableVec<T, FIRST_CHUNK, MAX_CHUNKS>
{
}

unsafe impl<T: Sync, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> Sync
    for StableVec<T, FIRST_CHUNK, MAX_CHUNKS>
{
}

impl<T, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> StableVec<T, FIRST_CHUNK, MAX_CHUNKS> {
    pub fn new() -> Self {
        assert!(FIRST_CHUNK > 0, "first stable vector chunk must be nonzero");
        assert!(MAX_CHUNKS > 0, "stable vector must have at least one chunk");
        Self {
            chunks: std::array::from_fn(|_| OnceLock::new()),
            len: AtomicUsize::new(0),
            write_lock: Mutex::new(()),
        }
    }

    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn push(&self, value: T) -> usize {
        let _guard = self
            .write_lock
            .lock()
            .expect("stable vector append lock should not be poisoned");
        let index = self.len.load(Ordering::Relaxed);
        let (chunk_index, offset) = locate_index::<FIRST_CHUNK, MAX_CHUNKS>(index)
            .expect("stable vector capacity exhausted");
        let chunk = self.ensure_chunk(chunk_index);
        // SAFETY: the append lock serializes writers, `index == len` points at
        // an unpublished slot, and the release-store below publishes it only
        // after the value has been fully written.
        unsafe {
            (*chunk[offset].get()).write(value);
        }
        self.len.store(index + 1, Ordering::Release);
        index
    }

    pub fn get(&self, index: usize) -> &T {
        assert!(
            index < self.len(),
            "stable vector index {index} out of bounds"
        );
        let (chunk_index, offset) =
            locate_index::<FIRST_CHUNK, MAX_CHUNKS>(index).expect("published index must resolve");
        let chunk = self.chunks[chunk_index]
            .get()
            .expect("published index chunk must be allocated");
        // SAFETY: `index < len(Acquire)` means the slot was initialized before
        // publication. Published entries are never mutated or moved.
        unsafe { (&*chunk[offset].get()).assume_init_ref() }
    }

    pub fn snapshot(self: &Arc<Self>) -> StableVecSnapshot<T, FIRST_CHUNK, MAX_CHUNKS> {
        StableVecSnapshot {
            vec: Arc::clone(self),
            len: self.len(),
        }
    }

    fn ensure_chunk(&self, chunk_index: usize) -> &[StableSlot<T>] {
        self.chunks[chunk_index]
            .get_or_init(|| {
                let capacity = chunk_capacity::<FIRST_CHUNK>(chunk_index)
                    .expect("stable vector chunk capacity overflowed");
                (0..capacity)
                    .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .as_ref()
    }
}

impl<T, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> Default
    for StableVec<T, FIRST_CHUNK, MAX_CHUNKS>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> fmt::Debug
    for StableVec<T, FIRST_CHUNK, MAX_CHUNKS>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_list()
            .entries((0..self.len()).map(|index| self.get(index)))
            .finish()
    }
}

impl<T: PartialEq, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> PartialEq
    for StableVec<T, FIRST_CHUNK, MAX_CHUNKS>
{
    fn eq(&self, other: &Self) -> bool {
        let len = self.len();
        len == other.len() && (0..len).all(|index| self.get(index) == other.get(index))
    }
}

impl<T: Eq, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> Eq
    for StableVec<T, FIRST_CHUNK, MAX_CHUNKS>
{
}

impl<T, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> Drop
    for StableVec<T, FIRST_CHUNK, MAX_CHUNKS>
{
    fn drop(&mut self) {
        let len = self.len.load(Ordering::Relaxed);
        for index in 0..len {
            let Some((chunk_index, offset)) = locate_index::<FIRST_CHUNK, MAX_CHUNKS>(index) else {
                break;
            };
            let Some(chunk) = self.chunks[chunk_index].get_mut() else {
                continue;
            };
            // SAFETY: during drop there are no remaining references to this
            // `StableVec`; indices below `len` were initialized.
            unsafe {
                (*chunk[offset].get()).assume_init_drop();
            }
        }
    }
}

impl<T: fmt::Debug, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> fmt::Debug
    for StableVecSnapshot<T, FIRST_CHUNK, MAX_CHUNKS>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.iter()).finish()
    }
}

impl<T: PartialEq, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> PartialEq
    for StableVecSnapshot<T, FIRST_CHUNK, MAX_CHUNKS>
{
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}

impl<T: Eq, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> Eq
    for StableVecSnapshot<T, FIRST_CHUNK, MAX_CHUNKS>
{
}

impl<T, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize>
    StableVecSnapshot<T, FIRST_CHUNK, MAX_CHUNKS>
{
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, index: usize) -> &T {
        assert!(
            index < self.len,
            "stable vector snapshot index out of bounds"
        );
        self.vec.get(index)
    }

    pub fn first(&self) -> Option<&T> {
        (!self.is_empty()).then(|| self.get(0))
    }

    pub fn iter(&self) -> StableVecIter<'_, T, FIRST_CHUNK, MAX_CHUNKS> {
        StableVecIter {
            snapshot: self,
            index: 0,
        }
    }
}

impl<T: Clone, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize>
    StableVecSnapshot<T, FIRST_CHUNK, MAX_CHUNKS>
{
    pub fn to_vec(&self) -> Vec<T> {
        self.iter().cloned().collect()
    }
}

impl<'a, T, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> Iterator
    for StableVecIter<'a, T, FIRST_CHUNK, MAX_CHUNKS>
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.snapshot.len {
            return None;
        }
        let item = self.snapshot.get(self.index);
        self.index += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.snapshot.len - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const FIRST_CHUNK: usize, const MAX_CHUNKS: usize> ExactSizeIterator
    for StableVecIter<'a, T, FIRST_CHUNK, MAX_CHUNKS>
{
}

fn locate_index<const FIRST_CHUNK: usize, const MAX_CHUNKS: usize>(
    index: usize,
) -> Option<(usize, usize)> {
    let mut base = 0usize;
    for chunk_index in 0..MAX_CHUNKS {
        let capacity = chunk_capacity::<FIRST_CHUNK>(chunk_index)?;
        let end = base.checked_add(capacity)?;
        if index < end {
            return Some((chunk_index, index - base));
        }
        base = end;
    }
    None
}

fn chunk_capacity<const FIRST_CHUNK: usize>(chunk_index: usize) -> Option<usize> {
    FIRST_CHUNK.checked_shl(
        chunk_index
            .try_into()
            .expect("chunk index should fit shift width"),
    )
}
