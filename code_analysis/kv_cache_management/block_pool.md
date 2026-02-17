# BlockPool Implementation

## Overview

The `BlockPool` class in [vllm/v1/core/block_pool.py](../vllm/v1/core/block_pool.py) manages all KV-cache blocks for vLLM's KV cache system. It provides a high-level interface for:
- Allocating and freeing KV cache blocks
- Managing prefix caching for identical KV cache prefixes
- Tracking block eviction through a free block queue
- Event tracking for distributed KV cache managers

## Architecture

### Class Hierarchy

```
BlockPool
├── FreeKVCacheBlockQueue (manages free blocks)
└── BlockHashToBlockMap (caches blocks by hash)
```

### Key Data Structures

#### BlockPool
```python
class BlockPool:
    blocks: list[KVCacheBlock]  # All KV cache blocks
    free_block_queue: FreeKVCacheBlockQueue
    cached_block_hash_to_block: BlockHashToBlockMap
    null_block: KVCacheBlock  # Placeholder block with block_id=0
    kv_event_queue: list[KVCacheEvent]  # Events for distributed systems
    metrics_collector: KVCacheMetricsCollector | None
```

#### BlockHashToBlockMap
A cache mapping block hashes to blocks used for prefix caching:
- Maps `BlockHashWithGroupId` → `KVCacheBlock | dict[int, KVCacheBlock]`
- Supports multiple blocks per hash (stores as dict when needed)
- Avoids deduplication to keep block IDs stable for append-only block tables

## Core Operations

### 1. Cache Full Blocks (`cache_full_blocks`)

Updates block hash metadata and caches full blocks for prefix caching.

```python
def cache_full_blocks(
    self,
    request: Request,
    blocks: list[KVCacheBlock],
    num_cached_blocks: int,
    num_full_blocks: int,
    block_size: int,
    kv_cache_group_id: int,
)
```

**Process:**
1. Retrieves block hashes from the request at appropriate granularity
2. Handles hash block size mismatches (recalculates hashes at larger granularity)
3. Updates `block_hash` metadata on each new full block
4. Inserts blocks into the `cached_block_hash_to_block` cache
5. Records `BlockStored` events if KV cache events are enabled

**Key Design Decisions:**
- Skips null blocks (used in sparse attention/sliding window)
- Handles multi-group scenarios by including group ID in cache key

### 2. Get Cached Block (`get_cached_block`)

Retrieves cached blocks by their hash, checking each KV cache group.

```python
def get_cached_block(
    self,
    block_hash: BlockHash,
    kv_cache_group_ids: list[int],
) -> list[KVCacheBlock] | None
```

**Returns:**
- List of cached blocks (one per group) if all groups have a cache hit
- `None` if any group misses the cache

### 3. Allocate New Blocks (`get_new_blocks`)

Allocates new blocks from the free block pool.

```python
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]
```

**Process:**
1. Checks sufficient free blocks exist
2. Pops blocks from `free_block_queue`
3. For each popped block:
   - Calls `_maybe_evict_cached_block()` to evict if cached
   - Increments reference count from 0 to 1
   - Records allocation metrics

### 4. Block Eviction (`_maybe_evict_cached_block`)

Evicts a cached block before allocation, removing its hash metadata.

```python
def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool
```

**Process:**
1. Records eviction metrics (lifetime, idle time, reuse gaps)
2. Retrieves block's hash
3. Removes block from `cached_block_hash_to_block` cache
4. Resets block's hash metadata via `reset_hash()`
5. Records `BlockRemoved` events if enabled

### 5. Touch Operation (`touch`)

Increases reference count and removes block from free queue (used when block hits prefix cache).

```python
def touch(self, blocks: Sequence[KVCacheBlock]) -> None
```

### 6. Free Blocks (`free_blocks`)

Decrements reference counts and returns freed blocks to the free queue.

```python
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None
```

**Ordering:** Blocks should be ordered by eviction priority (least recently used first).

### 7. Evict Blocks by ID (`evict_blocks`)

Evicts specific cached blocks by their IDs without freeing them from the pool.

```python
def evict_blocks(self, block_ids: set[int]) -> None
```

Used primarily when KV connectors (distributed KV cache) need to evict prefixes.

## Reference Counting & Block Lifecycle

Each block has a reference count (`ref_cnt`) that determines its state:

```
ref_cnt = 0: Block is in free queue (eviction candidate)
ref_cnt > 0: Block is allocated to one or more requests
```

**Lifecycle:**
1. **Allocated:** `ref_cnt = 1` (via `get_new_blocks`)
2. **Shared:** `ref_cnt += 1` (via `touch` when other request uses the same prefix)
3. **Freed:** `ref_cnt -= 1` (via `free_blocks`)
4. **In Free Queue:** When `ref_cnt = 0`, block is eligible for eviction

## Prefix Caching Strategy

### How Block Hashing Works

Blocks are cached using their content hash plus the KV cache group ID:
- **Block Hash**: SHA-256 or xxHash of block tokens and metadata (multimodal features, LoRA, salt)
- **BlockHashWithGroupId**: `block_hash + group_id.to_bytes(4, "big")`

### Cache Hit Flow

1. Request computes block hashes during tokenization
2. `find_longest_cache_hit()` searches for matching blocks in cache
3. Hit blocks are "touched" (ref_cnt incremented, removed from free queue)
4. Request reuses these blocks, avoiding recomputation

### Multi-Group Handling

When different KV cache groups have different block sizes:
- Hash blocks are computed at `hash_block_size` granularity
- When actual `block_size > hash_block_size`, hashes are recalculated at larger granularity
- All groups must have a cache hit for the block to be reused

## Metrics & Events

### Metrics Collection (`KVCacheMetricsCollector`)

When enabled, tracks:
- Block allocation and eviction times
- Block lifetime and idle periods
- Block reuse patterns

### KV Cache Events

When `enable_kv_cache_events=True`:
- `BlockStored`: Records which blocks are cached, with parent relationships
- `BlockRemoved`: Records which blocks are evicted
- `AllBlocksCleared`: Records when entire cache is reset

Used by distributed KV cache systems to synchronize state.

## Null Block Special Handling

The null block (`block_id=0`):
- Created at initialization with `is_null=True`
- Never freed back to free queue
- Used as placeholder in sparse attention scenarios
- Reference count not maintained (never freed)

## Reset Operation (`reset_prefix_cache`)

Completely resets the prefix cache:

```python
def reset_prefix_cache(self) -> bool
```

**Preconditions:** Only one block in use (the null block)

**Process:**
1. Creates new empty `BlockHashToBlockMap`
2. Resets all block hashes
3. Clears metrics
4. Records `AllBlocksCleared` event

**Use Cases:**
- RLHF flows after weight updates
- Benchmarking with cache reset

## Usage Example

```python
# Create a block pool
block_pool = BlockPool(
    num_gpu_blocks=4096,
    enable_caching=True,
    hash_block_size=16,
    enable_kv_cache_events=True,
    metrics_collector=KVCacheMetricsCollector(),
)

# Allocate blocks for a request
new_blocks = block_pool.get_new_blocks(num_blocks=10)

# Cache full blocks from a request
block_pool.cache_full_blocks(
    request=request,
    blocks=request_blocks,
    num_cached_blocks=0,
    num_full_blocks=10,
    block_size=16,
    kv_cache_group_id=0,
)

# Find cached blocks by hash
cached = block_pool.get_cached_block(
    block_hash=request.block_hashes[0],
    kv_cache_group_ids=[0],
)
if cached:
    block_pool.touch(cached)  # Increment ref_cnt

# Free blocks when request completes
block_pool.free_blocks(blocks_to_free)
```

## Performance Considerations

### Free Block Queue
- Uses doubly-linked list in `KVCacheBlock` objects (no separate Python objects)
- O(1) remove operation (unlike Python deque)
- LRU ordering: least recently used blocks at front

### Cache Lookup
- O(1) dictionary lookup by hash
- Handles multi-block case with dict of blocks

### No Deduplication
- Intentional design: prevents block ID changes that would break append-only tables
- Each request keeps separate copies of identical prefixes

## Integration Points

- **Scheduler**: Uses `BlockPool` for allocation/freeing
- **Coordinator**: Manages multiple managers that use `BlockPool`
- **KV Connectors**: Call `evict_blocks()` for distributed eviction
- **Metrics System**: Optionally collects block lifecycle metrics
- **Events System**: Publishes events for distributed sync

## Key Design Principles

1. **Immutability of Block IDs**: No deduplication to keep block IDs stable
2. **O(1) Operations**: Free queue supports O(1) remove and append
3. **Group Awareness**: Handles multiple KV cache groups with different block sizes
4. **Event Publishing**: Supports distributed KV cache systems
5. **Metrics Integration**: Optional sampling-based block lifecycle tracking
