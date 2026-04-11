# BlockPool Class - Detailed Line-by-Line Explanation

## Overview
BlockPool is the core memory management component in vLLM's v1 API that manages KV cache blocks for all requests. It handles allocation, freeing, caching (for prefix caching), and eviction of GPU memory blocks that store key-value cache data.

---

## Directory Structure & Integration

```
vLLM v1 KV Cache Management Architecture
├── Scheduler (vllm/v1/core/sched/scheduler.py)
│   └── Creates and owns KVCacheManager
├── KVCacheManager (vllm/v1/core/kv_cache_manager.py)
│   ├── Wraps a KVCacheCoordinator
│   └── Exposes block_pool (BlockPool instance)
├── BlockPool (vllm/v1/core/block_pool.py) ⭐ [You are here]
│   ├── Manages all KVCacheBlock instances
│   ├── Maintains FreeKVCacheBlockQueue for allocation/eviction
│   ├── Maintains BlockHashToBlockMap for prefix caching
│   └── Tracks KVCacheEvents for distributed systems
└── Supporting Classes
    ├── KVCacheBlock (vllm/v1/core/kv_cache_utils.py)
    ├── FreeKVCacheBlockQueue (vllm/v1/core/kv_cache_utils.py)
    └── KVCacheMetricsCollector (vllm/v1/core/kv_cache_metrics.py)
```

---

## CLASS: BlockPool

### 1. Imports & Logger initialization (Lines 1-26)
```python
from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,  # Event system for distributed KV transfer
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,  # Hash of block content for prefix caching
    FreeKVCacheBlockQueue,  # Queue managing free/eviction blocks
    KVCacheBlock,  # Individual block metadata
    ...
)
```
- Imports KV cache events for reporting block operations (used in distributed KV transfer)
- Imports hash types for prefix caching support
- Imports queue/block structures managed by BlockPool
- Logger for debugging block pool operations

---

## CLASS: BlockHashToBlockMap (Lines 31-125)

**Purpose**: Cache mapping between block hashes and actual KVCacheBlock objects for prefix caching.

### Structure:
```python
def __init__(self):
    self._cache: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]] = {}
```
- **Key**: `BlockHashWithGroupId` - combines block hash + KV cache group ID
- **Value**: Can be a single `KVCacheBlock` OR a dict of multiple blocks with the same hash
  - Single block: Most common case (optimization to reduce GC cost)
  - Dict: When multiple requests have identical prefixes (hash collision handling)

### Key Methods:

#### `get_one_block(key)` (Lines 64-72)
- Returns any cached block with given hash
- If multiple blocks exist, returns first one
- Returns None on cache miss

#### `insert(key, block)` (Lines 74-86)
- Inserts a newly cached block
- Converts from single block to dict if collision occurs
- Maintains all blocks with same hash without deduplication

#### `pop(key, block_id)` (Lines 88-116)
- Removes specific block_id from cache
- Returns block if found, None otherwise
- Keeps cache entry if other blocks with same hash exist

---

## CLASS: BlockPool (Lines 129-484)

### Constructor (Lines 130-177)

```python
def __init__(
    self,
    num_gpu_blocks: int,              # Total GPU blocks available
    enable_caching: bool,             # Enable prefix caching
    hash_block_size: int,             # Block size for hashing
    enable_kv_cache_events: bool = False,     # For distributed KV transfer
    metrics_collector: KVCacheMetricsCollector | None = None,  # Metrics tracking
):
```

#### Line-by-line initialization:

**Lines 141-142**: Validation
```python
assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
self.num_gpu_blocks = num_gpu_blocks
```
- Ensures num_gpu_blocks is positive integer
- Stores total GPU blocks count

**Lines 143-144**: Configuration
```python
self.enable_caching = enable_caching
self.hash_block_size = hash_block_size
```
- Store prefix caching flag
- Store block size for hashing granularity

**Lines 145-148**: Create all blocks
```python
self.blocks: list[KVCacheBlock] = [
    KVCacheBlock(idx) for idx in range(num_gpu_blocks)
]
```
- Initialize all block objects with metadata
- Each block has:
  - `block_id`: 0 to num_gpu_blocks-1
  - `ref_cnt`: Reference count (0 = free/in queue, >0 = allocated)
  - `block_hash`: Hash for prefix caching (None initially)
  - Linked list pointers for free queue

**Lines 149-152**: Initialize free block queue
```python
self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
```
- Creates doubly-linked list of all blocks initially
- All blocks start as free (available for allocation)
- Queue maintains LRU (Least Recently Used) order for eviction

**Lines 154-155**: Initialize prefix cache
```python
self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
```
- Empty when initialized
- Populated when blocks become full and cacheable

**Lines 157-162**: Create null block
```python
self.null_block = self.free_block_queue.popleft()
self.null_block.is_null = True
```
- Reserve first block as special "null" placeholder
- Used for alignment in sparse attention (e.g., sliding window)
- Never cached, never evicted, never frees (special care needed)
- Effectively reduces usable blocks by 1

**Lines 164-167**: KV events support
```python
self.enable_kv_cache_events = enable_kv_cache_events
self.kv_event_queue: list[KVCacheEvent] = []
```
- For reporting block operations to distributed KV transfer system
- Queue accumulates events (BlockStored, BlockRemoved, AllBlocksCleared)
- Used for KV cache synchronization across machines

**Lines 169**: Metrics tracking
```python
self.metrics_collector = metrics_collector
```
- Optional metrics collection for monitoring block usage
- Tracks block allocation, eviction, access patterns

---

### Method: get_cached_block() (Lines 179-205)

**Purpose**: Retrieve cached blocks for prefix caching hit

```python
def get_cached_block(
    self, 
    block_hash: BlockHash,
    kv_cache_group_ids: list[int]
) -> list[KVCacheBlock] | None:
```

**Process**:
1. Iterates through each KV cache group
2. For each group, computes `BlockHashWithGroupId` (hash + group_id)
3. Looks up in `cached_block_hash_to_block`
4. Returns None if ANY group misses cache
5. Returns list of blocks if ALL groups hit

**Use Case**: When new request's prefix matches existing cached blocks, reuse those blocks instead of allocating new ones

---

### Method: cache_full_blocks() (Lines 207-318)

**Purpose**: Cache newly full blocks for future prefix caching reuse

```python
def cache_full_blocks(
    self,
    request: Request,
    blocks: list[KVCacheBlock],
    num_cached_blocks: int,         # Already cached blocks count
    num_full_blocks: int,           # Total full blocks now
    block_size: int,                # Tokens per block
    kv_cache_group_id: int,
) -> None:
```

**Process**:

**Lines 220-223**: Early exit if no new blocks to cache
```python
if num_cached_blocks >= num_full_blocks:
    return
new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
```

**Lines 224-234**: Handle block size differences
```python
if block_size == self.hash_block_size:
    block_hashes: BlockHashList = request.block_hashes
else:
    # Different KV cache groups may have different block sizes
    # Recalculate hashes at granularity of actual block_size
    block_hashes = BlockHashListWithBlockSize(
        request.block_hashes, 
        self.hash_block_size, 
        block_size
    )
```
- Common case: block_size equals hash_block_size
- Edge case: Groups have different block sizes (e.g., MoE with different attention patterns)

**Lines 236-238**: Prepare event tracking
```python
new_hashes: list[ExternalBlockHash] | None = (
    [] if self.enable_kv_cache_events else None
)
```

**Lines 239-271**: Cache each block
```python
for i, blk in enumerate(new_full_blocks):
    if blk.is_null:  # Skip sparse attention/Mamba null blocks
        continue
    assert blk.block_hash is None  # Block not yet cached
    
    block_hash = new_block_hashes[i]
    block_hash_with_group_id = make_block_hash_with_group_id(
        block_hash, kv_cache_group_id
    )
    
    # Update block's hash and insert into cache
    blk.block_hash = block_hash_with_group_id
    self.cached_block_hash_to_block.insert(
        block_hash_with_group_id, blk
    )
    
    if new_hashes is not None:
        new_hashes.append(maybe_convert_block_hash(block_hash))
```

**Lines 272-318**: Generate KV cache event
```python
if self.enable_kv_cache_events:
    parent_block_hash = (
        None if num_cached_blocks == 0
        else maybe_convert_block_hash(block_hashes[num_cached_blocks - 1])
    )
    
    self.kv_event_queue.append(
        BlockStored(
            block_hashes=new_hashes,
            parent_block_hash=parent_block_hash,      # Chain info
            token_ids=request.all_token_ids[...],    # What tokens stored
            block_size=block_size,
            lora_id/lora_name,                        # LoRA metadata
            medium=MEDIUM_GPU,
        )
    )
```
- Reports to distributed KV system that new blocks are cached
- Includes chain information (parent block) for reconstruction

---

### Method: get_new_blocks() (Lines 320-354)

**Purpose**: Allocate new free blocks for a request

```python
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
```

**Process**:

**Lines 321-323**: Validation
```python
if num_blocks > self.get_num_free_blocks():
    raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
```
- Ensures sufficient free blocks available
- Raises error if OOM (out of memory)

**Lines 325**: Extract blocks from free queue
```python
ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
```
- Removes from eviction order (LRU first)
- These blocks are now "owned" and allocated

**Lines 327-343**: Process allocated blocks
```python
if self.enable_caching:
    for block in ret:
        self._maybe_evict_cached_block(block)  # If cached, remove from cache
        assert block.ref_cnt == 0
        block.ref_cnt += 1  # Mark as in-use (ref_cnt=1)
        if self.metrics_collector:
            self.metrics_collector.on_block_allocated(block)
else:
    for block in ret:
        assert block.ref_cnt == 0
        block.ref_cnt += 1
        if self.metrics_collector:
            self.metrics_collector.on_block_allocated(block)
```
- If caching enabled: Check if block is cached and evict it
- Increment reference count from 0 → 1 (allocated)
- Track allocation for metrics

---

### Method: _maybe_evict_cached_block() (Lines 356-390)

**Purpose**: Evict a block from prefix cache if it's cached

```python
def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
```

**Process**:

**Lines 361-363**: Update metrics
```python
if self.metrics_collector:
    self.metrics_collector.on_block_evicted(block)
```

**Lines 365-368**: Check if block is cached
```python
block_hash = block.block_hash
if block_hash is None:
    return False  # Not cached, nothing to evict
```

**Lines 370-375**: Remove from cache
```python
if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
    return False  # Not in cache map, failed to pop
```

**Lines 377**: Reset hash metadata
```python
block.reset_hash()  # Set block_hash back to None
```

**Lines 379-390**: Generate KV event
```python
if self.enable_kv_cache_events:
    self.kv_event_queue.append(
        BlockRemoved(
            block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
            medium=MEDIUM_GPU,
        )
    )
```
- Reports eviction to distributed KV system

**Returns**: True if evicted, False if not cached

---

### Method: touch() (Lines 392-406)

**Purpose**: Increase reference count when block is reused (cache hit)

```python
def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
```

**Process**:
```python
for block in blocks:
    if block.ref_cnt == 0 and not block.is_null:
        self.free_block_queue.remove(block)  # Remove from free queue
    block.ref_cnt += 1  # Mark in-use
    if self.metrics_collector:
        self.metrics_collector.on_block_accessed(block)
```

**Key Logic**:
- ref_cnt=0 means block is in free queue (eviction candidate)
- If cache hit: block is back in use, must remove from free queue
- Increment ref_cnt (now multiple requests may own this block)
- Track access for LRU positioning

---

### Method: free_blocks() (Lines 408-424)

**Purpose**: Release blocks when request finishes

```python
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
```

**Process**:

**Lines 419-420**: Materialize iterable and decrement all
```python
blocks_list = list(ordered_blocks)
for block in blocks_list:
    block.ref_cnt -= 1
```

**Lines 421-424**: Return freed blocks to queue
```python
self.free_block_queue.append_n(
    [block for block in blocks_list 
     if block.ref_cnt == 0 and not block.is_null]
)
```

**Key Logic**:
- Blocks ordered by eviction priority (least useful first)
- Only blocks with ref_cnt=0 go back to free queue (no other requests own them)
- Null blocks never returned to queue
- Queue maintains LRU order for future evictions

---

### Method: evict_blocks() (Lines 426-441)

**Purpose**: Evict specific blocks from prefix cache by ID

```python
def evict_blocks(self, block_ids: set[int]) -> None:
```

**Process**:
```python
for block_id in block_ids:
    assert block_id < len(self.blocks)
    block = self.blocks[block_id]
    self._maybe_evict_cached_block(block)  # Evict from cache
```

**Use Case**: 
- Manual cache invalidation
- Worker reports blocks to evict (from distributed KV system)
- Only evicts from prefix cache, doesn't free the block physically

---

### Method: reset_prefix_cache() (Lines 443-473)

**Purpose**: Clear all prefix caching (used in RLHF fine-tuning)

```python
def reset_prefix_cache(self) -> bool:
```

**Process**:

**Lines 450-456**: Validate state
```python
num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
if num_used_blocks != 1:  # Only null block should be used
    logger.warning("Failed to reset... blocks (%d) are not freed yet")
    return False
```

**Lines 458-465**: Clear all caches
```python
self.cached_block_hash_to_block = BlockHashToBlockMap()  # New empty cache
for block in self.blocks:
    block.reset_hash()  # Clear all hashes
```

**Lines 467-468**: Reset metrics
```python
if self.metrics_collector:
    self.metrics_collector.reset()
```

**Lines 470-473**: Generate event
```python
if self.enable_kv_cache_events:
    self.kv_event_queue.append(AllBlocksCleared())
```

---

### Method: get_num_free_blocks() (Lines 443-447)

**Purpose**: Query available free blocks

```python
def get_num_free_blocks(self) -> int:
    return self.free_block_queue.num_free_blocks
```

---

### Method: get_usage() (Lines 449-462)

**Purpose**: Get KV cache utilization percentage

```python
def get_usage(self) -> float:
    total_gpu_blocks = self.num_gpu_blocks - 1  # Exclude null block
    if not total_gpu_blocks:
        return 0
    return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)
```

**Returns**: 0.0 (empty) to 1.0 (full)

---

### Method: take_events() (Lines 464-473)

**Purpose**: Atomically retrieve and clear KV cache events

```python
def take_events(self) -> list[KVCacheEvent]:
    if not self.enable_kv_cache_events:
        return []
    events = self.kv_event_queue
    self.kv_event_queue = []  # Clear for next batch
    return events
```

**Use Case**: Called by distributed KV connector to send block operations to other nodes

---

## Supporting Data Structure: FreeKVCacheBlockQueue

Located in `kv_cache_utils.py`. Key properties:

```python
class FreeKVCacheBlockQueue:
    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        # Creates doubly-linked list from blocks
        # Fake head/tail to reduce branching
        # Block order: block ID initially, LRU after reuse
```

### Key Methods BlockPool uses:
- `popleft_n(num_blocks)`: Allocate blocks from front (next to be freed)
- `remove(block)`: Remove block from middle (cache hit on freed block)
- `append_n(blocks)`: Return blocks to queue (in eviction order)
- `num_free_blocks`: Property tracking free count

---

## Integration with Other Components

### 1. **Scheduler** → BlockPool Path

```
Scheduler.__init__()
  └─→ KVCacheManager(kv_cache_config)
       └─→ KVCacheCoordinator.create()
            └─→ BlockPool(
                  num_gpu_blocks=config.num_gpu_blocks,
                  enable_caching=config.enable_prefix_caching,
                  hash_block_size=block_size,
                  enable_kv_cache_events=config.enable_kv_cache_events,
                  metrics_collector=collector
                )
```

### 2. **Scheduler Uses BlockPool** for:

**During Scheduling**:
```python
# In scheduler.schedule() method:
req_to_new_blocks[req.request_id] = (
    self.kv_cache_manager.block_pool.get_new_blocks(num_blocks_needed)
)
```

**After Generation**:
```python
# In scheduler.update_from_output() method:
block_pool.cache_full_blocks(
    request=request,
    blocks=request.block_table,
    num_cached_blocks=request.num_prefix_cached_blocks,
    num_full_blocks=new_num_full_blocks,
    block_size=self.block_size,
    kv_cache_group_id=group_id,
)
```

**On Request Finish**:
```python
# Free blocks back to pool:
blocks_to_free = request.get_blocks_for_eviction()
block_pool.free_blocks(blocks_to_free)
```

### 3. **KVCacheManager Interface to Scheduler**

```python
class KVCacheManager:
    @property
    def usage(self) -> float:
        return self.block_pool.get_usage()
    
    def block_pool(self):
        return self.coordinator.block_pool
```

### 4. **KV Connector Integration** (Distributed Systems)

KV events go from BlockPool → Scheduler → KV Connector:
```python
# Scheduler collects events:
kv_events = self.block_pool.take_events()

# Sends to KV connector for distribution:
scheduler_output.kv_events = kv_events
```

### 5. **Metrics Collector** Integration

```python
class KVCacheMetricsCollector:
    def on_block_allocated(block): ...
    def on_block_evicted(block): ...
    def on_block_accessed(block): ...
    def reset(): ...
```

BlockPool calls these methods for:
- Tracking block lifetime
- Computing hit/miss rates
- Measuring fragmentation
- Monitoring eviction patterns

---

## Key State Transitions

### Block Lifecycle:

```
FREE BLOCK (ref_cnt=0, in queue)
    ↓ get_new_blocks()
ALLOCATED (ref_cnt=1, removed from queue)
    ↓ [tokens filled]
FULL & CACHED (block_hash set, in cache_map)
    ↓ new request reuses (touch())
MULTI-OWNER (ref_cnt>1)
    ↓ first owner free_blocks()
PARTIALLY_USED (ref_cnt>0)
    ↓ last owner free_blocks()
EVICTION_CANDIDATE (ref_cnt=0, back in queue for LRU eviction)
    ↓ get_new_blocks() allocates it again
ALLOCATED (cycle repeats)
```

### Prefix Cache Hit Flow:

```
Request with matching prefix
    ↓ get_cached_block(block_hash, group_ids)
Cache HIT → Returns cached blocks
    ↓ touch(cached_blocks)  
Increment ref_cnt, remove from free queue
    ↓ Request reuses blocks, faster generation
No physical allocation needed ✓
```

### Cache Eviction Flow:

```
Memory pressure: need N new blocks
    ↓ get_new_blocks(N)
Free queue has N blocks
    ↓ popleft_n(N)
Check each: _maybe_evict_cached_block()
    ↓ Remove from cached_block_hash_to_block
    ↓ Reset hash_metadata
Allocate clean blocks ✓
```

---

## Performance Characteristics

### Allocation - `get_new_blocks(N)`:
- **Time**: O(N) + O(N) for cache eviction checks = O(N)
- **Space**: Creates N block objects once at init

### Prefix Cache Hit - `get_cached_block()`:
- **Time**: O(num_groups) for hash lookups = O(1) typically
- **Space**: O(cached_blocks) in hash table

### Block Freeing - `free_blocks()`:
- **Time**: O(num_freed_blocks)
- **Append to queue**: O(1) per block

### Eviction(LRU) - via `free_block_queue`:
- **Time**: O(1) to track LRU order
- **Space**: Doubly-linked list pointers (no extra allocation)

---

## Configuration Interactions

| Config | BlockPool Impact |
|--------|-----------------|
| `enable_prefix_caching` | Activates `cache_full_blocks()`, `get_cached_block()` |
| `num_gpu_blocks` | Total blocks: `num_gpu_blocks - 1` (1 reserved for null) |
| `block_size` | Used in prefix cache (hash computation granularity) |
| `enable_kv_cache_events` | Generates KVCacheEvent for distributed KV sync |
| `log_stats` → metrics_collector | Tracks block lifecycle for monitoring |

---

## Edge Cases Handled

1. **Sparse Attention (Sliding Window, Mamba)**:
   - Null blocks (is_null=True) never cached/freed
   - Skipped in cache_full_blocks()

2. **Multi-group KV Cache**:
   - BlockHashWithGroupId combines hash + group_id
   - Each group may have different block_size
   - BlockHashListWithBlockSize recalculates hashes

3. **Hash Collisions**:
   - Multiple blocks with same hash allowed
   - Stored in dict when collision occurs
   - Reduced GC overhead vs list

4. **OOM Handling**:
   - get_new_blocks() raises ValueError if insufficient blocks
   - Scheduler handles by not scheduling new requests

5. **Distributed KV Transfer**:
   - Events queue blocks operations for worker notification
   - AllBlocksCleared event for cache reset
   - Parent block hash for chain reconstruction

---

## Summary

**BlockPool** is the fundamental KV cache allocator that:
1. **Manages memory**: Allocates/frees GPU blocks
2. **Enables prefix caching**: Discovers and reuses identical cache regions
3. **Handles eviction**: LRU ordering via FreeKVCacheBlockQueue
4. **Coordinates distribution**: Events for multi-node KV sync
5. **Tracks metrics**: Block lifecycle for monitoring

It sits between the **Scheduler** (which makes allocation requests) and **KVCacheCoordinator** (which manages multiple cache groups), providing O(1) typical allocation and prefix cache hits with minimal GC overhead.
