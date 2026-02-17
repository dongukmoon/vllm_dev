# KV Cache Manager System - Complete Architecture Guide

## System Overview

The vLLM KV Cache Manager is a sophisticated multi-layered system for managing Key-Value cache blocks in large language model inference. It handles block allocation, prefix caching, eviction, and supports multiple attention mechanisms through a flexible composition architecture.

## Architecture Layers

```
┌────────────────────────────────────────────────────────────────────┐
│                        SCHEDULER LAYER                             │
│                     (Orchestrates compute)                         │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                    KVCacheManager (Public API)                     │
│  - allocate_slots()                                                │
│  - get_computed_blocks()                                           │
│  - cache_blocks_for_requests()                                     │
│  - free_blocks_for_request()                                       │
│                                                                    │
│  ├─ KVCacheBlocks (Result Type)                                   │
│  └─ PrefixCacheStats (Optional)                                   │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│              KVCacheCoordinator (Multi-Group Orchestrator)         │
│  - Coordinates multiple attention types/groups                     │
│  - find_longest_cache_hit() [abstract method]                      │
│  - get_num_blocks_to_allocate()                                    │
│  - allocate_new_blocks()                                           │
│  - cache_blocks()                                                  │
│  - free()                                                          │
│                                                                    │
│  ├─ Concrete Coordinators:                                        │
│  │  ├─ FullAttentionCoordinator                                   │
│  │  ├─ SlidingWindowCoordinator                                   │
│  │  └─ ... (for each attention config)                            │
│  │                                                                 │
│  └─ BlockPool (Shared)                                            │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│        SingleTypeKVCacheManager[] (Per-Group Per-Type Managers)   │
│  - Handles one attention type + one KV cache group                 │
│  - allocate_new_blocks()                                           │
│  - get_num_blocks_to_allocate()                                    │
│  - cache_blocks()                                                  │
│  - free()                                                          │
│  - get_num_skipped_tokens() [abstract]                             │
│                                                                    │
│  ├─ Concrete Managers:                                            │
│  │  ├─ FullAttentionManager                                       │
│  │  ├─ SlidingWindowAttentionManager                              │
│  │  ├─ CrossAttentionManager (encoder-decoder)                    │
│  │  ├─ MLAAttentionManager                                        │
│  │  └─ MambaStateManager                                          │
│  │                                                                 │
│  └─ req_to_blocks: dict[str, list[KVCacheBlock]]                  │
│     (Tracks blocks per request)                                    │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                        BlockPool (Shared)                          │
│  - get_new_blocks()                                                │
│  - touch()                                                         │
│  - free_blocks()                                                   │
│  - cache_full_blocks()                                             │
│  - get_cached_block()                                              │
│  - evict_blocks()                                                  │
│                                                                    │
│  ├─ FreeKVCacheBlockQueue                                         │
│  │  ├─ Doubly-linked list of free blocks (LRU ordering)          │
│  │  ├─ O(1) arbitrary removal                                     │
│  │  └─ Batch append/popleft operations                            │
│  │                                                                 │
│  ├─ BlockHashToBlockMap                                           │
│  │  ├─ Cache: BlockHashWithGroupId → KVCacheBlock                │
│  │  └─ Enables prefix caching                                     │
│  │                                                                 │
│  └─ KVCacheBlock[] (All blocks)                                   │
│     ├─ Metadata: block_id, ref_cnt, block_hash                   │
│     └─ List pointers: prev_free_block, next_free_block           │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                  GPU KV Cache Memory                               │
│                 (num_blocks × block_size)                          │
└────────────────────────────────────────────────────────────────────┘
```

## Component Reference

### 1. KVCacheManager ([kv_cache_manager.md](kv_cache_manager.md))

**Responsibility:** High-level orchestration and scheduler interface

**Key Methods:**
- `allocate_slots()` - Allocate blocks for new tokens
- `get_computed_blocks()` - Find cached prefix blocks
- `cache_blocks_for_requests()` - Update block hashes for caching
- `free_blocks_for_request()` - Release blocks on completion

**Design Pattern:** Facade over coordinator and block pool

### 2. KVCacheCoordinator ([kv_cache_coordinator.md](kv_cache_coordinator.md))

**Responsibility:** Multi-group coordination and abstract cache searching

**Key Methods:**
- `get_num_blocks_to_allocate()` - Calculate total blocks needed
- `find_longest_cache_hit()` - Abstract prefix cache search
- `allocate_new_blocks()` - Delegate to all managers
- `cache_blocks()` - Cache blocks across all groups

**Design Pattern:** Abstract base with concrete implementations per attention config

### 3. SingleTypeKVCacheManager ([single_type_kv_cache_manager.md](single_type_kv_cache_manager.md))

**Responsibility:** Per-group per-attention-type block management

**Key Methods:**
- `allocate_new_blocks()` - Allocate for this group
- `get_num_blocks_to_allocate()` - Complex logic for this group
- `cache_blocks()` - Cache this group's blocks
- `get_num_skipped_tokens()` - Abstract (attention-specific)

**Design Pattern:** Abstract base with concrete implementations per attention type

### 4. BlockPool ([block_pool.md](block_pool.md))

**Responsibility:** Low-level block lifecycle and prefix cache storage

**Key Methods:**
- `get_new_blocks()` - Allocate from free pool
- `free_blocks()` - Return to free pool
- `touch()` - Prevent eviction and remove from free queue
- `cache_full_blocks()` - Store block hashes for prefix caching
- `_maybe_evict_cached_block()` - Evict on allocation

**Design Pattern:** Resource manager with eviction policy

### 5. KVCacheBlock & FreeKVCacheBlockQueue ([kv_cache_utils.md](kv_cache_utils.md))

**Responsibility:** Block metadata and free pool data structure

**Key Types:**
- `KVCacheBlock` - Metadata (block_id, ref_cnt, block_hash, linked list pointers)
- `FreeKVCacheBlockQueue` - LRU doubly-linked list with O(1) removal
- `BlockHash`, `BlockHashWithGroupId` - Composite keys for hashing

**Design Pattern:** Zero-allocation linked list in objects, immutable composite keys

### 6. KVCacheMetricsCollector ([kv_cache_metrics.md](kv_cache_metrics.md))

**Responsibility:** Optional metrics collection on block lifecycle

**Key Methods:**
- `on_block_allocated()` - Track allocation
- `on_block_accessed()` - Track prefix cache hits
- `on_block_evicted()` - Compute and record eviction stats

**Design Pattern:** Sampling-based with configurable overhead

## Key Interactions

### Request Lifecycle

```
1. NEW REQUEST
   └─→ allocate_slots()
       ├─→ get_computed_blocks()  # Check cache
       │   └─→ find_longest_cache_hit()
       │       └─→ Coordinator → Manager → BlockPool
       │
       └─→ (Allocate new blocks if cache miss)
           └─→ get_num_blocks_to_allocate()
               └─→ allocate_new_blocks()
                   └─→ BlockPool.get_new_blocks()

2. COMPUTE
   └─→ Worker fills allocated blocks with tokens

3. PREFIX CACHE HIT
   └─→ allocate_new_computed_blocks()
       └─→ BlockPool.touch()  # Increment ref_cnt, remove from free

4. CACHE BLOCKS
   └─→ cache_blocks_for_requests()
       └─→ BlockPool.cache_full_blocks()
           └─→ BlockHashToBlockMap.insert()

5. COMPLETION
   └─→ free_blocks_for_request()
       └─→ BlockPool.free_blocks()
           └─→ FreeKVCacheBlockQueue.append_n()
```

### Prefix Caching Flow

```
Request hashes tokens
    ↓
find_longest_cache_hit(block_hashes)
    ↓
BlockHashToBlockMap lookup
    ├─ Cache Miss → Find mismatch point → Return blocks up to mismatch
    └─ Cache Miss on first → Return empty
    ↓
Matched blocks returned
    ↓
allocate_new_computed_blocks(matched_blocks)
    ↓
BlockPool.touch(matched_blocks)
    ├─ ref_cnt += 1
    └─ Remove from free queue (if ref_cnt was 0)
    ↓
Request reuses matched blocks
```

### Block Eviction Flow

```
Insufficient free blocks for allocation
    ↓
BlockPool.get_new_blocks(n)
    ├─ FreeKVCacheBlockQueue.popleft_n(n)
    │   └─ LRU blocks from queue front
    ↓
For each popped block:
    └─→ _maybe_evict_cached_block()
        ├─ If block_hash exists:
        │   ├─ BlockHashToBlockMap.pop()
        │   ├─ block.reset_hash()
        │   └─ Record BlockRemoved event
        └─ Record metrics (lifetime, idle, reuse gaps)
    ↓
Block now ready for reallocation
```

## Configuration Examples

### Full Attention (Standard GPT)

```python
manager = KVCacheManager(
    kv_cache_config=KVCacheConfig(
        num_blocks=4096,
        kv_cache_groups=[
            KVCacheGroup(
                kv_cache_spec=FullAttentionSpec(block_size=16)
            )
        ]
    ),
    enable_caching=True,  # Prefix caching enabled
    hash_block_size=16,
)
```

### Sliding Window (Llama 3.1, Mistral)

```python
manager = KVCacheManager(
    kv_cache_config=KVCacheConfig(
        num_blocks=4096,
        kv_cache_groups=[
            KVCacheGroup(
                kv_cache_spec=SlidingWindowSpec(
                    block_size=16,
                    window_size=4096
                )
            )
        ]
    ),
    enable_caching=True,
)
```

### Encoder-Decoder (Whisper)

```python
manager = KVCacheManager(
    kv_cache_config=KVCacheConfig(
        num_blocks=4096,
        kv_cache_groups=[
            # Encoder self-attention
            KVCacheGroup(
                kv_cache_spec=FullAttentionSpec(block_size=16)
            ),
            # Decoder self-attention
            KVCacheGroup(
                kv_cache_spec=FullAttentionSpec(block_size=16)
            ),
            # Cross-attention
            KVCacheGroup(
                kv_cache_spec=CrossAttentionSpec(block_size=16)
            )
        ]
    ),
    enable_caching=True,
)
```

### Distributed (Tensor Parallelism)

```python
manager = KVCacheManager(
    kv_cache_config=config,
    dcp_world_size=8,  # 8-way tensor parallelism
    pcp_world_size=1,
    # Block size internally expanded by 8x
    hash_block_size=16,
)
```

## Data Structures - Quick Reference

| Type | Purpose | Location |
|------|---------|----------|
| `KVCacheBlock` | Block metadata | kv_cache_utils.md |
| `KVCacheBlocks` | Result type (groups of blocks) | kv_cache_manager.md |
| `BlockHash` | Hash of block contents | kv_cache_utils.md |
| `BlockHashWithGroupId` | Composite key for cache | kv_cache_utils.md |
| `FreeKVCacheBlockQueue` | LRU free block list | kv_cache_utils.md |
| `BlockHashToBlockMap` | Prefix cache storage | block_pool.md |
| `KVCacheMetricsCollector` | Block lifecycle metrics | kv_cache_metrics.md |
| `BlockMetricsState` | Per-block statistics | kv_cache_metrics.md |

## Important Patterns

### Reference Counting

All blocks track `ref_cnt`:
```
ref_cnt = 0   → In free queue (eligible for eviction)
ref_cnt > 0   → Allocated to one or more requests
```

Allocation: `ref_cnt = 0 → 1`
Prefix hit: `ref_cnt = 1 → 2` (shared by two requests)
Free: `ref_cnt = 2 → 1` (only one request now)

### Zero-Allocation Linked List

`FreeKVCacheBlockQueue` stores pointers directly in `KVCacheBlock`:
```
block.prev_free_block
block.next_free_block
```

No separate Node objects → Better cache locality

### Immutable Composite Keys

`BlockHashWithGroupId = block_hash + group_id.to_bytes(4, "big")`

One dictionary key → Simpler logic, better performance

### Abstract Method Pattern

```
KVCacheCoordinator.find_longest_cache_hit() [abstract]
    ↓ Implemented by attention-specific coordinators

SingleTypeKVCacheManager.get_num_skipped_tokens() [abstract]
    ↓ Implemented by attention-type-specific managers
```

### Decorator Pattern

`KVCacheManager` wraps `KVCacheCoordinator`
`KVCacheCoordinator` wraps `SingleTypeKVCacheManager[]`
`SingleTypeKVCacheManager` wraps `BlockPool`

Each layer adds logic without modifying lower layers.

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Allocate N blocks | O(N) | Batch pop from free queue |
| Free N blocks | O(N) | Batch append to free queue |
| Cache lookup | O(1) | Dictionary hash lookup |
| Block eviction | O(1) | Per-block, amortized |
| Remove from free | O(1) | Doubly-linked unlink |
| `get_num_blocks_to_allocate` | O(num_groups) | Typically 1-3 groups |
| `find_longest_cache_hit` | O(num_hashes) | Typically 1-128 hashes |

## Memory Layout Example

**Assuming 4096 total blocks:**

```
Block Pool:
├─ Blocks 0-3999: KVCacheBlock objects
├─ Block 0: null_block (special, never evicted)
└─ Blocks 1-3999: Usable for allocation

Block Metadata: ~4096 × 200 bytes = 819 KB

Free Queue: Doubly-linked structure in blocks themselves
  (no additional allocation)

BlockHashToBlockMap: Dictionary entries for cached blocks
  ~100-500 cached blocks typical × 32 bytes = 3-16 KB

Metrics (optional, 1% sample): 
  40 blocks × 112 bytes = 4.5 KB

Total Overhead: ~1 MB for 4096 blocks
```

## Common Issues and Solutions

### Cache Misses Despite Prefix Caching

**Issue:** Expected prefix cache hits not occurring

**Causes:**
1. Hash seed mismatch (`PYTHONHASHSEED` not set)
2. Extra hash keys changing (LoRA, multimodal, cache_salt)
3. Different request sequences

**Solution:**
- Set `PYTHONHASHSEED` for reproducibility
- Check extra key generation logic
- Use `prefix_cache_stats` to diagnose

### Out of Memory Preemptions

**Issue:** Frequent OOM and preemptions

**Causes:**
1. Block size too large for requests
2. Inefficient eviction policy (LRU may not be optimal)
3. Many requests with short unique prefixes

**Solution:**
- Reduce `batch_size`
- Increase `num_gpu_blocks`
- Use scheduled prefetching

### Uneven Block Usage

**Issue:** Some blocks heavily used, others unused

**Causes:**
1. Common prefix sharing (expected)
2. Sliding window with many old requests
3. Null blocks taking space

**Solution:**
- Monitor via `usage` metric
- Check `get_num_common_prefix_blocks()`
- Profile with metrics collector

## Extension Guide

### Adding New Attention Type

1. **Create KVCacheSpec subclass** (in kv_cache_interface.py)
   ```python
   @dataclass
   class NewAttentionSpec(KVCacheSpec):
       special_param: int
   ```

2. **Create Manager subclass** (in single_type_kv_cache_manager.py)
   ```python
   class NewAttentionManager(SingleTypeKVCacheManager):
       def get_num_skipped_tokens(self, total_tokens):
           # Attention-specific logic
   ```

3. **Create Coordinator subclass** (in kv_cache_coordinator.py)
   ```python
   class NewAttentionCoordinator(KVCacheCoordinator):
       def find_longest_cache_hit(self, block_hashes, max_hit_len):
           # Attention-specific cache search
   ```

4. **Update factory functions**
   - `get_manager_for_kv_cache_spec()`
   - `get_kv_cache_coordinator()`

5. **System automatically uses new types**

### Custom Eviction Policy

1. **Subclass FreeKVCacheBlockQueue** (or use custom ordering)
2. **Override append/popleft** with new priority logic
3. **Pass to BlockPool** via initialization

## Summary

The KV Cache Manager system provides:

✅ **Flexible multi-attention support** - One manager per attention type
✅ **Efficient prefix caching** - Zero-copy block reuse with hashing
✅ **LRU eviction** - Automatic block reclamation with O(1) operations
✅ **Distributed support** - Event publishing + tensor parallelism
✅ **Optional metrics** - Sampling-based collection with minimal overhead
✅ **Modular design** - Easy extension for new attention types
✅ **Performance optimized** - Zero-allocation algorithms, batch operations

Perfect for high-throughput inference with LLM batching.
