# KV Cache Manager Implementation

## Overview

The [vllm/v1/core/kv_cache_manager.py](../vllm/v1/core/kv_cache_manager.py) module provides the high-level `KVCacheManager` class and `KVCacheBlocks` data structure. It serves as the primary interface between the scheduler and the KV cache subsystem, managing request-level block allocation, caching, and lifecycle.

## Architecture

### Class Hierarchy

```
KVCacheManager
├── KVCacheCoordinator (via self.coordinator)
│   ├── BlockPool
│   └── SingleTypeKVCacheManager[]
└── Components:
    ├── KVCacheBlocks (result type)
    ├── PrefixCacheStats (optional)
    └── KVCacheMetricsCollector (optional)
```

## KVCacheBlocks Class

### Purpose

Encapsulates allocation results and provides utilities for working with multi-group block data.

### Data Structure

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
    # blocks[i][j] = i-th KV cache group, j-th block of the group
```

**Design:**
- Outer dimension: KV cache groups (typically 1-3)
- Inner dimension: blocks within group (typically 1-100+)
- Tuple ensures immutability

### Operations

#### Addition Operator

```python
def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
    return KVCacheBlocks(
        tuple(
            list(itertools.chain(blk1, blk2))
            for blk1, blk2 in zip(self.blocks, other.blocks)
        )
    )
```

**Purpose:** Combine two sets of blocks

**Example:**
```python
# Concatenate cached blocks with newly allocated blocks
all_blocks = cached_blocks + newly_allocated_blocks
```

**Process:**
1. Zip corresponding groups
2. Chain blocks within each group
3. Create new tuple

#### Get Block IDs

```python
@overload
def get_block_ids(self, allow_none: Literal[False] = False) -> tuple[list[int], ...]: ...

@overload
def get_block_ids(self, allow_none: Literal[True] = True) -> tuple[list[int], ...] | None: ...

def get_block_ids(self, allow_none: bool = False) -> tuple[list[int], ...] | None:
```

**Purpose:** Convert block objects to block IDs for scheduler

**Process:**
```python
if allow_none and all(len(group) == 0 for group in self.blocks):
    return None
return tuple([blk.block_id for blk in group] for group in self.blocks)
```

**Use Cases:**
- Scheduler needs IDs for GPU kernel calls
- Avoid passing object references

#### Get Unhashed Block IDs

```python
def get_unhashed_block_ids(self) -> list[int]:
    assert len(self.blocks) == 1, "Only one group is supported"
    return [block.block_id for block in self.blocks[0] if block.block_hash is None]
```

**Purpose:** Find blocks not yet cached (blocks without hash)

**Precondition:** Single KV cache group

**Use:** Identify blocks waiting for caching

#### New Empty

```python
def new_empty(self) -> "KVCacheBlocks":
    return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))
```

**Purpose:** Create empty blocks for same group structure

**Use:** Placeholder for requests without blocks

## KVCacheManager Class

### Initialization

```python
def __init__(
    self,
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    hash_block_size: int,
    enable_caching: bool = True,
    use_eagle: bool = False,
    log_stats: bool = False,
    enable_kv_cache_events: bool = False,
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
    metrics_collector: KVCacheMetricsCollector | None = None,
)
```

### Configuration

| Parameter | Purpose |
|-----------|---------|
| `kv_cache_config` | Cache layout (num_blocks, groups, specs) |
| `max_model_len` | Maximum tokens per request |
| `hash_block_size` | Base block size for hashing |
| `enable_caching` | Enable prefix caching |
| `use_eagle` | Enable eagle speculative decoding handling |
| `log_stats` | Collect prefix cache statistics |
| `enable_kv_cache_events` | Publish distributed events |
| `dcp_world_size` | Data center parallel factor |
| `pcp_world_size` | Pipeline cache parallel factor |
| `metrics_collector` | Block lifecycle metrics (optional) |

### State

```python
self.block_pool: BlockPool  # Access via self.coordinator
self.coordinator: KVCacheCoordinator  # Multi-group orchestrator
self.empty_kv_cache_blocks: KVCacheBlocks  # Reusable empty blocks

self.enable_caching: bool
self.use_eagle: bool
self.log_stats: bool
self.metrics_collector: KVCacheMetricsCollector | None
self.prefix_cache_stats: PrefixCacheStats | None  # If log_stats
```

### Pre-constructed Empty Blocks

```python
self.empty_kv_cache_blocks = KVCacheBlocks(
    tuple(() for _ in range(self.num_kv_cache_groups))
)
```

**Purpose:** Avoid GC overhead from repeatedly creating empty blocks

**Reuse:** Returned from methods when no blocks needed

## Core Operations

### 1. Get Usage (`usage` property)

```python
@property
def usage(self) -> float:
    return self.block_pool.get_usage()
```

**Returns:** Cache usage percent (0.0 to 1.0)

**Formula:** `1.0 - (free_blocks / (total_blocks - 1))`
- Subtracts 1 for null block

### 2. Get Prefix Cache Stats (`make_prefix_cache_stats`)

```python
def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
    if not self.log_stats:
        return None
    stats = self.prefix_cache_stats
    self.prefix_cache_stats = PrefixCacheStats()
    return stats
```

**Purpose:** Retrieve and reset prefix cache statistics

**Returns:**
- `None` if logging disabled
- Cumulative stats if enabled
- Resets stats to new instance

**Statistics Tracked:**
- Number of requests
- Cache hits vs misses
- Hit rates over time

### 3. Get Computed Blocks (`get_computed_blocks`)

```python
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
    if not self.enable_caching or request.skip_reading_prefix_cache:
        return self.empty_kv_cache_blocks, 0
    
    max_cache_hit_length = request.num_tokens - 1
    computed_blocks, num_new_computed_tokens = (
        self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length
        )
    )
    
    if self.log_stats:
        self.prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=num_new_computed_tokens,
            preempted=request.num_preemptions > 0,
        )
    
    return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens
```

**Purpose:** Find cached prefix blocks for request

**Key Logic:**

1. **Cache Disabled Check:** Return empty if caching off or request opted out
2. **Max Hit Length:** Leave at least 1 token unmatched (must recompute last token for logits)
3. **Cache Search:** Find longest prefix matching request's block hashes
4. **Statistics:** Record hit metrics

**Example Flow:**
```
Request: 100 tokens (hashes for 7 blocks of 16 tokens each)
→ Search for match up to 99 tokens (max 6 blocks)
→ Find 4 blocks cached
→ Return: 4 cached blocks + 64 tokens matched
→ Scheduler allocates blocks for remaining 36 tokens
```

### 4. Allocate Slots (`allocate_slots`)

```python
def allocate_slots(
    self,
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: KVCacheBlocks | None = None,
    num_lookahead_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
) -> KVCacheBlocks | None
```

**Purpose:** Allocate KV cache blocks for tokens to be computed

**Parameters:**

| Parameter | Meaning |
|-----------|---------|
| `num_new_tokens` | New tokens to allocate (including unverified draft) |
| `num_new_computed_tokens` | New tokens from prefix cache hit |
| `new_computed_blocks` | Blocks for prefix cache hits |
| `num_lookahead_tokens` | Eagle speculative tokens |
| `num_external_computed_tokens` | Tokens cached by KV connector |
| `delay_cache_blocks` | Skip caching in this call (for P/D) |
| `num_encoder_tokens` | For encoder-decoder models (Whisper) |

**Returns:** 
- `KVCacheBlocks` if allocation succeeds
- `None` if insufficient free blocks

**Complex Logic:**

The method handles a sophisticated block layout:

```
│  cached  │ new_computed │ external │  new  │ lookahead │
│ from vLLM│ from vLLM    │ from KV  │ to    │ speculative
│          │ (prefix hit) │ connector│compute│ (eagle)
```

**Process:**

1. **Validation:** Ensure num_new_tokens > 0 or external tokens > 0
2. **Check Free Blocks:** Calculate required blocks
3. **Free Old Blocks:** Release blocks outside sliding window
4. **Allocate New:** Get blocks from pool

### 5. Cache Blocks (`cache_blocks_for_requests`)

```python
def cache_blocks_for_requests(
    self,
    requests: list[Request],
) -> None
```

**Purpose:** Cache full blocks from requests

**Delegation:**
```python
for request in requests:
    self.coordinator.cache_blocks(request, request.num_computed_tokens)
```

**Process:**
1. Iterate completed requests
2. Update block hash metadata
3. Insert into prefix cache

### 6. Free Request Blocks (`free_blocks_for_request`)

```python
def free_blocks_for_request(self, request_id: str) -> None:
    self.coordinator.free(request_id)
```

**Purpose:** Free all blocks for completed request

**Delegation:** Calls coordinator, which delegates to all managers

### 7. Create KVCacheBlocks Wrapper

```python
def create_kv_cache_blocks(
    self,
    blocks: tuple[Sequence[KVCacheBlock], ...]
) -> KVCacheBlocks:
    return KVCacheBlocks(blocks)
```

**Purpose:** Wrap raw block tuples in API type

## Event Management

### KV Cache Events

```python
def take_events(self) -> list[KVCacheEvent]:
    return self.block_pool.take_events()
```

**Purpose:** Retrieve published KV cache events

**Used By:**
- Distributed KV cache systems
- Event logging
- System monitoring

## Statistics Integration

### Prefix Cache Stats

When `log_stats=True`:

```python
self.prefix_cache_stats: PrefixCacheStats | None = (
    PrefixCacheStats() if log_stats else None
)
```

**Tracks:**
- Per-request cache performance
- Preemption impact on cache hits
- Overall cache effectiveness

**Retrieval:**
```python
stats = manager.make_prefix_cache_stats()
print(f"Cache hit rate: {stats.hit_rate:.2%}")
```

## Data Flow Example: New Request

```
1. Scheduler receives request
   ↓
2. manager.get_computed_blocks(request)
   - Search cache for matching prefix
   - Return existing blocks + num_tokens_matched
   ↓
3. Scheduler allocates remaining blocks
   manager.allocate_slots(
       request_id, 
       num_new_tokens,
       new_computed_blocks,
       ...
   )
   - Check free blocks
   - Allocate from pool
   - Return new blocks
   ↓
4. Worker computes tokens and fills blocks
   ↓
5. Scheduler registers prefix cache hit
   manager.allocate_new_computed_blocks(
       request_id,
       new_computed_blocks,
       ...
   )
   - Touch blocks (increment ref_cnt)
   - Track for caching
   ↓
6. Scheduler caches full blocks (optional)
   manager.cache_blocks_for_requests([request])
   - Update block hashes
   - Insert into cache
   ↓
7. Request completes
   manager.free_blocks_for_request(request_id)
   - Decrement ref_cnt
   - Return to free pool
```

## Error Handling

### Return None for Insufficient Blocks

```python
def allocate_slots(...) -> KVCacheBlocks | None:
    # Returns None if insufficient free blocks
    # Scheduler should preempt lower-priority requests
```

**Caller Responsibility:** Scheduler decides preemption policy

### Input Validation

```python
if num_new_tokens == 0 and num_external_computed_tokens == 0:
    raise ValueError(
        "num_new_tokens must be > 0 when there are no external tokens"
    )
```

## Integration with Scheduler

### Typical Call Sequence

```python
manager = KVCacheManager(...)

# For each scheduling iteration:
for request in scheduler.running_requests:
    # Step 1: Check for cache hits
    computed_blocks, num_hits = manager.get_computed_blocks(request)
    
    # Step 2: Allocate for new tokens
    new_blocks = manager.allocate_slots(
        request=request,
        num_new_tokens=request.num_new_tokens,
        new_computed_blocks=computed_blocks,
        num_lookahead_tokens=request.num_lookahead_tokens,
    )
    
    if new_blocks is None:
        # OOM: Needs preemption
        scheduler.preempt(request)
        continue
    
    # Step 3: Add to batch for compute
    scheduler.add_to_batch(request, new_blocks)

# After compute:
for request in scheduler.completed_requests:
    manager.free_blocks_for_request(request.id)

# Periodically cache blocks:
if scheduler.cache_blocks_threshold_hit:
    manager.cache_blocks_for_requests(scheduler.running_requests)

# Get events for distributed sync:
events = manager.take_events()
distribute_to_other_workers(events)
```

## Performance Considerations

### Empty Block Reuse

Pre-allocated `empty_kv_cache_blocks` avoids GC:
```python
# Instead of:
return KVCacheBlocks(tuple(() for _ in range(n)))  # Creates each time

# Uses:
return self.empty_kv_cache_blocks  # Reuse
```

### Minimal Copying

Operations use references when possible:
```python
# KVCacheBlocks.__add__ chains iterators
# Not creating new list of all blocks
```

### Lazy Statistics

Stats only collected if `log_stats=True`:
```python
self.prefix_cache_stats = (
    PrefixCacheStats() if log_stats else None
)
```

## Key Design Principles

1. **High-Level Abstraction:** Hides complex multi-group coordination
2. **Immutable Results:** Returns are wrapped in immutable types
3. **Statistics Optional:** No overhead when not needed
4. **Event Publishing:** Supports distributed KV cache systems
5. **Graceful Degradation:** Returns None for insufficient resources
6. **Reusable Objects:** Minimizes allocation in critical paths

## Extension Points

To support new features:
1. Add parameter to `allocate_slots()` if needed
2. Update `KVCacheBlocks` if new result format needed
3. Coordinator handles new attention types automatically
4. Statistics can be extended via `PrefixCacheStats`
