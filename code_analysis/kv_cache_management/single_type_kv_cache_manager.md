# Single Type KV Cache Manager Implementation

## Overview

The [vllm/v1/core/single_type_kv_cache_manager.py](../vllm/v1/core/single_type_kv_cache_manager.py) module provides the abstract `SingleTypeKVCacheManager` class and concrete implementations for managing KV cache blocks for specific attention types. Each manager handles a single KV cache group with a single attention mechanism.

## Architecture

### Class Hierarchy

```
SingleTypeKVCacheManager (ABC)
├── FullAttentionManager
├── SlidingWindowAttentionManager
├── MLAAttentionManager
├── Mamba/StateManager
├── SinkFullAttentionManager
└── CrossAttentionManager (special)
```

Each concrete class customizes behavior for different attention patterns.

## Core Concepts

### KV Cache Group

A logical grouping of blocks:
- Serves one attention mechanism
- Has fixed `block_size`
- Manages blocks independently
- Multiple groups can coexist (e.g., encoder + decoder)

### Block Size

Total tokens per block:
- For single worker: `spec.block_size`
- For distributed: `spec.block_size * dcp_world_size * pcp_world_size`

```python
self.block_size = kv_cache_spec.block_size
if dcp_world_size * pcp_world_size > 1:
    self.block_size *= dcp_world_size * pcp_world_size
```

### Request Block Tracking

```python
self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
```

Maps request ID → allocated blocks for that request

## Initialization

```python
def __init__(
    self,
    kv_cache_spec: KVCacheSpec,
    block_pool: BlockPool,
    enable_caching: bool,
    kv_cache_group_id: int,
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
)
```

### Parameters

| Parameter | Purpose |
|-----------|---------|
| `kv_cache_spec` | Attention config (type, block size, window) |
| `block_pool` | Shared pool of all KV cache blocks |
| `enable_caching` | Enable prefix caching |
| `kv_cache_group_id` | Index in coordinator's manager list |
| `dcp_world_size` | Data parallel world size (tensor parallelism) |
| `pcp_world_size` | Pipeline cache parallel size |

### State

```python
self.block_size: int  # Adjusted for parallelism
self.kv_cache_spec: KVCacheSpec  # Original spec
self.block_pool: BlockPool  # Shared
self.enable_caching: bool
self.kv_cache_group_id: int

self.req_to_blocks: dict[str, list[KVCacheBlock]]
self.num_cached_block: dict[str, int]  # num_cached_blocks per request
self._null_block: KVCacheBlock  # Reference to pool's null block
```

## Core Operations

### 1. Calculate Blocks to Allocate (`get_num_blocks_to_allocate`)

```python
def get_num_blocks_to_allocate(
    self,
    request_id: str,
    num_tokens: int,
    new_computed_blocks: Sequence[KVCacheBlock],
    total_computed_tokens: int,
    num_tokens_main_model: int,
) -> int
```

**Purpose:** Determine how many NEW blocks must be allocated

**Complex Logic:**

```python
num_required_blocks = cdiv(num_tokens, self.block_size)
num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

if request_id in self.num_cached_block:
    # Fast path: running request, no new prefix hits
    assert len(new_computed_blocks) == 0
    return max(num_required_blocks - num_req_blocks, 0)

# Slow path: new request or preempted, compute skipped blocks
num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
num_skipped_blocks = num_skipped_tokens // self.block_size

num_new_blocks = max(
    num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
    0,
)
```

**Key Insight:**

```
Total Tokens: [  skipped  ][  computed  ][  new  ]
              <-- attention window doesn't cover

Blocks Needed:
- If local_computed >= skipped: Only need blocks for "new" part
- Otherwise: Need blocks for both skipped blocks and "new" part
```

**The Formula:**
```
num_required = ceil(num_tokens / block_size)
num_existing = num_computed + num_req_blocks + num_skipped

num_new = max(num_required - max(num_skipped, num_computed), 0)
```

**Skipped New Computed Blocks:**

```python
num_skipped_new_computed_blocks = max(
    0, num_skipped_blocks - num_req_blocks
)

# Among new_computed_blocks, first num_skipped_new_computed_blocks
# worth are discarded (outside attention window)
num_evictable_blocks = self._get_num_evictable_blocks(
    new_computed_blocks[num_skipped_new_computed_blocks:]
)
```

**Evictable Blocks:**
- Computed blocks that are eviction candidates
- Must be counted as free capacity needed
- When touched, removed from free queue

### 2. Allocate New Computed Blocks (`allocate_new_computed_blocks`)

```python
def allocate_new_computed_blocks(
    self,
    request_id: str,
    new_computed_blocks: Sequence[KVCacheBlock],
    num_local_computed_tokens: int,
    num_external_computed_tokens: int,
) -> None
```

**Purpose:** Register prefix cache hits and allocate for external tokens

**Process:**

```
1. Touch computed blocks (remove from free queue)
2. Skip blocks outside attention window (replace with null blocks)
3. Add remaining computed blocks to request
4. Allocate new blocks for external KV connector tokens
```

**Step-by-step:**

```python
# Step 1: Get skipped block count
num_skipped_tokens = self.get_num_skipped_tokens(
    num_local_computed_tokens + num_external_computed_tokens
)
num_skipped_blocks = num_skipped_tokens // self.block_size

# Step 2: Touch non-skipped blocks
blocks_to_touch = new_computed_blocks[num_skipped_blocks:]
self.block_pool.touch(blocks_to_touch)

# Step 3: Add to request (with null block padding)
num_skipped_new_computed_blocks = max(
    0, num_skipped_blocks - len(self.req_to_blocks.get(request_id, ()))
)
null_padding = [self._null_block] * num_skipped_new_computed_blocks
self.req_to_blocks[request_id].extend(null_padding + list(blocks_to_touch))

# Step 4: Mark as having computed blocks
self.num_cached_block[request_id] = len(blocks_to_touch)

# Step 5: Allocate for external tokens
if num_external_computed_tokens > 0:
    # Allocate blocks for KV connector tokens
    num_external_blocks = cdiv(
        num_external_computed_tokens, self.block_size
    )
    external_blocks = self.block_pool.get_new_blocks(num_external_blocks)
    self.req_to_blocks[request_id].extend(external_blocks)
```

### 3. Allocate New Blocks (`allocate_new_blocks`)

```python
def allocate_new_blocks(
    self,
    request_id: str,
    num_tokens: int,
    num_tokens_main_model: int,
) -> list[KVCacheBlock]
```

**Purpose:** Allocate blocks for new tokens to be computed

**Returns:** List of new blocks

**Process:**

```python
num_blocks_to_allocate = self.get_num_blocks_to_allocate(
    request_id,
    num_tokens,
    [],  # No new computed blocks here
    num_tokens,
    num_tokens_main_model,
)

if num_blocks_to_allocate > 0:
    new_blocks = self.block_pool.get_new_blocks(num_blocks_to_allocate)
    self.req_to_blocks[request_id].extend(new_blocks)
    return new_blocks

return []
```

### 4. Cache Blocks (`cache_blocks`)

```python
def cache_blocks(self, request: Request, num_computed_tokens: int) -> None
```

**Purpose:** Cache full blocks using request's block hashes

**Process:**

```python
num_full_blocks = num_computed_tokens // self.block_size
num_cached_blocks = self.num_cached_block.get(request_id, 0)

if num_cached_blocks >= num_full_blocks:
    # All eligible blocks already cached
    return

# Get blocks to cache
blocks = self.req_to_blocks[request_id]

# Delegate to BlockPool
self.block_pool.cache_full_blocks(
    request=request,
    blocks=blocks,
    num_cached_blocks=num_cached_blocks,
    num_full_blocks=num_full_blocks,
    block_size=self.block_size,
    kv_cache_group_id=self.kv_cache_group_id,
)

# Update cache count
self.num_cached_block[request_id] = num_full_blocks
```

### 5. Free Blocks (`free`)

```python
def free(self, request_id: str) -> None
```

**Purpose:** Free all blocks for completed request

**Process:**

```python
blocks = self.req_to_blocks.pop(request_id, [])
self.block_pool.free_blocks(blocks)
self.num_cached_block.pop(request_id, None)
```

### 6. Remove Skipped Blocks (`remove_skipped_blocks`)

```python
def remove_skipped_blocks(
    self,
    request_id: str,
    total_computed_tokens: int,
) -> None
```

**Purpose:** Replace blocks outside sliding window with null blocks

**Scenario:** Sliding window attention with fixed window size

**Process:**

```python
num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
num_skipped_blocks = num_skipped_tokens // self.block_size

blocks = self.req_to_blocks[request_id]

# Replace first num_skipped_blocks with null blocks
for i in range(num_skipped_blocks):
    self.block_pool.free_blocks([blocks[i]])
    blocks[i] = self._null_block
```

## Attention Type Customization

### `get_num_skipped_tokens()`

**Abstract Method:** Implemented per attention type

```python
@abstractmethod
def get_num_skipped_tokens(self, total_computed_tokens: int) -> int
```

#### FullAttention Implementation

```python
class FullAttentionManager(SingleTypeKVCacheManager):
    def get_num_skipped_tokens(self, total_computed_tokens: int) -> int:
        return 0  # Never skip tokens
```

#### SlidingWindow Implementation

```python
class SlidingWindowAttentionManager(SingleTypeKVCacheManager):
    def __init__(self, ..., kv_cache_spec: SlidingWindowSpec, ...):
        self.window_size = kv_cache_spec.window_size
    
    def get_num_skipped_tokens(self, total_computed_tokens: int) -> int:
        # Tokens before the sliding window
        skipped = total_computed_tokens - self.window_size
        return max(skipped, 0)
```

#### Mamba Implementation

```python
class MambaManager(SingleTypeKVCacheManager):
    def get_num_skipped_tokens(self, total_computed_tokens: int) -> int:
        # Mamba is stateful; only need recent state
        # No blocks are skipped; state is memory-efficient
        return 0
```

## CrossAttentionManager Special Case

For encoder-decoder models (Whisper):

```python
class CrossAttentionManager(SingleTypeKVCacheManager):
    """Manages KV cache for cross-attention (encoder output)."""
    
    def get_num_skipped_tokens(self, total_computed_tokens: int) -> int:
        return 0  # Never skip encoder tokens
    
    def allocate_new_blocks(
        self,
        request_id: str,
        num_encoder_tokens: int,
        num_tokens_main_model: int,
    ) -> list[KVCacheBlock]:
        # Static allocation: allocate once per request
        if request_id in self.req_to_blocks:
            return []  # Already allocated
        
        num_blocks = cdiv(num_encoder_tokens, self.block_size)
        blocks = self.block_pool.get_new_blocks(num_blocks)
        self.req_to_blocks[request_id] = blocks
        return blocks
```

## Block Eviction Tracking

### Evictable Blocks Counting

```python
@classmethod
def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
    return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)
```

**Purpose:** Count blocks eligible for eviction in new_computed_blocks

**Why:** These blocks will be removed from free queue when touched

## Common Prefix Discovery

### `get_num_common_prefix_blocks()`

```python
@abstractmethod
def get_num_common_prefix_blocks(
    self, running_request_id: str
) -> int
```

**Purpose:** Find common prefix blocks across all requests

**Used for:** Raypipe and other prefix-sharing optimizations

**Typical Implementation:**

```python
def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
    if running_request_id not in self.req_to_blocks:
        return 0
    
    running_blocks = self.req_to_blocks[running_request_id]
    
    # Find longest suffix of running_blocks
    # that matches prefix of other requests' blocks
    common = 0
    for block in running_blocks:
        if block.block_id == self._null_block.block_id:
            break
        if block.block_hash is not None:
            common += 1
        else:
            break
    
    return common
```

## Advanced Concepts

### Speculative Decoding (Eagle)

When `num_lookahead_tokens > 0`:

```python
# allocate_slots receives num_lookahead_tokens parameter
# These are blocks for unverified draft tokens

# If verification fails:
# - Remove lookahead blocks
# - Keep verified blocks

# If verification succeeds:
# - Accept lookahead blocks
# - Mark for caching
```

### External KV Cache (Connectors)

```python
# When num_external_computed_tokens > 0:
num_external_blocks = cdiv(
    num_external_computed_tokens, self.block_size
)
external_blocks = self.block_pool.get_new_blocks(num_external_blocks)
self.req_to_blocks[request_id].extend(external_blocks)
```

Maps external data into allocated blocks.

### Data Center Parallel (Tensor Parallelism)

```python
# Each worker has dcp_world_size data copies
# Block size expanded to accommodate all:
block_size = spec.block_size * dcp_world_size * pcp_world_size

# Block hashing still at original granularity:
hash_block_size = coordinator.hash_block_size  # Fixed at creation
```

## Data Flow: Request Through Manager

```
┌─────────────────────────────────────────┐
│ Request arrives with block_hashes       │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ Scheduler: get_computed_blocks()        │
│ → find_longest_cache_hit(block_hashes)  │
│ → manager.find_longest_cache_hit()      │
│ → Returns cached blocks OR empty        │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ Scheduler: allocate_slots()             │
│ → get_num_blocks_to_allocate()          │
│ → manager.get_new_blocks()              │
│ → req_to_blocks[request_id] += blocks   │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ Worker: Compute tokens, fill blocks     │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ Scheduler: allocate_new_computed_blocks()
│ → block_pool.touch(blocks)              │
│ → Update num_cached_block[request_id]   │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ Scheduler: cache_blocks()               │
│ → block_pool.cache_full_blocks()        │
│ → Update block_hash metadata            │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ More tokens? → Back to allocate_slots() │
│ Request done? → free()                  │
└─────────────────────────────────────────┘
```

## Performance Optimizations

### Fast Path for Running Requests

```python
if request_id in self.num_cached_block:
    # Fast path: skip complex skipped block calculation
    assert len(new_computed_blocks) == 0
    return max(num_required_blocks - num_req_blocks, 0)
```

### Batch Operations

```python
# append_n() for multiple blocks
block_pool.free_blocks(blocks)  # Handled as batch internally
```

### Null Block Reuse

```python
# Instead of allocating new blocks for skipped regions:
for i in range(num_skipped_blocks):
    blocks[i] = self._null_block  # Reuse singleton
```

## Error Handling

### Invalid Block IDs

Asserts catch eviction issues:
```python
@invariant
def blocks_are_valid(self):
    for block_list in self.req_to_blocks.values():
        assert all(0 <= b.block_id < num_blocks for b in block_list)
```

## Integration Pattern

```python
# Coordinator orchestrates multiple managers:
for manager in coordinator.single_type_managers:
    manager.allocate_new_blocks(...)  # Each handles their group
```

## Key Design Principles

1. **Per-Group Specialization:** Each manager handles one attention type
2. **Modular Skipping Logic:** Abstract method for attention-specific skip rules
3. **Reference Counting:** Tracks block usage across requests
4. **Lazy Caching:** Blocks cached only when full
5. **Null Block Reuse:** Efficient padding for skipped regions
6. **Fast Path:** Running requests skip complex calculations

## Extension Points

To add new attention type:
1. Subclass `SingleTypeKVCacheManager`
2. Implement `get_num_skipped_tokens()`
3. Implement `get_num_common_prefix_blocks()`
4. Register in `get_manager_for_kv_cache_spec()` factory
5. Coordinator uses automatically
