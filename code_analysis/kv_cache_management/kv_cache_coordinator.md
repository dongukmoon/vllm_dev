# KV Cache Coordinator Implementation

## Overview

The [vllm/v1/core/kv_cache_coordinator.py](../vllm/v1/core/kv_cache_coordinator.py) module defines the abstract `KVCacheCoordinator` class that orchestrates KV cache management across multiple attention types and KV cache groups. It serves as the central coordinator between the scheduler and individual single-type managers.

## Architecture

### Class Hierarchy

```
KVCacheCoordinator (ABC - abstract)
├── Concrete implementations (various attention configurations)
└── Components:
    ├── BlockPool (shared)
    ├── SingleTypeKVCacheManager[] (one per KV cache group)
    └── Optional: CrossAttentionManager
```

## Core Components

### BlockPool

```python
self.block_pool = BlockPool(
    kv_cache_config.num_blocks,
    enable_caching,
    hash_block_size,
    enable_kv_cache_events,
    metrics_collector,
)
```

**Shared across all managers:** All KV cache groups allocate from same pool

### Single Type Managers

```python
self.single_type_managers: tuple[SingleTypeKVCacheManager, ...] = tuple(
    get_manager_for_kv_cache_spec(
        kv_cache_spec=kv_cache_group.kv_cache_spec,
        block_pool=self.block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=i,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
    )
    for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
)
```

**One manager per group:**
- Handles specific attention type (full, sliding window, etc.)
- Manages block allocation/freeing for that group
- Implements type-specific logic

## Initialization

### Constructor Parameters

```python
def __init__(
    self,
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    use_eagle: bool,  # Speculative decoding
    enable_caching: bool,  # Prefix caching
    enable_kv_cache_events: bool,  # Distributed KV cache
    dcp_world_size: int,  # Data center parallel world size
    pcp_world_size: int,  # Pipeline parallel world size (KV cache)
    hash_block_size: int,  # Block size for hashing
    metrics_collector: KVCacheMetricsCollector | None = None,
)
```

### Key Configuration Options

| Parameter | Purpose |
|-----------|---------|
| `use_eagle` | Enable special handling for eagle-based speculative decoding |
| `enable_caching` | Enable prefix caching optimization |
| `enable_kv_cache_events` | Publish events for distributed KV cache systems |
| `dcp_world_size` × `pcp_world_size` | Expand block size for tensor parallelism |
| `hash_block_size` | Base granularity for block hashing |
| `metrics_collector` | Optional metrics tracking |

## Block Allocation Operations

### 1. Calculate Blocks to Allocate (`get_num_blocks_to_allocate`)

```python
def get_num_blocks_to_allocate(
    self,
    request_id: str,
    num_tokens: int,
    new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
    num_encoder_tokens: int,
    total_computed_tokens: int,
    num_tokens_main_model: int,
) -> int
```

**Purpose:** Determine how many new blocks must be allocated

**Process:**

```python
num_blocks_to_allocate = 0
for i, manager in enumerate(self.single_type_managers):
    if isinstance(manager, CrossAttentionManager):
        # Static allocation for encoder tokens
        num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
            request_id, num_encoder_tokens, [], 0, num_encoder_tokens
        )
    else:
        # Dynamic allocation for decoder tokens
        num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks[i],
            total_computed_tokens,
            num_tokens_main_model,
        )
return num_blocks_to_allocate
```

**Special Handling:**
- **Cross-Attention (Encoder):** Single static allocation based on encoder length
- **Self-Attention (Decoder):** Dynamic allocation as new tokens are generated

### 2. Cache Full Blocks (`cache_blocks`)

```python
def cache_blocks(self, request: Request, num_computed_tokens: int) -> None
```

**Purpose:** Update block hash metadata and cache full blocks

**Delegation:**
```python
for manager in self.single_type_managers:
    manager.cache_blocks(request, num_computed_tokens)
```

**Process:**
1. Each manager caches its group's full blocks
2. Blocks are indexed by `request.block_hashes`
3. Only full blocks are cached (optimization)

### 3. Allocate New Blocks (`allocate_new_blocks`)

```python
def allocate_new_blocks(
    self,
    request_id: str,
    num_tokens: int,
    num_tokens_main_model: int,
    num_encoder_tokens: int = 0,
) -> tuple[list[KVCacheBlock], ...]
```

**Purpose:** Allocate blocks for both encoder and decoder

**Process:**

```python
return tuple(
    manager.allocate_new_blocks(
        request_id,
        num_encoder_tokens if isinstance(manager, CrossAttentionManager)
        else num_tokens,
        num_tokens_main_model,
    )
    for manager in self.single_type_managers
)
```

**Returns:** Tuple of block lists, one per KV cache group

## Block Management Operations

### 4. Allocate New Computed Blocks (`allocate_new_computed_blocks`)

```python
def allocate_new_computed_blocks(
    self,
    request_id: str,
    new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
    num_local_computed_tokens: int,
    num_external_computed_tokens: int,
) -> None
```

**Purpose:** Register prefix cache hits and allocate for external tokens

**Process:**
```python
for i, manager in enumerate(self.single_type_managers):
    manager.allocate_new_computed_blocks(
        request_id,
        new_computed_blocks[i],
        num_local_computed_tokens,
        num_external_computed_tokens,
    )
```

### 5. Free Blocks (`free`)

```python
def free(self, request_id: str) -> None
```

**Purpose:** Free all blocks for completed request

**Process:**
```python
for manager in self.single_type_managers:
    manager.free(request_id)
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

**Scenario:** Sliding window attention can skip old blocks

**Process:**
```python
for manager in self.single_type_managers:
    manager.remove_skipped_blocks(request_id, total_computed_tokens)
```

## Prefix Cache Operations

### 7. Find Longest Cache Hit (`find_longest_cache_hit`)

```python
@abstractmethod
def find_longest_cache_hit(
    self,
    block_hashes: list[BlockHash],
    max_cache_hit_length: int,
) -> tuple[tuple[list[KVCacheBlock], ...], int]
```

**Purpose:** Find longest prefix that matches cached blocks

**Returns:**
- Blocks matching the prefix (one list per group)
- Number of matched tokens

**Abstract Method:** Concrete implementations handle:
- Regular prefix matching
- Eagle speculative decoding optimization

### 8. Get Common Prefix Blocks (`get_num_common_prefix_blocks`)

```python
def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]
```

**Purpose:** Find common prefix for all running requests (for prefix caching)

**Returns:** Number of common prefix blocks per group

**Delegation:**
```python
return [
    manager.get_num_common_prefix_blocks(running_request_id)
    for manager in self.single_type_managers
]
```

## Block Retrieval

### 9. Get Blocks (`get_blocks`)

```python
def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]
```

**Purpose:** Retrieve allocated blocks for a request

**Returns:** Tuple of block lists, one per group

**Delegation:**
```python
return tuple(
    manager.req_to_blocks.get(request_id) or []
    for manager in self.single_type_managers
)
```

## Encoder-Decoder Support

### CrossAttentionManager Handling

For models like Whisper (encoder-decoder):

1. **Separate Manager:** CrossAttentionManager tracks encoder blocks separately
2. **Static Allocation:** Allocated once based on encoder length
3. **Identified by:** `isinstance(manager, CrossAttentionManager)`

**Example Flow:**
```python
# For an encoder-decoder model:
# - 1st manager: CrossAttentionManager (encoder self-attention)
# - 2nd manager: Self-attention for decoder
# - (Optional) 3rd manager: Cross-attention (decoder attends to encoder)

# During allocation:
for i, manager in enumerate(self.single_type_managers):
    if isinstance(manager, CrossAttentionManager):
        # Use encoder token count
        num_blocks = cdiv(num_encoder_tokens, manager.block_size)
    else:
        # Use decoder token count
        num_blocks = cdiv(num_decoder_tokens, manager.block_size)
```

## Configuration Parameters

### Tensor Parallelism Adjustment

```python
if dcp_world_size * pcp_world_size > 1:
    self.block_size *= dcp_world_size * pcp_world_size
```

**Rationale:** When multiple workers share KV cache:
- Block must accommodate all workers' data
- Typically: 1 block = `original_size × tp_world_size`

### Eagle Speculative Decoding

```python
self.use_eagle = use_eagle
```

**Special Handling:** Concrete implementations of `find_longest_cache_hit`:
- Regular: Find longest matching prefix
- With Eagle: May need to handle speculative tokens differently

## Data Flow Diagram

```
Scheduler
    │
    ├─→ get_num_blocks_to_allocate()
    │   ├─→ For each manager: calculate blocks needed
    │   └─→ Return total blocks to allocate
    │
    ├─→ allocate_new_blocks()
    │   ├─→ For each manager: allocate_new_blocks()
    │   └─→ Return tuple(blocks_group1, blocks_group2, ...)
    │
    ├─→ allocate_new_computed_blocks()
    │   ├─→ For each manager: touch and track prefix hits
    │   └─→ Allocate for external tokens if any
    │
    ├─→ cache_blocks()
    │   ├─→ For each manager: cache_blocks()
    │   └─→ Update block hashes in cache
    │
    └─→ free()
        ├─→ For each manager: free()
        └─→ Return blocks to free pool
```

## Abstract Method Pattern

### Why Abstract?

Different attention patterns require different cache hit logic:

1. **FullAttention:** Sequential prefix matching
2. **SlidingWindow:** Prefix matching with attention window constraint
3. **MLAAttention:** Multi-level attention handling
4. **Mamba:** State-based model with different requirements

### Concrete Implementation

Concrete coordinator classes (e.g., `FullAttentionCoordinator`):

```python
class FullAttentionCoordinator(KVCacheCoordinator):
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        # Implementation specific to full attention
        # Finds longest matching prefix in cache
        pass
```

## Integration with Scheduler

### Typical Request Flow

1. **Request arrives** → Scheduler calls `get_num_blocks_to_allocate()`
2. **Check free blocks** → If insufficient, wait/preempt
3. **Allocate blocks** → Scheduler calls `allocate_new_blocks()`
4. **Compute tokens** → Worker fills allocated blocks
5. **Check cache hits** → Scheduler calls `find_longest_cache_hit()`
6. **Cache blocks** → Scheduler calls `allocate_new_computed_blocks()` and `cache_blocks()`
7. **More tokens** → Repeat from step 3 as needed
8. **Request completes** → Scheduler calls `free()`

## Key Design Principles

1. **Multi-Manager Support:** Handles multiple attention types
2. **Group Abstraction:** Hides per-group logic behind single interface
3. **Abstract Coordination:** Concrete implementations handle layout-specific logic
4. **Balanced Division:** `get_num_blocks_to_allocate` checks all groups
5. **Delegation Pattern:** Coordinator delegates to managers, doesn't duplicate logic

## Usage Example

```python
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

coordinator = get_kv_cache_coordinator(
    kv_cache_config=config,
    max_model_len=2048,
    use_eagle=False,
    enable_caching=True,
    enable_kv_cache_events=False,
    dcp_world_size=1,
    pcp_world_size=1,
    hash_block_size=16,
)

# Allocate blocks for new request
num_new_blocks = coordinator.get_num_blocks_to_allocate(
    request_id="req_1",
    num_tokens=100,
    new_computed_blocks=([], []),
    num_encoder_tokens=0,
    total_computed_tokens=0,
    num_tokens_main_model=100,
)

# Check for cache hits
blocks, num_hits = coordinator.find_longest_cache_hit(
    block_hashes=request.block_hashes,
    max_cache_hit_length=99,
)

# Allocate blocks
new_blocks = coordinator.allocate_new_blocks(
    request_id="req_1",
    num_tokens=100,
    num_tokens_main_model=100,
    num_encoder_tokens=0,
)

# Cache full blocks
coordinator.cache_blocks(request, num_computed_tokens=80)

# Free blocks on completion
coordinator.free(request_id="req_1")
```

## Performance Considerations

### Tuple Unpacking Overhead

The coordinator returns tuples for multi-group results. This is intentional:
- Ensures immutability
- Clear API boundary
- Minimal overhead (tuple creation is O(n) where n = num_groups, typically 1-3)

### Manager Query

Finding `CrossAttentionManager` uses `isinstance()`:
- O(1) per manager
- Total O(num_managers) ≈ O(constant)
- Acceptable for typically 1-3 managers

## Extension Points

To add new attention types:
1. Create new `SingleTypeKVCacheManager` subclass
2. Update `get_manager_for_kv_cache_spec()` factory
3. Optionally update `find_longest_cache_hit` if needs special logic
4. Coordinator handles new manager automatically
