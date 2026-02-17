# KV Cache Utilities Implementation

## Overview

The [vllm/v1/core/kv_cache_utils.py](../vllm/v1/core/kv_cache_utils.py) module provides foundational utilities for KV cache management including block metadata, free block queue operations, and block hashing infrastructure. It's a critical component that handles the low-level mechanics of the KV cache system.

## Core Data Types

### BlockHash Type

```python
BlockHash = NewType("BlockHash", bytes)
```

- Represents the hash of a single KV-cache block
- Treated as distinct type from `bytes` to catch accidental misuse
- Computed using SHA-256 or xxHash of block tokens and metadata

### BlockHashWithGroupId Type

```python
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)
```

- Combines a `BlockHash` with its KV cache group ID
- Packed as: `block_hash + group_id.to_bytes(4, "big", signed=False)`
- Compact representation (no tuple overhead)
- Used as key in `cached_block_hash_to_block` dictionary

### ExternalBlockHash Type

```python
ExternalBlockHash: TypeAlias = bytes | int
```

- Union type for reproducible block hashing
- Supports both raw bytes (SHA-256) and 64-bit integers (xxHash)
- Maintains backward compatibility after defaulting to SHA-256

## Block Hash Functions

### `make_block_hash_with_group_id()`

```python
def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId
```

**Purpose:** Pack block hash and group ID into a single key for cache lookups

**Implementation:**
- Appends 4-byte big-endian group ID to block hash
- No tuple allocation (efficiency optimization)

### `get_block_hash()`

```python
def get_block_hash(key: BlockHashWithGroupId) -> BlockHash
```

**Purpose:** Extract the block hash from a composite key

**Implementation:** Returns all bytes except the last 4 bytes

### `get_group_id()`

```python
def get_group_id(key: BlockHashWithGroupId) -> int
```

**Purpose:** Extract the group ID from a composite key

**Implementation:** Parses last 4 bytes as big-endian integer

### `maybe_convert_block_hash()`

```python
def maybe_convert_block_hash(hash_bytes: BlockHash) -> ExternalBlockHash
```

**Purpose:** Convert block hash for reproducible KV events

**Behavior:**
- If `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=True`: converts to 64-bit integer
- Otherwise: returns bytes unchanged
- Masks to 64 bits: `int.from_bytes(...) & ((1 << 64) - 1)`

## NONE_HASH Initialization

### Global Hash Seed

```python
NONE_HASH: BlockHash  # Initialized via init_none_hash()
_CBOR_HASH_FUNCTIONS = frozenset({sha256_cbor, xxhash_cbor})
```

**Purpose:** Provides consistent hash seed for first block in any prefix

**Initialization Logic:**

```python
def init_none_hash(hash_fn: Callable[[Any], bytes]):
    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None and hash_fn in _CBOR_HASH_FUNCTIONS:
        logger.warning("PYTHONHASHSEED not set - non-reproducible hashes")
    
    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(32))  # Random 32 bytes
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed))  # Deterministic
```

**Importance:**
- Enables reproducible hash sequences across process restarts
- Alignment with Python's `hash()` function behavior
- Critical for distributed KV cache synchronization

## KVCacheBlock Class

### Data Structure

```python
@dataclass
class KVCacheBlock:
    block_id: int  # 0 to num_gpu_blocks - 1
    ref_cnt: int = 0  # Reference count
    _block_hash: BlockHashWithGroupId | None = None  # Cached hash
    
    # Doubly-linked list pointers (for FreeKVCacheBlockQueue)
    prev_free_block: KVCacheBlock | None = None
    next_free_block: KVCacheBlock | None = None
    
    is_null: bool = False  # Null block flag
```

### Block Hash Management

**Property:** `block_hash` (with setter)

```python
@property
def block_hash(self) -> BlockHashWithGroupId | None:
    return self._block_hash

@block_hash.setter
def block_hash(self, block_hash: BlockHashWithGroupId):
    assert self.block_hash is None, "Block already has a hash"
    self._block_hash = block_hash
```

**Design:** Assert on double-assignment prevents accidental hash mutations

### Hash Reset

```python
def reset_hash(self):
    """Reset the block hash when the block is evicted."""
    self._block_hash = None
```

### Reference Count States

| ref_cnt | State | In Free Queue? | Cached? |
|---------|-------|----------------|---------|
| 0 | Eviction candidate | Yes | Maybe |
| 1+ | In-use | No | Maybe |

### __repr__ Implementation

Avoids recursive block representation by using block IDs instead of object references:

```python
def __repr__(self) -> str:
    prev_block_id = self.prev_free_block.block_id if self.prev_free_block else None
    next_block_id = self.next_free_block.block_id if self.next_free_block else None
    return (
        f"KVCacheBlock(block_id={self.block_id}, "
        f"ref_cnt={self.ref_cnt}, "
        f"_block_hash={self._block_hash!r}, "
        f"prev_free_block={prev_block_id}, "
        f"next_free_block={next_block_id})"
    )
```

## FreeKVCacheBlockQueue Class

### Purpose

Manages free KV cache blocks using a doubly-linked list optimized for:
- O(1) arbitrary block removal (unlike Python deque)
- LRU eviction ordering
- Zero Python object allocation for linked list operations

### Architecture

```python
class FreeKVCacheBlockQueue:
    fake_free_list_head: KVCacheBlock  # Sentinel at start
    fake_free_list_tail: KVCacheBlock  # Sentinel at end
    num_free_blocks: int  # Count of actual free blocks
```

**Sentinel Nodes:**
- Reduce branching in `popleft()` and `append()` operations
- Never popped or removed (safe to assume all real blocks have valid neighbors)
- Connected during initialization

### Eviction Order

Blocks are ordered by eviction priority:

1. **Primary:** Least Recently Used (LRU) - accessed earliest goes first
2. **Secondary:** For same access time, block with more hash tokens (tail of chain) goes first

Order is maintained by reversing block order during `free_blocks()` in scheduler.

### Core Operations

#### `popleft()` - Allocate Single Block

```python
def popleft(self) -> KVCacheBlock
```

**Process:**
1. Gets block after fake_head (first block)
2. Updates pointers: `fake_head.next = block.next`
3. Clears block's pointers: `block.prev = None; block.next = None`
4. Decrements `num_free_blocks`

**Complexity:** O(1)

#### `popleft_n()` - Allocate Multiple Blocks

```python
def popleft_n(self, n: int) -> list[KVCacheBlock]
```

**Optimization:** Single pointer update instead of n operations

**Process:**
1. Skips pointers for first n-1 blocks
2. Does single update for nth block's successor
3. Clears pointers for all n blocks in one pass

**Complexity:** O(n) with single update, not O(n²)

#### `remove()` - Remove Arbitrary Block

```python
def remove(self, block: KVCacheBlock) -> None
```

**Key Feature:** Arbitrary removal in O(1) unlike Python deque

**Process:**
1. Links previous block to next block: `block.prev.next = block.next`
2. Links next block to previous: `block.next.prev = block.prev`
3. Clears block's pointers
4. Decrements `num_free_blocks`

**Used by:** `BlockPool.touch()` when prefix cache hits

#### `append()` - Return Single Block to Free Queue

```python
def append(self, block: KVCacheBlock) -> None
```

**Process:**
1. Gets current last block before fake_tail
2. Inserts new block after last: `last.next = block; block.prev = last`
3. Connects to fake_tail: `block.next = tail; tail.prev = block`
4. Increments `num_free_blocks`

#### `append_n()` - Return Multiple Blocks to Free Queue

```python
def append_n(self, blocks: list[KVCacheBlock]) -> None
```

**Optimization:** Single pass through blocks list

**Process:**
1. Connects consecutive blocks in input list
2. Links first block to last before fake_tail
3. Links last block to fake_tail
4. Increments `num_free_blocks` by len(blocks)

**Complexity:** O(n) for n blocks

#### `get_all_free_blocks()` - Testing Utility

```python
def get_all_free_blocks(self) -> list[KVCacheBlock]
```

Traverses linked list from fake_head to fake_tail

## Extra Key Generation

### `need_extra_keys()`

```python
def need_extra_keys(request: Request) -> bool
```

**Returns True if:**
- Request has multimodal features (`request.mm_features`)
- Request uses LoRA (`request.lora_request`)
- Request has cache salt (`request.cache_salt`)

**Purpose:** Determines if block hash needs additional components beyond tokens

### `_gen_mm_extra_hash_keys()`

```python
def _gen_mm_extra_hash_keys(
    request: Request,
    start_token_idx: int,
    end_token_idx: int,
    start_mm_idx: int
) -> tuple[list[Any], int]
```

**Purpose:** Generate hash keys for multimodal content

**Returns:**
- List of extra keys: `[(mm_hash, start_offset), ...]`
- Updated MM index after block

**Key Insight:** For multi-modal inputs, includes both identity (`mm_hash`) and position (`start_offset`) in the block

## Block Hash List Classes

### BlockHashList

```python
class BlockHashList:
    """Container of block hashes with index access."""
    def __getitem__(self, index: int) -> BlockHash
```

Simple wrapper for block hashes computed at `hash_block_size` granularity.

### BlockHashListWithBlockSize

```python
class BlockHashListWithBlockSize:
    def __init__(
        self,
        block_hashes: BlockHashList,
        hash_block_size: int,
        block_size: int
    )
```

**Purpose:** Recalulates block hashes at different granularity

**Scenario:** When `block_size > hash_block_size` (different KV groups have different block sizes)

**Operation:**
- Original hashes computed at 16-token granularity
- Needs hashes at 32-token granularity
- Combines multiple hash block results into single block hash

## Performance & Design Optimizations

### Zero Python Object Allocation for Linked List
- Directly manipulates block object pointers
- No separate Node objects
- More cache-friendly than separate node allocations

### Sentinel Nodes
- Eliminates boundary checks in append/remove
- All real blocks guaranteed to have valid prev/next
- Slight memory overhead for predictable performance

### Batch Operations
- `popleft_n()` and `append_n()` for multiple blocks
- Single list update instead of per-block operations
- Especially important for large block counts

### Immutable Composite Keys
- `BlockHashWithGroupId` packing avoids tuple allocation
- Dictionary key lookups are hash-based, not tuple-unpacking
- Significant performance benefit in tight loops

## Usage Examples

```python
# Create KVCacheBlock
block = KVCacheBlock(block_id=0)

# Initialize free queue
block_pool_blocks = [KVCacheBlock(i) for i in range(1000)]
free_queue = FreeKVCacheBlockQueue(block_pool_blocks)

# Allocate blocks
blocks = free_queue.popleft_n(10)

# Touch block (remove from free queue on cache hit)
free_queue.remove(blocks[0])
blocks[0].ref_cnt += 1

# Free blocks
free_queue.append_n(blocks)

# Composite key operations
from vllm.v1.core.kv_cache_utils import (
    make_block_hash_with_group_id,
    get_block_hash,
    get_group_id,
)

hash_key = make_block_hash_with_group_id(BlockHash(b"..."), group_id=0)
extracted_hash = get_block_hash(hash_key)
extracted_group_id = get_group_id(hash_key)
```

## Integration Points

- **BlockPool**: Uses `KVCacheBlock` and `FreeKVCacheBlockQueue` for block management
- **Request**: Computes `block_hashes` which are stored in `BlockHashList`
- **Prefix Cache**: Uses block hashes for cache lookups
- **Metrics System**: Tracks block lifecycle via `KVCacheMetricsCollector`
- **Hash Functions**: Uses SHA-256 or xxHash for deterministic hashing

## Key Design Principles

1. **Zero-Allocation Linked Lists**: Direct pointer manipulation in objects
2. **Composite Key Optimization**: Pack multiple values into single dictionary key
3. **Immutable Types**: Distinguish `BlockHash` from raw `bytes`
4. **Batch Operations**: Optimize multi-block operations
5. **Deterministic Hashing**: Support reproducible cache across restarts
