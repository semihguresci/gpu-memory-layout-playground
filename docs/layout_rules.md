# Buffer Layout Rules (Vulkan Compute)

This document defines the layout rules used in this repository for host/device buffer compatibility.

## 1. General Rules
- Use `std430` for storage buffers (SSBO) unless an experiment explicitly studies other layouts.
- Keep host-side structs and shader-side structs byte-compatible.
- Prefer explicit, fixed-width scalar types (`float`, `std::uint32_t`, etc.).
- Avoid implicit padding assumptions; validate with compile-time assertions.

## 2. Host Type Requirements
- Host buffer structs must be:
  - standard-layout
  - trivially copyable
- Validate the following at compile time:
  - `sizeof(T)`
  - `alignof(T)`
  - `offsetof(T, member)` for each field used by shader

Use helpers from:
- `include/utils/layout_assert.hpp`

## 3. `std430` Practical Notes
- Scalars (`float`, `int`, `uint`) align to 4 bytes.
- `vec2` aligns to 8 bytes.
- `vec3`/`vec4` align to 16 bytes.
- Arrays use element base alignment as array stride alignment.
- Struct alignment is the max alignment of its members.
- Struct size must include trailing padding to satisfy struct alignment.

For this project, avoid `vec3` in storage layouts unless explicitly studying padding effects.

## 4. Transfer and Staging Rules
- Upload staging buffer: `VK_BUFFER_USAGE_TRANSFER_SRC_BIT`, host-visible memory.
- Device compute buffer: `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT`, device-local memory.
- Readback staging buffer: `VK_BUFFER_USAGE_TRANSFER_DST_BIT`, host-visible memory.

Use explicit synchronization barriers between:
- transfer write -> compute read/write
- compute write -> transfer read

## 5. Descriptor and Offset Alignment
- Dynamic offsets and range offsets must respect `minStorageBufferOffsetAlignment`.
- If non-coherent memory is used, flush/invalidate ranges must respect `nonCoherentAtomSize`.
- In this codebase, host-coherent staging buffers are currently used to simplify synchronization.

## 6. Checklist Before Adding a New Buffer Struct
- Define shader struct and host struct side by side (or clearly linked).
- Add static layout assertions for type traits, size, alignment, and offsets.
- Run with validation layers enabled at least once.
- Keep benchmark metadata and notes explicit if layout assumptions are experimental.
