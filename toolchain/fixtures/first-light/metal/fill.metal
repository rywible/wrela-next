// first-light's compute pipeline: derives palette entries 4.. from the four seed colours, adding
// `step` once per group of four (wrapping u32 arithmetic, so a host can check it exactly).
// Metal shading language as the toolchain's MSL writer (naga 30's, `wrela_wgsl::msl`) wrote it
// from the WGSL this fixture held until 2026-11-15, its bindings in `wrela_abi::metal`'s slots.
// language: metal4.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
uint size1;
};

struct Params {
uint step;
uint count;
metal::uint2 _pad;
};
typedef uint type_2[1];
uint naga_mod(uint lhs, uint rhs) {
return lhs % metal::select(rhs, 1u, rhs == 0u);
}

uint naga_div(uint lhs, uint rhs) {
return lhs / metal::select(rhs, 1u, rhs == 0u);
}


struct fillInput {
};
[[max_total_threads_per_threadgroup(64)]] kernel void fill(
metal::uint3 id [[thread_position_in_grid]]
, constant Params& params [[buffer(0)]]
, device type_2& palette [[buffer(1)]]
, constant _mslBufferSizes& _buffer_sizes [[buffer(2)]]
) {
bool local = {};
uint i = id.x;
if (i >= 4u) {
uint _e8 = params.count;
local = i < _e8;
} else {
local = false;
}
bool _e11 = local;
if (_e11) {
uint _e18 = palette[metal::min(unsigned(naga_mod(i, 4u)), (_buffer_sizes.size1 - 0 - 4) / 4)];
uint _e23 = params.step;
palette[metal::min(unsigned(i), (_buffer_sizes.size1 - 0 - 4) / 4)] = _e18 + (naga_div(i, 4u) * _e23);
return;
} else {
return;
}
}
