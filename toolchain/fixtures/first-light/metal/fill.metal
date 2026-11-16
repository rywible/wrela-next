// language: metal4.0
#include <metal_stdlib>
#include <simd/simd.h>
#pragma METAL fp math_mode(relaxed)
#pragma METAL fp contract(fast)
#pragma clang diagnostic ignored "-Wtautological-compare"
using metal::uint;
struct _mslBufferSizes {
uint size0;
};
struct S3_Params {
uint m0_step;
uint m1_count;
metal::uint2 m2_pad;
};
typedef uint B0[1];
uint naga_div(uint lhs, uint rhs) {
return lhs / metal::select(rhs, 1u, rhs == 0u);
}

uint naga_mod(uint lhs, uint rhs) {
return lhs % metal::select(rhs, 1u, rhs == 0u);
}

[[max_total_threads_per_threadgroup(64)]] kernel void fill_e(
metal::uint3 input0 [[thread_position_in_grid]]
, constant S3_Params& r0_params [[buffer(0)]]
, device B0& r1_palette [[buffer(1)]]
, constant _mslBufferSizes& _buffer_sizes [[buffer(2)]]
) {
metal::uint3 v0 = metal::uint3 {input0};
uint v4 = r0_params.m1_count;
if ((v0.x >= 4u) && (v0.x < v4)) {
uint v8 = r1_palette[metal::min(unsigned(naga_mod(v0.x, 4u)), (_buffer_sizes.size0 - 0 - 4) / 4)];
uint v10 = r0_params.m0_step;
r1_palette[metal::min(unsigned(v0.x), (_buffer_sizes.size0 - 0 - 4) / 4)] = v8 + (naga_div(v0.x, 4u) * v10);
}
}

