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
typedef uint B0[1];
struct vs_eOutput {
metal::float4 v0 [[position]];
};
struct S9_VertexIndex {
uint m0_index;
};
struct S8_ClipPosition {
metal::float4 m0_position;
};
struct fs_eOutput {
metal::float4 color0 [[color(0)]];
};
struct S6_Frame {
metal::float2 m0_resolution;
metal::float2 m1_center;
float m2_time;
float m3_radius;
metal::float2 m4_pad;
};
struct S10_FragCoord {
metal::float4 m0_position;
};
metal::float3 f1_colour(
uint p0_p0
, device B0 const& r1_palette
, constant _mslBufferSizes& _buffer_sizes
) {
uint v0 = p0_p0;
uint v1 = r1_palette[metal::min(unsigned(v0), (_buffer_sizes.size0 - 0 - 4) / 4)];
return metal::float4(static_cast<float>((v1 >> 0u) & 255u) / 255.0, static_cast<float>((v1 >> 8u) & 255u) / 255.0, static_cast<float>((v1 >> 16u) & 255u) / 255.0, static_cast<float>((v1 >> 24u) & 255u) / 255.0).xyz;
}

vertex vs_eOutput vs_e(
uint input0 [[vertex_id]]
) {
S9_VertexIndex v0 = S9_VertexIndex {input0};
const S8_ClipPosition _tmp1 = S8_ClipPosition {metal::float4(static_cast<float>(as_type<int>(as_type<uint>(as_type<int>(as_type<uint>(static_cast<int>(v0.m0_index & 1u)) * as_type<uint>(4))) - as_type<uint>(1))), static_cast<float>(as_type<int>(as_type<uint>(as_type<int>(as_type<uint>(static_cast<int>(v0.m0_index >> 1u)) * as_type<uint>(4))) - as_type<uint>(1))), 0.0, 1.0)};
return vs_eOutput { _tmp1.m0_position };
}

fragment fs_eOutput fs_e(
metal::float4 input0 [[position]]
, constant S6_Frame& r0_frame [[buffer(0)]]
, device B0 const& r1_palette [[buffer(1)]]
, constant _mslBufferSizes& _buffer_sizes [[buffer(2)]]
) {
S10_FragCoord v0 = S10_FragCoord {input0};
metal::float2 v2 = v0.m0_position.xy;
metal::float3 v4 = f1_colour(0u, r1_palette, _buffer_sizes);
metal::float3 v6 = f1_colour(2u, r1_palette, _buffer_sizes);
metal::float2 v7 = r0_frame.m0_resolution;
metal::float2 v13 = r0_frame.m1_center;
float v14 = r0_frame.m3_radius;
float v16 = metal::distance(v2, v13) - v14;
metal::float3 v18 = f1_colour(7u, r1_palette, _buffer_sizes);
float v32 = r0_frame.m2_time;
metal::float3 v41 = f1_colour(5u, r1_palette, _buffer_sizes);
metal::float3 v50 = metal::min(metal::mix(v41 * metal::float3(0.85 + (0.15 * metal::cos(v32 * 3.0))), metal::mix(v4, v6, metal::float3(v2.y / v7.y)) + ((v18 * metal::float3(0.4)) * metal::float3(metal::exp((-(metal::max(v16, 0.0))) / (0.3 * v14)))), metal::float3(metal::smoothstep(-1.0, 1.0, v16))), metal::float3(1.0));
const metal::float4 _tmp1 = metal::float4(v50.x, v50.y, v50.z, 1.0);
return fs_eOutput { _tmp1 };
}

