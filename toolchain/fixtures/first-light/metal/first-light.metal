// first-light's render pipeline: one full-screen triangle; each pixel is coloured by its distance
// to a circle whose centre the CPU computes each frame. Colours come from the palette that the
// `fill` compute pipeline derived on the first frame.
// Metal shading language as the toolchain's MSL writer (naga 30's, `wrela_wgsl::msl`) wrote it
// from the WGSL this fixture held until 2026-11-15, its bindings in `wrela_abi::metal`'s slots.
// language: metal4.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
uint size1;
};

struct Frame {
metal::float2 resolution;
metal::float2 center;
float time;
float radius;
metal::float2 _pad;
};
typedef uint type_3[1];

metal::float3 colour(
uint i_1,
device type_3 const& palette,
constant _mslBufferSizes& _buffer_sizes
) {
uint _e3 = palette[metal::min(unsigned(i_1), (_buffer_sizes.size1 - 0 - 4) / 4)];
return metal::unpack_unorm4x8_to_float(_e3).xyz;
}

struct vsInput {
};
struct vsOutput {
metal::float4 member [[position]];
};
vertex vsOutput vs(
uint i [[vertex_id]]
) {
float x = static_cast<float>(as_type<int>(as_type<uint>(as_type<int>(as_type<uint>(static_cast<int>(i & 1u)) * as_type<uint>(4))) - as_type<uint>(1)));
float y = static_cast<float>(as_type<int>(as_type<uint>(as_type<int>(as_type<uint>(static_cast<int>(i >> 1u)) * as_type<uint>(4))) - as_type<uint>(1)));
return vsOutput { metal::float4(x, y, 0.0, 1.0) };
}


struct fsInput {
};
struct fsOutput {
metal::float4 member_1 [[color(0)]];
};
fragment fsOutput fs(
metal::float4 pos [[position]]
, constant Frame& frame [[buffer(0)]]
, device type_3 const& palette [[buffer(1)]]
, constant _mslBufferSizes& _buffer_sizes [[buffer(2)]]
) {
metal::float3 c = {};
metal::float2 p = pos.xy;
metal::float3 _e3 = colour(0u, palette, _buffer_sizes);
metal::float3 _e5 = colour(2u, palette, _buffer_sizes);
float _e10 = frame.resolution.y;
c = metal::mix(_e3, _e5, p.y / _e10);
metal::float2 _e16 = frame.center;
float _e20 = frame.radius;
float d = metal::distance(p, _e16) - _e20;
metal::float3 _e22 = c;
metal::float3 _e24 = colour(7u, palette, _buffer_sizes);
float _e32 = frame.radius;
c = _e22 + ((_e24 * 0.4) * metal::exp(-(metal::max(d, 0.0)) / (0.3 * _e32)));
float _e41 = frame.time;
float pulse = 0.85 + (0.15 * metal::cos(_e41 * 3.0));
metal::float3 _e50 = colour(5u, palette, _buffer_sizes);
metal::float3 _e52 = c;
c = metal::mix(_e50 * pulse, _e52, metal::smoothstep(-1.0, 1.0, d));
metal::float3 _e57 = c;
return fsOutput { metal::float4(metal::min(_e57, metal::float3(1.0)), 1.0) };
}
