//! The GPU's real limits (AC10): both hosts open the device with the adapter's limits, a program
//! sees them through `std::gpu::limits()`, and commands are checked against them, so a texture
//! may be as wide as the GPU allows, past WebGPU's default 8192. Needs a GPU:
//! `cargo test -p wrela-tests --test suite gpu_limits:: -- --ignored`.

use wrela_host::Host;
use wrela_tests::{one_u32, page};

#[test]
#[ignore = "needs a GPU"]
fn a_program_sees_the_adapters_limits() {
    let (dir, _) = page("toolchain/tests/gpu-limits", "gpu-limits-native");
    let mut host = Host::load(&dir).expect("load");
    // What the GPU says of itself, through the same device request.
    let device = wrela_host::open_device("limits test", None).expect("device");
    let l = device.limits().clone();
    let texture = one_u32(&mut host, "max_texture_size", &[]);
    assert_eq!(texture, l.max_texture_dimension_2d);
    let buffer = l.max_buffer_size.min(l.max_storage_buffer_binding_size).min(u64::from(u32::MAX));
    assert_eq!(u64::from(one_u32(&mut host, "max_buffer_size", &[])), buffer);
    assert_eq!(
        one_u32(&mut host, "max_workgroups_per_dimension", &[]),
        l.max_compute_workgroups_per_dimension
    );
    // At least WebGPU's defaults; the texture `init` made is as wide as the limit.
    assert!(texture >= 8192, "a texture limit of {texture}");
    assert_eq!(one_u32(&mut host, "wide", &[]), texture);
    eprintln!("the adapter's textures go to {texture} texels, its buffers to {buffer} bytes");
}

/// A level as wide and tall as the adapter allows can hold more bytes than a `u32` counts (16384
/// × 16384 `Rgba32Float` texels are 4 GiB): std counts a write's, a copy's and a read's bytes in
/// `u64`, and one that can't hold them panics, saying why, rather than overflow.
#[test]
#[ignore = "needs a GPU"]
fn the_largest_levels_bytes_are_counted_past_4_gib() {
    let text = "use std::gpu::{
    GpuBuffer, Rgba32Float, Texture, buffer, copy_buffer_to_texture, copy_texture_to_buffer,
    limits, read_texture, texture,
}

pub fn frame(time: f32, width: u32, height: u32) {}

pub fn side() -> u32 {
    limits().max_texture_size
}

fn largest() -> Texture<Rgba32Float> {
    texture(width: 16384, height: 16384)
}

pub fn write() {
    var t = largest()
    t.write(x: 0, y: 0, width: 16384, height: 16384, bytes: [u8(0); 16])
}

pub fn to_buffer() {
    var b: GpuBuffer<u32> = buffer(64)
    copy_texture_to_buffer(largest().level(0), mut b)
}

pub fn from_buffer() {
    let b: GpuBuffer<u32> = buffer(64)
    var t = largest()
    copy_buffer_to_texture(b.span(start: 0, count: 64), t.level_mut(0))
}

pub fn read() {
    let _ = read_texture(largest().level(0))
}
";
    let dir = crate::package("gpu-limits-largest", text);
    wrela_tests::must_build(&dir, &dir.join("build"));
    let mut host = Host::load(dir.join("build")).expect("load");
    let side = one_u32(&mut host, "side", &[]);
    assert!(side >= 16384, "the adapter's textures go to {side} texels; this needs 16384");
    for (export, why) in [
        ("write", "write: the bytes aren't the rectangle's texels"),
        ("to_buffer", "copy_texture_to_buffer: the buffer is smaller than the texture's"),
        ("from_buffer", "copy_buffer_to_texture: the span is smaller than the texture's"),
        ("read", "read_texture: the level's padded rows are 4 GiB or more"),
    ] {
        let err = host.call_export(export, &[]).expect_err(export);
        assert!(format!("{err:#}").contains(why), "{export}: {err:#}");
    }
}

/// Pipeline tiers (toolchain/tests/tiers, language.md §18.9): the manifest marks `mark`, which
/// only frame 1 dispatches, tier 1 (`conditional`) and `fill` tier 0.
#[test]
fn tiers_are_marked_and_reported() {
    let (dir, _) = wrela_tests::page("toolchain/tests/tiers", "tiers");
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("manifest.json")).expect("manifest"),
    )
    .expect("JSON");
    let tiers: Vec<(&str, bool)> = manifest["pipelines"]
        .as_array()
        .expect("pipelines")
        .iter()
        .map(|p| (p["name"].as_str().unwrap_or(""), p["conditional"].as_bool().unwrap_or(false)))
        .collect();
    assert_eq!(tiers, [("fill", false), ("mark", true)]);
}

/// `std::gpu::LANE_WORK_LEAST` is the least of the targets' `lane_work`: what an engine that
/// chooses at build how much one invocation does divides by.
#[test]
fn the_least_lane_work_is_the_targets() {
    let std_gpu = include_str!("../../../../lib/std/gpu.wrela");
    let line = std_gpu
        .lines()
        .find_map(|l| l.strip_prefix("pub const LANE_WORK_LEAST: u32 = "))
        .expect("std::gpu::LANE_WORK_LEAST");
    let least = wrela_abi::Target::ALL.iter().map(|t| t.limits().device.lane_work).min();
    assert_eq!(line.trim().parse::<u32>().ok(), least);
}
