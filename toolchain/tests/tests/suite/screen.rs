//! The screen's colour (language.md §18.8): a program asks for an HDR screen in Display P3
//! (`std::gpu::screen`), and once every pipeline that draws the screen has a variant for it the
//! browser shows the canvas in `rgba16float`, tone mapped `extended`, in `display-p3`; a white of
//! 2.0 stays 2.0 in the screen's texels, and `screen_is` says so. Test mode does this only when
//! the test asks (`hdr=1`); otherwise its screen, so its goldens, stay as they were. The native
//! host, drawing offscreen, keeps SDR in sRGB.

use wrela_host::{Host, Value};

/// A program that asks for an HDR screen in Display P3 at its first frame, draws it white at
/// 2.0, and tells what `screen_is` says.
const HDR: &str =
    "use std::gpu::{ClipPosition, Gamut, VertexIndex, begin_screen_pass, draw, screen, screen_is}

pub struct S {
    frames: u32,
}

pub fn init() -> S {
    S { frames: 0 }
}

@vertex
fn cover(v: VertexIndex) -> ClipPosition {
    let x = f32((v.index << 1) & 2) * 2.0 - 1.0
    let y = f32(v.index & 2) * 2.0 - 1.0
    ClipPosition { position: vec4(x, y, 0.5, 1.0) }
}

@fragment
fn bright() -> vec4 {
    vec4(2.0, 2.0, 2.0, 1.0)
}

pub fn frame(s: mut S, time: f32, width: u32, height: u32) {
    if s.frames == 0 {
        screen(hdr: true, gamut: Gamut::DisplayP3)
    }
    s.frames += 1
    let p = begin_screen_pass(clear: vec4(0.0))
    draw(p, cover, bright, vertices: 3)
    p.present()
}

/// Bit 0: the screen is HDR; bit 1: Display P3.
pub fn shown(s: S) -> u32 {
    let is = screen_is()
    u32(is.hdr) + 2 * u32(is.gamut == Gamut::DisplayP3)
}
";

/// The native host keeps its screen: the command changes nothing, the white clamps, and
/// `screen_is` says SDR in sRGB.
#[test]
#[ignore = "needs a GPU"]
fn a_screen_stays_sdr_where_it_cant_change() {
    let pkg = crate::package("screen-sdr", HDR);
    let built = wrela_tests::build(&pkg).unwrap_or_else(|e| panic!("doesn't build:\n{e}"));
    let dir = crate::scratch("screen-sdr-page");
    built.write_to(&dir).expect("the build");
    let mut host = Host::load(&dir).expect("load");
    host.init().expect("init");
    for k in 0..3 {
        host.lockstep_frame(k, 60.0, 64, 64, &[]).expect("a frame");
    }
    let shown = host.call_export("shown", &[]).expect("shown");
    assert_eq!(shown, [Value::I32(0)], "the native host keeps SDR in sRGB");
}

/// Let (`Options::screen_hdr`, a test's `hdr=1`), the native host shows the screen HDR in
/// Display P3: `screen_is` says so, and the white of 2.0 stays 2.0 in the screen's half floats.
#[test]
#[ignore = "needs a GPU"]
fn the_native_host_shows_an_hdr_screen_where_its_let() {
    let pkg = crate::package("screen-hdr", HDR);
    let built = wrela_tests::build(&pkg).unwrap_or_else(|e| panic!("doesn't build:\n{e}"));
    let dir = crate::scratch("screen-hdr-page");
    built.write_to(&dir).expect("the build");
    let options = wrela_host::Options { screen_hdr: true, ..Default::default() };
    let mut host = Host::load_with(&dir, &options).expect("load");
    host.init().expect("init");
    for k in 0..3 {
        host.lockstep_frame(k, 60.0, 64, 64, &[]).expect("a frame");
    }
    assert_eq!(host.call_export("shown", &[]).expect("shown"), [Value::I32(3)]);
    let floats = host.read_screen_floats().expect("the screen");
    assert_eq!(&floats[..4], [2.0, 2.0, 2.0, 1.0], "past white, kept");
}

/// The opening on an HDR screen shows, below white, the light its SDR screen shows: each
/// channel of a pixel the SDR screen doesn't clip, sRGB's curve undone, within the 8-bit
/// screen's rounding (lib/frame's output pass, `last::output`). An SDR screen, the picture is
/// as before to the bit (`tools/hosts` against a build without the output transform).
#[test]
#[ignore = "needs a GPU"]
fn the_opening_shows_the_same_light_on_an_hdr_screen_below_white() {
    let (dir, _) = wrela_tests::page("game/opening", "opening-hdr");
    let run = |hdr: bool| {
        let options = wrela_host::Options { screen_hdr: hdr, ..Default::default() };
        let mut host = Host::load_with(&dir, &options).expect("load");
        host.init().expect("init");
        for k in 0..4 {
            host.lockstep_frame(k, 60.0, 160, 90, &[]).expect("a frame");
        }
        (host.screen_is(), host.read_screen_floats().expect("the screen"))
    };
    let (sdr_is, sdr) = run(false);
    let (hdr_is, hdr) = run(true);
    assert_eq!((sdr_is & 1, hdr_is & 1), (0, 1));
    let linear = |v: f32| {
        if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
    };
    let mut seen = 0;
    for (k, (&s, &h)) in sdr.iter().zip(&hdr).enumerate() {
        if k % 4 == 3 || s >= 254.0 / 255.0 {
            continue;
        }
        // Half an 8-bit step either way, through sRGB's curve; and two half floats' rounding
        // (the display value drawn, then the light: a step of 2^-11 near 1, which sRGB's curve
        // up to doubles).
        let (lo, hi) = (linear((s - 0.5 / 255.0).max(0.0)), linear(s + 0.5 / 255.0));
        let slack = 3.0e-3 * hi + 1.0e-4;
        assert!(
            h >= lo - slack && h <= hi + slack,
            "value {k}: SDR {s} is linear {lo}..{hi}, HDR {h}"
        );
        seen += 1;
    }
    assert!(seen > 1000, "{seen} values compared");
}
