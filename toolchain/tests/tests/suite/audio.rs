//! `@audio` (AC6, language.md §16.4): toolchain/tests/audio's modal-synthesis voice, rendered
//! offline by the native host and in Chrome's AudioWorklet, gives the same samples, bit for bit.

use wrela_tests::page;

/// Two seconds at 48 kHz, in quanta of 128 samples.
const QUANTA: u32 = 750;

/// The voice's samples from the native host's offline run.
fn native(dir: &std::path::Path) -> Vec<f32> {
    let mut host = crate::cpu_host(dir);
    host.frame(0.0, 16, 16).expect("frame");
    host.render_audio(QUANTA).expect("render")
}

#[test]
fn the_voice_sounds() {
    let (dir, _) = page("toolchain/tests/audio", "audio-native");
    let samples = native(&dir);
    assert_eq!(samples.len(), QUANTA as usize * 128 * 2);
    // Two channels, interleaved: the bars are to the left.
    let (l, r): (Vec<f32>, Vec<f32>) = samples.chunks_exact(2).map(|p| (p[0], p[1])).unzip();
    assert!(l.iter().zip(&r).all(|(a, b)| *b == 0.5 * *a));
    // Silent until the first strike, then ringing, within [-1, 1].
    let peak = samples.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(peak > 0.05 && peak <= 1.0, "peak {peak}");
    assert!(samples.iter().all(|x| x.is_finite()));
    // Rendering again from the start gives the same samples.
    assert_eq!(native(&dir), samples);
}

/// The voice played live on the machine's audio device (backend Phase 3d, `wrela_host::voice`):
/// the samples its callbacks wrote are the offline render's, bit for bit. It plays a second of
/// the voice aloud, so it runs only with `WRELA_LIVE_AUDIO=1` (and the GPU tests' `--ignored`),
/// and needs an output device at 48 kHz.
#[test]
#[ignore]
fn the_live_voice_is_the_offline_voice() {
    if std::env::var_os("WRELA_LIVE_AUDIO").is_none() {
        eprintln!("not run: it plays aloud; WRELA_LIVE_AUDIO=1 runs it");
        return;
    }
    const LIVE: usize = 375 * 128 * 2; // a second's samples, both channels
    let (dir, _) = page("toolchain/tests/audio", "audio-native");
    let mut host = crate::cpu_host(&dir);
    host.frame(0.0, 16, 16).expect("frame");
    let playing = host.play_voice(LIVE).expect("an audio device at 48 kHz");
    let began = std::time::Instant::now();
    while playing.tapped() < LIVE {
        assert!(began.elapsed().as_secs() < 10, "the device played {} samples", playing.tapped());
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let played = host.stop_voice(playing).expect("the voice played");
    eprintln!(
        "{}: {} channels, {} callbacks of {}..{} frames, slowest {:.1}% of its buffer, {} over, {} busy, errors {:?}",
        played.device,
        played.channels,
        played.callbacks,
        played.frames.0,
        played.frames.1,
        played.worst * 100.0,
        played.over,
        played.busy,
        played.errors
    );
    assert_eq!(played.tap.len(), LIVE);
    let offline = native(&dir);
    let same = played.tap.iter().zip(&offline).take_while(|(a, b)| a.to_bits() == b.to_bits());
    assert_eq!(same.count(), LIVE, "the live voice differs from the offline render there");
    assert_eq!(played.busy, 0, "no callback found the voice busy");
    // The voice goes on offline from where the device stopped.
    let next = host.render_audio(1).expect("the next quantum");
    let quanta = played.quanta as usize;
    let mut fresh = crate::cpu_host(&dir);
    fresh.frame(0.0, 16, 16).expect("frame");
    let through = fresh.render_audio(quanta as u32 + 1).expect("render");
    assert_eq!(next, through[quanta * 256..], "the quantum after the device's");
}
