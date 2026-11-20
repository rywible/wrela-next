use super::*;
use wrela_abi::manifest::Pipeline;

#[test]
fn hardware_rendering_rejects_software_virtual_and_unknown_adapters() {
    for kind in [
        wrela_gpu::DeviceType::Cpu,
        wrela_gpu::DeviceType::VirtualGpu,
        wrela_gpu::DeviceType::Other,
    ] {
        let error = admits("test adapter", kind, false).unwrap_err().to_string();
        assert!(
            error.contains("hardware GPU required") && error.contains("test adapter"),
            "{error}"
        );
    }
    for kind in [wrela_gpu::DeviceType::IntegratedGpu, wrela_gpu::DeviceType::DiscreteGpu] {
        assert!(admits("hardware", kind, false).is_ok());
        assert!(admits("hardware", kind, true).is_ok());
    }
}

#[test]
fn a_software_adapter_runs_only_when_allowed() {
    let refused = admits("llvmpipe", wrela_gpu::DeviceType::Cpu, false).unwrap_err().to_string();
    assert!(refused.contains("WRELA_ALLOW_SOFTWARE=1"), "{refused}");
    assert!(admits("llvmpipe", wrela_gpu::DeviceType::Cpu, true).is_ok());
    // Allowing software never admits a virtual or unknown adapter.
    for kind in [wrela_gpu::DeviceType::VirtualGpu, wrela_gpu::DeviceType::Other] {
        assert!(admits("other", kind, true).is_err());
    }
}

#[test]
#[ignore = "needs a GPU"]
fn indirect_draws_enable_nonzero_first_instances() {
    let gpu = Gpu::new(&Manifest::new("game.wasm"), &[], Timing::Off, false).unwrap();
    assert!(gpu.device.features().has("indirect-first-instance"));
}

#[test]
#[ignore = "needs a GPU"]
fn buffer_readback_reuses_storage_and_reads_only_the_requested_range() {
    let mut gpu = Gpu::new(&Manifest::new("game.wasm"), &[], Timing::Off, false).unwrap();
    gpu.create_buffer(1, 64);
    let mut previous: Option<wrela_gpu::Buffer> = None;
    for (frame, size) in [16, 8, 16, 32, 4, 32, 0].into_iter().enumerate() {
        let bytes: Vec<u8> = (0..64).map(|i| (frame * 29 + i) as u8).collect();
        let offset = (frame % 3 * 4) as u32;
        gpu.write_buffer(1, 0, &bytes).unwrap();
        assert_eq!(
            gpu.read_range(1, offset, size).unwrap(),
            bytes[offset as usize..(offset + size) as usize],
        );
        let staging = gpu.buffer_readback.as_ref().unwrap();
        if let Some(before) = &previous {
            if before.size() >= u64::from(size.max(4)) {
                assert!(staging.same(before), "a read that fits reuses its storage");
            } else {
                assert!(!staging.same(before), "a larger read grows its storage");
            }
        }
        previous = Some(staging.clone());
    }
    gpu.check_errors().unwrap();
}

/// The present pass draws each pixel of the screen into a frame of each format a surface has
/// (`rgba8unorm` on Vulkan, `rgba16float` on Metal): a gradient's every texel, exactly
/// (to 8 bits, read back from the half floats).
#[test]
#[ignore = "needs a GPU"]
fn the_present_pass_draws_the_screen_into_a_frame() {
    let device = open_device("present test", None).unwrap();
    let (w, h) = (37u32, 23u32);
    let texels: Vec<u8> = (0..w * h)
        .flat_map(|i| [(i * 7) as u8, (i * 13 + 5) as u8, (i % 251) as u8, 255 - i as u8])
        .collect();
    let src = device.create_texture(&wrela_gpu::TextureDesc {
        label: "screen",
        width: w,
        height: h,
        depth_or_layers: 1,
        levels: 1,
        dimension: wrela_gpu::TextureDimension::D2,
        format: TextureFormat::Rgba8,
        usage: wrela_gpu::TextureUsage { sampled: true, copy_dst: true, ..Default::default() },
    });
    let at =
        wrela_gpu::TextureCopy { texture: &src, level: 0, origin: [0, 0, 0], depth_only: false };
    device.write_texture(at, &texels, w * 4, h, [w, h, 1]);
    let view = |t: &wrela_gpu::Texture| {
        t.create_view(&wrela_gpu::ViewDesc {
            dimension: wrela_gpu::ViewDimension::D2,
            base_level: 0,
            level_count: Some(1),
            base_layer: 0,
            layer_count: Some(1),
        })
    };
    for format in [TextureFormat::Rgba8, TextureFormat::Rgba16Float] {
        let (pipeline, layout) = present_pipeline(&device, format).unwrap();
        let dst = device.create_texture(&wrela_gpu::TextureDesc {
            label: "frame",
            width: w,
            height: h,
            depth_or_layers: 1,
            levels: 1,
            dimension: wrela_gpu::TextureDimension::D2,
            format,
            usage: wrela_gpu::TextureUsage { render: true, copy_src: true, ..Default::default() },
        });
        let cb = draw_presented(&device, &pipeline, &layout, &view(&src), &view(&dst));
        let texel = format.bytes_per_texel();
        let row = align_to(u64::from(w * texel), 256) as u32;
        let read = readback_buffer(&device, u64::from(row * h));
        let mut enc = device.create_command_encoder("read");
        enc.copy_texture_to_buffer(
            wrela_gpu::TextureCopy {
                texture: &dst,
                level: 0,
                origin: [0, 0, 0],
                depth_only: false,
            },
            wrela_gpu::BufferCopy {
                buffer: &read,
                offset: 0,
                bytes_per_row: row,
                rows_per_image: h,
            },
            [w, h, 1],
        );
        device.submit(vec![cb, enc.finish()]);
        let bytes = map_read(&device, &read).unwrap();
        let got: Vec<u8> = (0..h as usize)
            .flat_map(|y| {
                let line = &bytes[y * row as usize..][..(w * texel) as usize];
                line.chunks(texel as usize).flat_map(|t| as_rgba8(format, t)).collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(got, texels, "{format:?}");
        assert!(device.take_errors().is_empty());
    }
}

/// A build of one compute pipeline, `increment`, that adds one to its buffer's first word.
fn increment() -> (Manifest, [Vec<u8>; 1]) {
    let mut manifest = Manifest::new("game.wasm");
    manifest.pipelines.push(Pipeline {
        name: "increment".into(),
        shader: "increment.metal".into(),
        stage: Stage::Compute { entry: "increment".into(), workgroup_size: [1, 1, 1] },
        uniform: None,
        bindings: vec![ResourceBinding {
            binding: 0,
            kind: BindingKind::ReadWrite,
            stage: wrela_abi::manifest::BindingStage::Both,
            format: None,
        }],
        debug_flag: None,
        requires: Vec::new(),
        conditional: false,
        standin: None,
        sizes: None,
        kept: 0,
        // Hand-written: its bound is stated here (a read, an add and a write).
        work: vec![Some(24)],
        unbounded: Vec::new(),
    });
    // `count[0] += 1u`, of a storage buffer of `u32`s at binding 0.
    use wrela_shader::build::{BinOp, Builder, Scalar};
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let count = b.storage("count", 0, u, true);
    b.compute("increment", [1, 1, 1], &[], |f| {
        let zero = f.u32(0);
        let one = f.u32(1);
        let old = f.load(u, Builder::element(count, zero));
        let new = f.binary(BinOp::Add, old, one);
        f.store(Builder::element(count, zero), new);
    });
    let target = wrela_abi::Target::HERE;
    let p = &mut manifest.pipelines[0];
    p.shader = format!("increment.{}", wrela_shader::extension(target));
    let shader = b.write(p, target).unwrap();
    (manifest, [shader.bytes])
}

/// The calibration kernel (the guard's unit, `crate::step`): five timings on this machine,
/// each snapping to the same unit; on the M4 and the Radeon, the target's own
/// (`step_femtoseconds`), so nothing a program paces by `step_work` moves.
#[test]
#[ignore = "needs a GPU"]
fn the_calibration_kernel_snaps_this_machine_to_its_targets_unit() {
    let limits = wrela_abi::Target::HERE.limits();
    let mut units = Vec::new();
    for _ in 0..5 {
        let (fs, ratio, ns) = measured_unit().unwrap();
        eprintln!("calibration kernel: {ns:.0} ns, {ratio:.2} times the reference: {fs} fs");
        units.push(fs);
    }
    assert!(units.iter().all(|&u| u == limits.step_femtoseconds), "{units:?}");
}

/// A submission takes dispatches and passes until one more would take it past two steps of
/// the guard's bound or 64 of them; one alone always goes.
#[test]
fn a_submission_is_full_at_two_steps_of_bound_or_64_commands() {
    let q = |commands, bound_ms| Queued { commands, bound_ms, passes_ms: 0.0 };
    assert!(!q(0, 0.0).full(1.0e9), "a dispatch alone");
    assert!(!q(1, 50.0).full(50.0));
    assert!(q(1, 50.0).full(50.1));
    assert!(!q(63, 0.0).full(0.0));
    assert!(q(64, 0.0).full(0.0));
}

/// A frame of 5,000 short dispatches (a band of an audit each, well within the guard) is
/// many submissions, none past 64 of them, not one job of seconds; and a frame of heavy ones
/// (each a third of the bound a submission may hold) is many too, none past that bound
/// (2026-11-19: the Radeon's resets, backend.md item 32). `tools/check` runs it on each
/// machine's GPU: any change to when the host submits is tested against it.
#[test]
#[ignore = "needs a GPU"]
fn a_frame_of_many_dispatches_is_many_submissions() {
    let (manifest, shaders) = increment();
    let mut gpu = Gpu::new(&manifest, &shaders, Timing::Off, false).unwrap();
    gpu.create_buffer(1, 4);
    let (before, _) = gpu.submissions();
    for _ in 0..5000 {
        gpu.dispatch(0, Ok([1, 1, 1]), &[Binding::range(1, 0, 4)], &[]).unwrap();
    }
    gpu.end_frame().unwrap();
    let (after, largest) = gpu.submissions();
    let made = after - before;
    assert!(made >= 5000 / u64::from(SUBMIT_COMMANDS), "{made} submissions");
    assert!(largest.commands <= SUBMIT_COMMANDS, "{largest:?}");
    assert!(largest.bound_ms <= SUBMIT_BOUND_MS, "{largest:?}");
    assert_eq!(gpu.read_buffer(1).unwrap(), 5000u32.to_le_bytes());
    gpu.check_errors().unwrap();

    // Heavy by its bound (a stated bound, not real work: the kernel adds one).
    let target = wrela_abi::Target::HERE;
    let one = crate::step::ms_of(1, 1, target);
    let work = (SUBMIT_BOUND_MS / 3.0 / one).floor() as u64;
    let ms = crate::step::ms_of(work, 1, target);
    assert!(ms <= crate::step::STEP_MS && ms > SUBMIT_BOUND_MS / 4.0, "{ms} ms");
    let (mut manifest, shaders) = increment();
    manifest.pipelines[0].work = vec![Some(work)];
    let mut gpu = Gpu::new(&manifest, &shaders, Timing::Off, false).unwrap();
    gpu.create_buffer(1, 4);
    let (before, _) = gpu.submissions();
    for _ in 0..30 {
        gpu.dispatch(0, Ok([1, 1, 1]), &[Binding::range(1, 0, 4)], &[]).unwrap();
    }
    gpu.end_frame().unwrap();
    let (after, largest) = gpu.submissions();
    assert!(after - before >= 10, "{} submissions of 30 heavy dispatches", after - before);
    assert!(largest.bound_ms <= SUBMIT_BOUND_MS, "{largest:?}");
    assert_eq!(gpu.read_buffer(1).unwrap(), 30u32.to_le_bytes());
    gpu.check_errors().unwrap();
}

/// A small pass after a long stretch of CPU work, frame after frame (the engine's judge): its
/// submission's span on Metal starts near its commit, so a wait before it is in the span;
/// the guard judges a piece alone by its pass's own stamps, and the run goes on. A pass that
/// truly holds the GPU over the line still ends it, by its own time; over four lines, at once.
#[test]
#[ignore = "needs a GPU"]
fn a_small_pass_after_cpu_work_is_judged_by_its_own_time() {
    let (manifest, shaders) = heavy_draw();
    let mut gpu = Gpu::new(&manifest, &shaders, Timing::Off, false).unwrap();
    if gpu.stamps.is_none() {
        eprintln!("this GPU stamps no passes: nothing to test");
        return;
    }
    gpu.create_texture(
        2,
        [HEAVY_SIDE, HEAVY_SIDE, 1],
        TextureFormat::Rgba16Float,
        false,
        TextureKind::D2,
        1,
    );
    // The pass's own time, warmed, and kept warm (a GPU left idle runs it slow for real).
    let warmer = Warmer::start(gpu.device.clone());
    for _ in 0..200 {
        heavy_frame(&mut gpu, "small", 1, 1).unwrap();
    }
    let slot = (gpu.stamp_next + STAMP_RING - 1) % STAMP_RING;
    let own = stamped_ms(&gpu.device, gpu.stamps.as_ref().unwrap(), slot).expect("stamped");
    eprintln!("the small pass's own time: {own:.4} ms");
    // A line of four times that: a wait of more than three passes in any span would be over.
    let line = gpu.line_ms;
    gpu.line_ms = (4.0 * own).max(0.05);
    let mut over_by_span = 0;
    for _ in 0..200 {
        std::thread::sleep(std::time::Duration::from_millis(20));
        heavy_frame(&mut gpu, "small", 1, 1).unwrap();
        over_by_span += gpu.measured.iter().filter(|(_, ms)| *ms > gpu.line_ms).count();
        gpu.measured.clear();
    }
    eprintln!("spans over the line {over_by_span} of 200, and the run went on");
    // The same judgement, by hand: a span far over the line (within four lines), its own small.
    let job = |label: &str, slot: u32| Job {
        index: Default::default(),
        frame: 0,
        call: 0,
        pieces: vec![Piece { label: Arc::from(label), draws: 0..1, of: 1 }],
        alone: true,
        dispatches: false,
        stamp: Some(slot),
    };
    let slot = (gpu.stamp_next + STAMP_RING - 1) % STAMP_RING;
    // The stamp judged by hand: the last piece measured alone, after the run above.
    let own = stamped_ms(&gpu.device, gpu.stamps.as_ref().unwrap(), slot).expect("stamped");
    gpu.line_ms = (4.0 * own).max(0.05);
    for _ in 0..3 {
        assert!(gpu.judge(job("small", slot), 3.0 * gpu.line_ms).is_ok(), "a wait in the span");
    }
    // Truly over: its own time over a line lowered under it ends the run the second time.
    gpu.line_ms = own / 2.0;
    assert!(gpu.judge(job("heavy", slot), own).is_ok());
    let e = gpu.judge(job("heavy", slot), own).unwrap_err().to_string();
    assert!(e.contains("the pass `heavy`"), "{e}");
    // And far over (its own over four lines): at once.
    gpu.line_ms = own / 8.0;
    let e = gpu.judge(job("runaway", slot), own).unwrap_err().to_string();
    assert!(e.contains("the pass `runaway`"), "{e}");
    // Far over its usual (learned above, under the line): said.
    let unstamped = Job { stamp: None, ..job("small", slot) };
    let e = gpu.judge(unstamped, 5.0 * own).unwrap_err().to_string();
    assert!(e.contains("times its usual"), "{e}");
    gpu.line_ms = line;
    drop(warmer);
}

/// A pass of `instances` heavy triangles: its own time (serial stamps), in ms.
fn own_pass_ms(gpu: &mut Gpu, instances: u32) -> f64 {
    heavy_frame(gpu, "small", 1, instances).unwrap();
    let t = gpu.take_timings().unwrap();
    t.iter().filter(|t| t.label == "small").map(|t| t.nanos / 1e6).sum()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// A measurement, not a check: how much slower a pass runs, by its own time, after the GPU
/// sat idle (5 ms to a second), and in a run of passes each after 20 ms of CPU work (the
/// engine's judge's shape); passes of 1, 15 and 150 triangles; without a warmer and kept
/// clocked ([`Warmer`]), with the CPU it costs while the GPU idles.
#[test]
#[ignore = "a measurement: run by hand"]
fn how_much_a_pass_slows_after_idle() {
    let (manifest, shaders) = heavy_draw();
    let mut gpu = Gpu::new(&manifest, &shaders, Timing::Serial, false).unwrap();
    gpu.create_texture(
        2,
        [HEAVY_SIDE, HEAVY_SIDE, 1],
        TextureFormat::Rgba16Float,
        false,
        TextureKind::D2,
        1,
    );
    let cpu_ms = || {
        let pid = std::process::id().to_string();
        let out = std::process::Command::new("ps").args(["-o", "time=", "-p", &pid]).output();
        // `[hh:]mm:ss[.ss]` (macOS gives minutes and hundredths, Linux hours and seconds).
        let t = String::from_utf8_lossy(&out.unwrap().stdout).trim().to_string();
        let t = t.rsplit('-').next().unwrap_or_default();
        t.split(':').fold(0.0, |a, f| a * 60.0 + f.parse::<f64>().unwrap()) * 1e3
    };
    for instances in [1, 15, 150] {
        for kept in [false, true] {
            let warmer = kept.then(|| Warmer::start(gpu.device.clone()));
            for _ in 0..200 {
                own_pass_ms(&mut gpu, instances);
            }
            let busy = median((0..21).map(|_| own_pass_ms(&mut gpu, instances)).collect());
            let mut line = format!(
                "{instances:>3} triangles, kept clocked {kept:>5}: back to back {busy:.3} ms; after idle (median / most of 5):"
            );
            for idle in [5u64, 20, 100, 1000] {
                let mut r = Vec::new();
                for _ in 0..5 {
                    for _ in 0..10 {
                        own_pass_ms(&mut gpu, instances);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(idle));
                    r.push(own_pass_ms(&mut gpu, instances));
                }
                let most = r.iter().copied().fold(0.0, f64::max);
                line += &format!("  {idle} ms {:.3} / {most:.3}", median(r));
            }
            let mut run = Vec::new();
            for _ in 0..100 {
                std::thread::sleep(std::time::Duration::from_millis(20));
                run.push(own_pass_ms(&mut gpu, instances));
            }
            let most = run.iter().copied().fold(0.0, f64::max);
            line += &format!("; 100 after 20 ms each {:.3} / {most:.3}", median(run));
            let before = cpu_ms();
            std::thread::sleep(std::time::Duration::from_secs(2));
            line += &format!("; CPU while idle {:.1}% of a core", (cpu_ms() - before) / 20.0);
            eprintln!("{line}");
            drop(warmer);
        }
    }
    gpu.check_errors().unwrap();
}

/// Kept clocked, a pass after the program's CPU work runs at its warm time, by its own
/// stamps: 200 frames, each a pass of 15 triangles after 20 ms of CPU work (the engine's
/// judge's shape): the median within a third of the pass back to back, nine in ten within
/// twice it. (Without the warmer, on the M4, the median was 2.4 to 4 times.)
#[test]
#[ignore = "needs a GPU"]
fn a_pass_after_cpu_work_runs_at_its_warm_time_when_kept_clocked() {
    let (manifest, shaders) = heavy_draw();
    let mut gpu = Gpu::new(&manifest, &shaders, Timing::Serial, false).unwrap();
    gpu.create_texture(
        2,
        [HEAVY_SIDE, HEAVY_SIDE, 1],
        TextureFormat::Rgba16Float,
        false,
        TextureKind::D2,
        1,
    );
    let warmer = Warmer::start(gpu.device.clone());
    for _ in 0..200 {
        own_pass_ms(&mut gpu, 15);
    }
    let warm = median((0..21).map(|_| own_pass_ms(&mut gpu, 15)).collect());
    let mut run = Vec::new();
    for _ in 0..200 {
        std::thread::sleep(std::time::Duration::from_millis(20));
        run.push(own_pass_ms(&mut gpu, 15) / warm);
    }
    drop(warmer);
    run.sort_by(f64::total_cmp);
    let (mid, ninth, most) = (run[100], run[180], run[199]);
    eprintln!(
        "kept clocked: the pass {warm:.3} ms back to back; after 20 ms of CPU work, 200 times: median {mid:.2}x, nine in ten within {ninth:.2}x, most {most:.2}x"
    );
    assert!(mid < 1.33 && ninth < 2.0, "{mid} {ninth}");
    gpu.check_errors().unwrap();
}

/// Indirect dispatches whose most groups are stated (`most:`) are booked as that many, so a
/// few small ones share a submission as direct ones do; unstated, each is booked at the
/// limit, and every second one closes a submission (backend.md 72). The work is the same.
#[test]
#[ignore = "needs a GPU"]
fn an_indirect_dispatch_of_stated_most_shares_a_submission() {
    let (manifest, shaders) = increment();
    let run = |most: u32| {
        let mut gpu = Gpu::new(&manifest, &shaders, Timing::Off, false).unwrap();
        gpu.create_buffer(1, 4);
        gpu.create_buffer(2, 12);
        gpu.write_buffer(2, 0, &[1u32, 1, 1].map(u32::to_le_bytes).concat()).unwrap();
        let (before, _) = gpu.submissions();
        for _ in 0..8 {
            gpu.dispatch(0, Err((2, 0, most)), &[Binding::range(1, 0, 4)], &[]).unwrap();
        }
        gpu.end_frame().unwrap();
        let (after, _) = gpu.submissions();
        assert_eq!(gpu.read_buffer(1).unwrap(), 8u32.to_le_bytes());
        gpu.check_errors().unwrap();
        after - before
    };
    let unstated = run(0);
    let stated = run(1);
    assert!(unstated >= 4, "{unstated} submissions unstated");
    assert!(stated <= 2, "{stated} submissions stated");
}

/// How many inputs the fused-or-not fixture computes.
const FUSED_N: u32 = 4096;

/// A build of two compute pipelines over buffers `a`, `b`, `c` (bindings 0 to 2) and `out`
/// (3): `alone` computes `a * b + c` into `out`; `shared` the same, and the product into
/// `prod` (4) too. The same arithmetic: what a GPU compiler may contract into one rounding
/// (a fused multiply-add) where the product has no other use, and not where it does.
/// Written with `checks` (exact mode's, or the default).
fn fused_or_not(checks: wrela_shader::Checks) -> (Manifest, Vec<Vec<u8>>) {
    use wrela_abi::manifest::BindingStage;
    use wrela_shader::build::{BinOp, Builder, BuiltinInput, Scalar};
    let mut manifest = Manifest::new("game.wasm");
    let mut shaders = Vec::new();
    for (name, shared) in [("alone", false), ("shared", true)] {
        let binding = |b: u32| ResourceBinding {
            binding: b,
            kind: BindingKind::ReadWrite,
            stage: BindingStage::Both,
            format: None,
        };
        manifest.pipelines.push(Pipeline {
            name: name.into(),
            shader: String::new(),
            stage: Stage::Compute { entry: name.into(), workgroup_size: [64, 1, 1] },
            uniform: None,
            bindings: (0..if shared { 5 } else { 4 }).map(binding).collect(),
            debug_flag: None,
            requires: Vec::new(),
            conditional: false,
            standin: None,
            sizes: None,
            kept: 0,
            work: vec![Some(64)],
            unbounded: Vec::new(),
        });
        let mut b = Builder::new();
        let u = b.scalar(Scalar::U32);
        let f = b.scalar(Scalar::F32);
        let u3 = b.vector(Scalar::U32, 3);
        let a = b.storage("a", 0, f, true);
        let bb = b.storage("b", 1, f, true);
        let c = b.storage("c", 2, f, true);
        let out = b.storage("out", 3, f, true);
        let prod = shared.then(|| b.storage("prod", 4, f, true));
        b.compute(name, [64, 1, 1], &[BuiltinInput::GlobalInvocationId], |k| {
            let id = k.input(0, u3);
            let i = k.extract(u, id, 0);
            let x = k.load(f, Builder::element(a, i));
            let y = k.load(f, Builder::element(bb, i));
            let z = k.load(f, Builder::element(c, i));
            let p = k.binary(BinOp::Mul, x, y);
            let r = k.binary(BinOp::Add, p, z);
            k.store(Builder::element(out, i), r);
            if let Some(prod) = prod {
                k.store(Builder::element(prod, i), p);
            }
            k.ret(None);
        });
        let target = wrela_abi::Target::HERE;
        let pl = manifest.pipelines.last_mut().unwrap();
        pl.shader = format!("{name}.{}", wrela_shader::extension(target));
        shaders.push(b.write_with(pl, target, checks).unwrap().bytes);
    }
    (manifest, shaders)
}

/// How many of the two kernels' results differ, and how many of the first's are the CPU's
/// two roundings, with their shaders written with `checks`.
fn arithmetic_differences(checks: wrela_shader::Checks) -> (usize, usize) {
    let (manifest, shaders) = fused_or_not(checks);
    let mut gpu = Gpu::new(&manifest, &shaders, Timing::Off, false).unwrap();
    let n = FUSED_N;
    let bytes = 4 * n;
    // a * b nearly -c: the product's rounding is what's left.
    let mut s: u32 = 12345;
    let mut next = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 8) as f32 / (1u32 << 24) as f32
    };
    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let x = 1.0 + next();
        let y = 1.0 + next();
        a.push(x);
        b.push(y);
        c.push(-(x * y) * (1.0 + 1.0e-7 * next()));
    }
    let le = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    for (h, v) in [(1, &a), (2, &b), (3, &c)] {
        gpu.create_buffer(h, bytes);
        gpu.write_buffer(h, 0, &le(v)).unwrap();
    }
    gpu.create_buffer(4, bytes);
    gpu.create_buffer(5, bytes);
    gpu.create_buffer(6, bytes);
    let bind = |h: u32| Binding::range(h, 0, bytes);
    let groups = Ok([n / 64, 1, 1]);
    gpu.dispatch(0, groups, &[bind(1), bind(2), bind(3), bind(4)], &[]).unwrap();
    gpu.dispatch(1, groups, &[bind(1), bind(2), bind(3), bind(5), bind(6)], &[]).unwrap();
    gpu.end_frame().unwrap();
    let (x, y) = (gpu.read_buffer(4).unwrap(), gpu.read_buffer(5).unwrap());
    let words = |v: &[u8]| {
        v.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect::<Vec<_>>()
    };
    let (x, y) = (words(&x), words(&y));
    let differ = x.iter().zip(&y).filter(|(p, q)| p != q).count();
    // Two roundings, as the CPU computes it: what both give where nothing is fused.
    let two = (0..n as usize).filter(|&i| (a[i] * b[i] + c[i]).to_bits() == x[i]).count();
    gpu.check_errors().unwrap();
    (differ, two)
}

/// In exact mode (`wrela build --exact`, `Checks::exact`) GPU arithmetic doesn't depend on
/// the code around it: `a * b + c` is the same to the bit in a kernel where the product has
/// no other use and in one where it has (so a compiler free to contract would fuse the first
/// into one rounding and not the second), on inputs where one rounding and two differ; and
/// it is the CPU's two roundings. (The default build contracts: on the M4 every one of them
/// differs, which it prints.) `tools/check` runs it on each machine's GPU.
#[test]
#[ignore = "needs a GPU"]
fn arithmetic_is_the_same_whatever_surrounds_it_in_exact_mode() {
    let default = arithmetic_differences(wrela_shader::Checks::CHECKED);
    eprintln!("the default build: {default:?} (differing, the CPU's two roundings)");
    let exact = wrela_shader::Checks { exact: true, ..wrela_shader::Checks::CHECKED };
    let (differ, two) = arithmetic_differences(exact);
    assert_eq!(differ, 0, "the same arithmetic gave different bits in two kernels");
    assert_eq!(two, FUSED_N as usize, "each is the CPU's two roundings");
}

/// The heavy draw's target's side, and its chain of sines a pixel.
const HEAVY_SIDE: u32 = 256;
const HEAVY_CHAIN: u32 = 128;

/// A build of one render pipeline, `heavy`: a triangle over the whole target whose every
/// pixel runs a chain of `HEAVY_CHAIN` sines and adds 1/64 to what's there.
fn heavy_draw() -> (Manifest, [Vec<u8>; 1]) {
    use wrela_abi::manifest::{Output, RenderTarget};
    use wrela_shader::build::{BinOp, Builder, Builtin, BuiltinInput, Scalar};
    let mut manifest = Manifest::new("game.wasm");
    manifest.pipelines.push(Pipeline {
        name: "heavy".into(),
        shader: String::new(),
        stage: Stage::Render {
            vertex_entry: "cover".into(),
            fragment_entry: "heavy".into(),
            blends: vec![Blend::Add],
            cull: Cull::None,
            depth_bias: DepthBias::default(),
            depth: DepthState::default(),
            writes_depth: false,
            outputs: vec![Output::Float],
            targets: vec![RenderTarget::of(&[TextureFormat::Rgba16Float], false)],
        },
        uniform: None,
        bindings: Vec::new(),
        debug_flag: None,
        requires: Vec::new(),
        conditional: false,
        standin: None,
        sizes: None,
        kept: 0,
        work: vec![Some(16), Some(u64::from(HEAVY_CHAIN) * 32 + 64)],
        unbounded: Vec::new(),
    });
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let fl = b.scalar(Scalar::F32);
    let v4 = b.vector(Scalar::F32, 4);
    b.vertex("cover", &[BuiltinInput::VertexIndex], |f| {
        let i = f.input(0, u);
        let one = f.u32(1);
        let xb = f.binary(BinOp::BitAnd, i, one);
        let yb = f.binary(BinOp::Shr, i, one);
        let (four, minus, zero, unit) = (f.f32(4.0), f.f32(1.0), f.f32(0.0), f.f32(1.0));
        let corner = |f: &mut wrela_shader::build::Body<'_>, bit| {
            let v = f.convert(fl, bit, Scalar::F32);
            let v = f.binary(BinOp::Mul, v, four);
            f.binary(BinOp::Sub, v, minus)
        };
        let (x, y) = (corner(f, xb), corner(f, yb));
        let p = f.construct(v4, &[x, y, zero, unit]);
        f.ret_clip(p);
    });
    b.fragment("heavy", &[BuiltinInput::Position], |f| {
        let p = f.input(0, v4);
        let x = f.local("x", fl);
        let n = f.local("n", u);
        let px = f.extract(fl, p, 0);
        f.store(x.clone(), px);
        let zero = f.u32(0);
        f.store(n.clone(), zero);
        f.loop_(|f| {
            let c = f.load(u, n.clone());
            let limit = f.u32(HEAVY_CHAIN);
            let done = f.binary(BinOp::Ge, c, limit);
            f.if_(done, |f| f.break_(), |_| {});
            let v = f.load(fl, x.clone());
            let s = f.builtin(fl, Builtin::Sin, &[v]);
            let a = f.f32(0.999);
            let m = f.binary(BinOp::Mul, s, a);
            let b2 = f.f32(0.001);
            let y = f.binary(BinOp::Add, m, b2);
            f.store(x.clone(), y);
            let one = f.u32(1);
            let c2 = f.binary(BinOp::Add, c, one);
            f.store(n.clone(), c2);
        });
        // 1/64, and the chain's end kept: it's never past 2.
        let r = f.load(fl, x);
        let two = f.f32(2.0);
        let over = f.binary(BinOp::Sub, r, two);
        let none = f.f32(0.0);
        let kept = f.builtin(fl, Builtin::Max, &[over, none]);
        let part = f.f32(1.0 / 64.0);
        let v = f.binary(BinOp::Add, part, kept);
        let out = f.construct(v4, &[v, v, v, v]);
        f.ret(Some(out));
    });
    let target = wrela_abi::Target::HERE;
    let p = &mut manifest.pipelines[0];
    p.shader = format!("heavy.{}", wrela_shader::extension(target));
    let shader = b.write(p, target).unwrap();
    (manifest, [shader.bytes])
}

/// One pass labelled `label` of `draws` heavy draws, each of `instances` triangles over the
/// whole of texture 2, cleared first, then its frame's end, waited for and measured.
fn heavy_frame(gpu: &mut Gpu, label: &str, draws: u32, instances: u32) -> Result<()> {
    use crate::program::Executor;
    gpu.execute(&Command::Label { name: label })?;
    gpu.execute(&Command::BeginPass(Pass::one(2, false, [0.0; 4])))?;
    for _ in 0..draws {
        gpu.execute(&Command::Draw {
            pipeline: 0,
            vertices: 3,
            instances,
            bindings: Vec::new(),
            uniforms: &[],
        })?;
    }
    gpu.execute(&Command::EndPass)?;
    gpu.end_frame()?;
    wait(&gpu.device)?;
    gpu.measure()
}

/// Four passes labelled `label` in one frame, each one draw of `instances` triangles over the
/// whole of texture 2 (kept), then its frame's end, waited for and measured.
fn passes_frame(gpu: &mut Gpu, label: &str, instances: u32) -> Result<()> {
    use crate::program::Executor;
    for _ in 0..4 {
        gpu.execute(&Command::Label { name: label })?;
        gpu.execute(&Command::BeginPass(Pass::one(2, true, [0.0; 4])))?;
        gpu.execute(&Command::Draw {
            pipeline: 0,
            vertices: 3,
            instances,
            bindings: Vec::new(),
            uniforms: &[],
        })?;
        gpu.execute(&Command::EndPass)?;
    }
    gpu.end_frame()?;
    wait(&gpu.device)?;
    gpu.measure()
}

/// The pieces of the heavy pass measured since the last take: how many draws each held, and
/// its time (ms); and the most any other submission took.
fn heavy_jobs(gpu: &mut Gpu, label: &str) -> (Vec<(u32, f64)>, f64) {
    let mut pieces = Vec::new();
    let mut other = 0.0f64;
    for (piece, ms) in std::mem::take(&mut gpu.measured) {
        match piece {
            Some(p) if &*p.label == label => pieces.push((p.draws.len() as u32, ms)),
            _ => other = other.max(ms),
        }
    }
    (pieces, other)
}

/// A pass of 40 heavy draws over the whole target (the shape of 2026-11-19's 65 ms pass of
/// edge rays) never runs as one job: first seen, each draw is a submission of its own; then
/// pieces of a few draws, none over the line, the image the same as one pass would draw;
/// and where one draw alone is over the line, the run ends naming the pass; a split forced
/// (`WRELA_GPU_SPLIT`'s) draws the same image. The line is lowered here to a few of the
/// draws, so nothing runs long. Not bounded, and so not tested: one draw over the line on its
/// first frame (it runs once, then the run ends). `tools/check` runs it on each machine's
/// GPU.
#[test]
#[ignore = "needs a GPU"]
fn a_pass_of_many_heavy_draws_is_cut_into_jobs_under_the_line() {
    const DRAWS: u32 = 40;
    let (manifest, shaders) = heavy_draw();
    let mut gpu = Gpu::new(&manifest, &shaders, Timing::Off, false).unwrap();
    assert!(gpu.device.has_clock(), "this GPU doesn't time its submissions");
    let side = HEAVY_SIDE;
    gpu.create_texture(2, [side, side, 1], TextureFormat::Rgba16Float, false, TextureKind::D2, 1);
    // Each draw about half a millisecond on this GPU: as many triangles over the target as
    // that takes, from the time of one (warmed, under a label of its own).
    for _ in 0..3 {
        heavy_frame(&mut gpu, "warm", 8, 1).unwrap();
    }
    let (warm, _) = heavy_jobs(&mut gpu, "warm");
    let mut one: Vec<f64> = warm.iter().rev().take(8).map(|&(n, ms)| ms / f64::from(n)).collect();
    one.sort_by(f64::total_cmp);
    let instances = (0.5 / one[4]).ceil().clamp(1.0, 32.0) as u32;
    eprintln!("a triangle over the target: {:.3} ms; {instances} a draw", one[4]);
    let heavy = |gpu: &mut Gpu| heavy_frame(gpu, "heavy", DRAWS, instances);
    let jobs = |gpu: &mut Gpu| heavy_jobs(gpu, "heavy");
    let image = |gpu: &mut Gpu| {
        let (w, h, _, raw) = gpu.read_target_raw("heavy").unwrap();
        assert_eq!((w, h), (side, side));
        let want = half_bits((DRAWS * instances) as f32 / 64.0);
        let texels: Vec<u16> =
            raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        assert!(texels.iter().all(|&t| t == want), "every draw adds once");
    };

    // First seen: each draw alone.
    heavy(&mut gpu).unwrap();
    let (first, _) = jobs(&mut gpu);
    assert_eq!(first.len(), DRAWS as usize, "{first:?}");
    assert!(first.iter().all(|&(n, _)| n == 1), "{first:?}");
    image(&mut gpu);
    let most = first.iter().map(|&(_, ms)| ms).fold(0.0, f64::max);
    let whole: f64 = first.iter().map(|&(_, ms)| ms).sum();
    eprintln!("heavy draws: {DRAWS} alone, {whole:.2} ms in all, the most {most:.3} ms");
    assert!(most > 0.02, "a draw took {most} ms: too little to test the cut");

    // Two frames more, so the cost is the warm GPU's (a first draw may take ten times the
    // rest); then a line of 20 draws by it: pieces of 5, the whole pass two lines (fewer
    // pieces where the GPU's clock rises since, never one).
    for _ in 0..2 {
        heavy(&mut gpu).unwrap();
        jobs(&mut gpu);
    }
    let per_draw = gpu.costs[&Arc::<str>::from("heavy")].per_draw();
    gpu.line_ms = 20.0 * per_draw;
    let mut packed = false;
    for _ in 0..6 {
        heavy(&mut gpu).unwrap();
        let (pieces, other) = jobs(&mut gpu);
        eprintln!("pieces of the heavy pass (the line {:.2} ms): {pieces:?}", gpu.line_ms);
        // Never whole; a piece the GPU took slowly makes the next frame's smaller, down to
        // single draws.
        assert!(pieces.len() >= 2, "{pieces:?}");
        packed |= pieces.iter().any(|&(n, _)| n > 1);
        assert_eq!(pieces.iter().map(|&(n, _)| n).sum::<u32>(), DRAWS);
        assert!(pieces.iter().all(|&(_, ms)| ms <= gpu.line_ms), "{pieces:?}");
        assert!(other <= gpu.line_ms, "{other} ms");
        image(&mut gpu);
    }
    assert!(packed, "draws measured are drawn a few to a piece");

    // Passes of whole draws of the size above (about half a millisecond), timed by an eighth
    // of the slower of two passes of eight (a submission's own cost, a tenth of a millisecond
    // on the Radeon, spread over them): a line of 16 of them (at most 12 ms, so nothing here
    // runs long), a heavy pass 0.4 of the line (the M4's clock has moved a draw's time 1.9
    // times between frames; four of them 1.6 lines), a cheap one 1.
    let line = gpu.line_ms;
    for _ in 0..2 {
        passes_frame(&mut gpu, "unit", 8 * instances).unwrap();
    }
    let mut unit: Vec<f64> = std::mem::take(&mut gpu.measured)
        .into_iter()
        .filter(|(p, _)| p.as_ref().is_some_and(|p| &*p.label == "unit"))
        .map(|(_, ms)| ms)
        .collect();
    unit.sort_by(f64::total_cmp);
    let unit = unit[unit.len() - 1] / 8.0;
    gpu.line_ms = (16.0 * unit).min(12.0);
    let big = ((0.4 * gpu.line_ms / unit).round() as u32).max(2) * instances;
    eprintln!("a unit {unit:.3} ms: the line {:.2} ms, a heavy pass {big} triangles", gpu.line_ms);
    // Four heavy passes in one frame: each a job of its own (first seen, and then by what
    // each took), none over the line.
    for _ in 0..3 {
        passes_frame(&mut gpu, "four", big).unwrap();
    }
    let jobs_all = std::mem::take(&mut gpu.measured);
    assert!(jobs_all.iter().all(|(_, ms)| *ms <= gpu.line_ms), "{jobs_all:?}");
    // Four passes cheap for three frames (sharing a submission), then heavy: the frame they
    // grow in holds them in one job over the line (over two: what ended the engine's judged
    // run), once; the run goes on, and every job after it is under the line, each pass
    // measured alone.
    for _ in 0..3 {
        passes_frame(&mut gpu, "grows", instances).unwrap();
    }
    let shared = std::mem::take(&mut gpu.measured);
    assert!(shared.iter().any(|(p, _)| p.is_none()), "cheap passes share a submission");
    passes_frame(&mut gpu, "grows", big).unwrap();
    let grew = std::mem::take(&mut gpu.measured);
    let most = grew.iter().map(|(_, ms)| *ms).fold(0.0, f64::max);
    eprintln!("the frame they grew in: its longest job {most:.2} ms, the line {:.2}", gpu.line_ms);
    for _ in 0..3 {
        passes_frame(&mut gpu, "grows", big).unwrap();
    }
    let after = std::mem::take(&mut gpu.measured);
    assert!(after.iter().all(|(_, ms)| *ms <= gpu.line_ms), "{after:?}");
    gpu.line_ms = line;

    // Forced into seven pieces of uneven draws: the same image.
    gpu.forced = Some((Some("heavy".to_string()), 7));
    heavy(&mut gpu).unwrap();
    let (pieces, _) = jobs(&mut gpu);
    assert_eq!(pieces.iter().map(|&(n, _)| n).collect::<Vec<_>>(), [5, 6, 6, 5, 6, 6, 6]);
    image(&mut gpu);
    gpu.forced = None;

    // A line under one draw: the run ends at the first, naming the pass.
    let (fastest, _) = {
        heavy(&mut gpu).unwrap();
        let (pieces, _) = jobs(&mut gpu);
        let fastest = pieces.iter().map(|&(n, ms)| ms / f64::from(n)).fold(f64::INFINITY, f64::min);
        (fastest, pieces)
    };
    gpu.line_ms = fastest / 2.0;
    let error = heavy(&mut gpu).unwrap_err().to_string();
    assert!(error.contains("the pass `heavy`") && error.contains("over the line"), "{error}");
    assert!(error.contains("draw "), "{error}");
    let (pieces, _) = jobs(&mut gpu);
    assert!(pieces.iter().all(|&(n, _)| n == 1), "{pieces:?}");
}

/// A piece alone over the line once (a wait before it charged to it: a small scissored draw
/// after a long stretch of CPU work, the engine's judge's "59 ms" for 32 px) doesn't end the
/// run; the pass's next piece alone is measured, and only a second time over ends it; one over
/// `RECHECK_LINES` lines ends it at once.
#[test]
#[ignore = "needs a GPU"]
fn a_piece_alone_over_the_line_once_is_measured_again_and_twice_ends_the_run() {
    let (manifest, shaders) = heavy_draw();
    let mut gpu = Gpu::new(&manifest, &shaders, Timing::Off, false).unwrap();
    let line = gpu.line_ms;
    let job = |label: &str, frame: usize| Job {
        index: Default::default(),
        frame,
        call: frame,
        pieces: vec![Piece { label: Arc::from(label), draws: 0..1, of: 1 }],
        alone: true,
        dispatches: false,
        stamp: None,
    };
    // A wait charged once: the run goes on; measured again it takes what it draws.
    assert!(gpu.judge(job("scissored", 1), 1.2 * line).is_ok());
    assert!(gpu.judge(job("scissored", 2), 0.01).is_ok());
    assert!(gpu.judge(job("scissored", 3), 1.2 * line).is_ok(), "a second wait, later");
    // A draw that truly holds the GPU over the line: the second time running ends the run.
    assert!(gpu.judge(job("heavy", 4), 1.5 * line).is_ok());
    let e = gpu.judge(job("heavy", 5), 1.5 * line).unwrap_err().to_string();
    assert!(e.contains("the pass `heavy`") && e.contains("over the line"), "{e}");
    // One far over the line (a runaway): at once.
    let e = gpu.judge(job("runaway", 6), 5.0 * line).unwrap_err().to_string();
    assert!(e.contains("the pass `runaway`"), "{e}");
}

/// `x`'s half-float bits (exact for the small multiples of 1/64 the test draws).
fn half_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = ((b >> 13) & 0x3ff) as u16;
    if x == 0.0 { sign } else { sign | ((exp as u16) << 10) | man }
}

/// A lost device is made again at the end of its frame (both backends; `lose`, what
/// `WRELA_GPU_LOSE_AT` calls): the run carries on, every buffer is there at its handle and
/// size, zeroed, the epoch is one more (`std::gpu::device_epoch`), and work recorded after
/// runs on the new device.
#[test]
#[ignore = "needs a GPU"]
fn a_lost_device_is_made_again_with_its_resources_zeroed() {
    let (manifest, shaders) = increment();
    let mut gpu = Gpu::new(&manifest, &shaders, Timing::Off, false).unwrap();
    gpu.create_buffer(1, 4);
    for _ in 0..3 {
        gpu.dispatch(0, Ok([1, 1, 1]), &[Binding::range(1, 0, 4)], &[]).unwrap();
    }
    gpu.end_frame().unwrap();
    assert_eq!(gpu.read_buffer(1).unwrap(), 3u32.to_le_bytes());
    assert_eq!(gpu.program_limits.epoch, 0);
    gpu.device.lose();
    gpu.dispatch(0, Ok([1, 1, 1]), &[Binding::range(1, 0, 4)], &[]).unwrap();
    gpu.end_frame().unwrap();
    assert_eq!(gpu.program_limits.epoch, 1);
    assert_eq!(gpu.read_buffer(1).unwrap(), 0u32.to_le_bytes(), "zeroed");
    gpu.dispatch(0, Ok([1, 1, 1]), &[Binding::range(1, 0, 4)], &[]).unwrap();
    gpu.end_frame().unwrap();
    assert_eq!(gpu.read_buffer(1).unwrap(), 1u32.to_le_bytes());
    gpu.check_errors().unwrap();
}

#[test]
#[ignore = "needs a GPU"]
fn timestamp_collections_reuse_buffers_and_return_only_current_passes() {
    let (manifest, shaders) = increment();
    for timing in [Timing::Span, Timing::Serial] {
        let mut gpu = Gpu::new(&manifest, &shaders, timing, false).unwrap();
        gpu.create_buffer(1, 4);
        let timer = gpu.timer.as_ref().unwrap();
        let resolved = timer.resolved.clone();
        let readback = timer.readback.clone();
        let mut count = 0u32;
        for (frame, passes) in [3, 1, 5, 2].into_iter().enumerate() {
            gpu.set_frame(frame);
            for _ in 0..passes {
                gpu.dispatch(0, Ok([1, 1, 1]), &[Binding::range(1, 0, 4)], &[]).unwrap();
                count += 1;
            }
            let timings = gpu.take_timings().unwrap();
            assert_eq!(timings.len(), passes);
            for t in timings {
                assert_eq!(t.frame, frame);
                assert_eq!(t.label, "increment");
                assert!(t.nanos.is_finite() && t.nanos >= 0.0 && t.end >= t.start);
            }
            let timer = gpu.timer.as_ref().unwrap();
            assert!(timer.resolved.same(&resolved));
            assert!(timer.readback.same(&readback));
            assert_eq!(gpu.read_buffer(1).unwrap(), count.to_le_bytes());
        }
        assert!(gpu.take_timings().unwrap().is_empty());
        gpu.check_errors().unwrap();
    }
}

/// A program's ask for a label's time (`std::gpu::timed`) is answered once, after a call
/// that ran the label's work, with its microseconds, and times no other label's work (outside
/// a run timed from its start); a build that retires drops its asks, so
/// the next build's request numbers get no stale answer; and a run timed from its start still
/// keeps every timing beside a program's asks.
#[test]
#[ignore = "needs a GPU"]
fn a_timed_ask_is_answered_once_and_a_retired_build_drops_its_asks() {
    use crate::program::Executor;
    let (manifest, shaders) = increment();
    for timing in [Timing::Off, Timing::Span] {
        let mut gpu = Gpu::new(&manifest, &shaders, timing, false).unwrap();
        let run = |gpu: &mut Gpu| {
            gpu.create_buffer(1, 4);
            gpu.execute(&Command::Label { name: "count" }).unwrap();
            gpu.dispatch(0, Ok([1, 1, 1]), &[Binding::range(1, 0, 4)], &[]).unwrap();
            gpu.end_frame().unwrap();
        };
        gpu.ask_time(1, "count").unwrap();
        assert!(gpu.times().is_empty(), "answered before the work ran");
        // Outside a timed run, an ask times its own label's work alone.
        gpu.create_buffer(3, 4);
        gpu.execute(&Command::Label { name: "other" }).unwrap();
        gpu.dispatch(0, Ok([1, 1, 1]), &[Binding::range(3, 0, 4)], &[]).unwrap();
        gpu.end_frame().unwrap();
        let other = gpu.timer.as_ref().unwrap().noted == 1;
        assert_eq!(other, timing == Timing::Span, "an unasked label was timed");
        run(&mut gpu);
        let answered = gpu.times();
        assert_eq!(answered.len(), 1);
        let (request, us) = &answered[0];
        let us = f32::from_le_bytes(us.clone().unwrap().try_into().unwrap());
        assert!(*request == 1 && us.is_finite() && us >= 0.0, "{us}");
        run(&mut gpu);
        assert!(gpu.times().is_empty(), "answered twice");
        if timing == Timing::Span {
            assert_eq!(gpu.take_timings().unwrap().len(), 3, "the profiled run's timings");
        }
        gpu.ask_time(2, "count").unwrap();
        gpu.reload(&manifest, &shaders).unwrap();
        run(&mut gpu);
        assert!(gpu.times().is_empty(), "a retired build's ask was answered");
        gpu.check_errors().unwrap();
    }
}

/// An ask times only its label's work recorded after it, in the stream's order: one made
/// after that work in a call is answered by a later call's.
#[test]
#[ignore = "needs a GPU"]
fn a_timed_ask_times_only_work_recorded_after_it_in_the_same_call() {
    use crate::program::Executor;
    let (manifest, shaders) = increment();
    let mut gpu = Gpu::new(&manifest, &shaders, Timing::Off, false).unwrap();
    gpu.create_buffer(1, 4);
    let requests = |gpu: &mut Gpu| gpu.times().into_iter().map(|(r, _)| r).collect::<Vec<_>>();
    let count = |gpu: &mut Gpu| {
        gpu.execute(&Command::Label { name: "count" }).unwrap();
        gpu.dispatch(0, Ok([1, 1, 1]), &[Binding::range(1, 0, 4)], &[]).unwrap();
    };
    gpu.ask_time(1, "count").unwrap();
    count(&mut gpu);
    gpu.ask_time(2, "count").unwrap();
    gpu.end_frame().unwrap();
    assert_eq!(requests(&mut gpu), [1], "the second ask came after the call's work");
    gpu.end_frame().unwrap();
    assert!(requests(&mut gpu).is_empty(), "a call without the label's work answers none");
    count(&mut gpu);
    gpu.end_frame().unwrap();
    assert_eq!(requests(&mut gpu), [2]);
    gpu.check_errors().unwrap();
}

/// An indexed draw whose instance or index range runs past the last `u32` is the command's
/// error, not a panic when its pass is recorded.
#[test]
#[ignore = "needs a GPU"]
fn an_indexed_draw_past_the_last_instance_fails_its_command() {
    use crate::program::Executor;
    let mut gpu = Gpu::new(&Manifest::new("game.wasm"), &[], Timing::Off, false).unwrap();
    gpu.create_texture(2, [4, 4, 1], TextureFormat::Rgba8, false, TextureKind::D2, 1);
    gpu.execute(&Command::BeginPass(Pass::one(2, false, [0.0; 4]))).unwrap();
    let draw = |instances: u32, first_instance: u32, index_count: u32, first_index: u32| {
        Command::DrawIndexed {
            pipeline: 0,
            indices: 1,
            index_offset: 0,
            index_size: 12,
            index_count,
            instances,
            first_index,
            base_vertex: 0,
            first_instance,
            bindings: Vec::new(),
            uniforms: &[],
        }
    };
    let error = gpu.execute(&draw(2, u32::MAX, 3, 0)).unwrap_err().to_string();
    assert!(error.contains("run past instance"), "{error}");
    let error = gpu.execute(&draw(1, 0, 2, u32::MAX)).unwrap_err().to_string();
    assert!(error.contains("run past index"), "{error}");
    gpu.execute(&Command::EndPass).unwrap();
    gpu.flush().unwrap();
    gpu.check_errors().unwrap();
}

/// A build that retires lets its readback buffers go: they're as large as its largest read.
#[test]
#[ignore = "needs a GPU"]
fn a_retired_build_lets_its_readback_buffers_go() {
    use crate::program::Executor;
    let (manifest, shaders) = increment();
    let mut gpu = Gpu::new(&manifest, &shaders, Timing::Off, false).unwrap();
    gpu.create_buffer(1, 1 << 16);
    gpu.read_range(1, 0, 1 << 16).unwrap();
    assert!(gpu.buffer_readback.is_some());
    gpu.reload(&manifest, &shaders).unwrap();
    assert!(gpu.buffer_readback.is_none() && gpu.image_readback.is_none());
    gpu.check_errors().unwrap();
}

/// Copies a texture before a later write can replace it, leaving the copy in the current
/// encoder so the test checks ordering within and between submissions.
fn snapshot_texture(
    gpu: &mut Gpu,
    handle: u32,
    width: u32,
    height: u32,
    format: TextureFormat,
) -> wrela_gpu::Buffer {
    let row = align_to(
        u64::from(width * format.bytes_per_texel()),
        u64::from(wrela_gpu::COPY_BYTES_PER_ROW_ALIGNMENT),
    ) as u32;
    let buffer = readback_buffer(&gpu.device, u64::from(row) * u64::from(height));
    let mut encoder = gpu.take_encoder();
    encoder.copy_texture_to_buffer(
        gpu.image_copy(handle, 0, 0, 0, 0),
        wrela_gpu::BufferCopy {
            buffer: &buffer,
            offset: 0,
            bytes_per_row: row,
            rows_per_image: height,
        },
        [width, height, 1],
    );
    gpu.encoder = Some(encoder);
    buffer
}

/// A packed texture upload keeps every row and changes only its rectangle. Reads recorded
/// before another write keep the earlier texels, even beside buffer uploads whose lengths
/// don't meet texture-copy alignment.
#[test]
#[ignore = "needs a GPU"]
fn texture_uploads_preserve_rows_rectangles_and_order() {
    let mut gpu = Gpu::new(&Manifest::new("game.wasm"), &[], Timing::Off, false).unwrap();
    gpu.create_buffer(2, 4);
    for format in [
        TextureFormat::Rgba8,
        TextureFormat::Rgba16Float,
        TextureFormat::R16Float,
        TextureFormat::Rg16Float,
        TextureFormat::R32Float,
        TextureFormat::R32Uint,
    ] {
        let texel = format.bytes_per_texel() as usize;
        gpu.create_texture(1, [5, 4, 1], format, false, TextureKind::D2, 1);
        let original: Vec<u8> = (0..5 * 4 * texel).map(|i| (i % 251) as u8).collect();
        gpu.write_buffer(2, 0, &[1, 2, 3, 4]).unwrap();
        gpu.write_texture(1, 0, 0, [0, 0], [5, 4], &original).unwrap();
        let before = snapshot_texture(&mut gpu, 1, 5, 4, format);
        let replacement: Vec<u8> = (0..3 * 2 * texel).map(|i| 255 - i as u8).collect();
        gpu.write_buffer(2, 0, &[5, 6, 7, 8]).unwrap();
        gpu.write_texture(1, 0, 0, [1, 1], [3, 2], &replacement).unwrap();
        gpu.write_texture(1, 0, 0, [0, 0], [0, 4], &[]).unwrap();
        let after = snapshot_texture(&mut gpu, 1, 5, 4, format);
        gpu.flush().unwrap();
        let read = |buffer: &wrela_gpu::Buffer| {
            map_read(&gpu.device, buffer)
                .unwrap()
                .chunks_exact(256)
                .flat_map(|row| row[..5 * texel].iter().copied())
                .collect::<Vec<_>>()
        };
        assert_eq!(read(&before), original, "{format:?}: before overwrite");
        let mut expected = original;
        for row in 0..2 {
            let at = ((row + 1) * 5 + 1) * texel;
            expected[at..at + 3 * texel]
                .copy_from_slice(&replacement[row * 3 * texel..(row + 1) * 3 * texel]);
        }
        assert_eq!(read(&after), expected, "{format:?}: after overwrite");
        assert_eq!(gpu.read_buffer(2).unwrap(), [5, 6, 7, 8]);
        gpu.destroy(1);
    }
    // Padding makes a narrow upload larger than the ring's starting capacity. Growing
    // it must first submit the buffer write already staged there.
    let height = stream::UPLOAD_START / wrela_gpu::COPY_BYTES_PER_ROW_ALIGNMENT + 1;
    gpu.create_texture(1, [1, height, 1], TextureFormat::Rgba8, false, TextureKind::D2, 1);
    let original: Vec<u8> = (0..height * 4).map(|i| (i % 251) as u8).collect();
    gpu.write_buffer(2, 0, &[9, 10, 11, 12]).unwrap();
    gpu.write_texture(1, 0, 0, [0, 0], [1, height], &original).unwrap();
    let snapshot = snapshot_texture(&mut gpu, 1, 1, height, TextureFormat::Rgba8);
    gpu.flush().unwrap();
    let read: Vec<u8> = map_read(&gpu.device, &snapshot)
        .unwrap()
        .chunks_exact(256)
        .flat_map(|row| row[..4].iter().copied())
        .collect();
    assert_eq!(read, original);
    assert_eq!(gpu.read_buffer(2).unwrap(), [9, 10, 11, 12]);
    gpu.check_errors().unwrap();
}

/// A hot reload keeps a pipeline wherever its WGSL file is: a pipeline added before it
/// renumbers its file, not its entry or its code.
#[test]
fn a_pipelines_key_is_its_entry_and_code_not_its_files_name() {
    let p = |shader: &str, entry: &str| Pipeline {
        name: "fill".into(),
        shader: shader.into(),
        stage: Stage::Compute { entry: entry.into(), workgroup_size: [64, 1, 1] },
        uniform: None,
        bindings: Vec::new(),
        debug_flag: None,
        requires: Vec::new(),
        conditional: false,
        standin: None,
        sizes: None,
        kept: 0,
        work: vec![Some(1)],
        unbounded: Vec::new(),
    };
    let key = pipeline_key(&p("pipeline_0.metal", "main"), b"fn main() {}");
    assert_eq!(key, pipeline_key(&p("pipeline_3.metal", "main"), b"fn main() {}"));
    assert_ne!(key, pipeline_key(&p("pipeline_0.metal", "main"), b"fn main() { }"));
    assert_ne!(key, pipeline_key(&p("pipeline_0.metal", "fill"), b"fn main() {}"));
}

#[cfg(test)]
mod target_tests {
    use super::*;

    #[test]
    fn a_float_targets_texels_become_rgba8_clamped() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        let half = |x: u16| x.to_le_bytes();
        let t: Vec<u8> = [half(0x3800), half(0x3c00), half(0x4000), half(0xbc00)].concat();
        assert_eq!(as_rgba8(TextureFormat::Rgba16Float, &t), [128, 255, 255, 0]);
        let grey = 0.25f32.to_le_bytes();
        assert_eq!(as_rgba8(TextureFormat::Depth32Float, &grey), [64, 64, 64, 255]);
        assert_eq!(as_rgba8(TextureFormat::Rg16Float, &t[..4]), [128, 255, 0, 255]);
    }
}
