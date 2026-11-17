//! The device end to end on the machine's own GPU: a kernel the compiler's writers wrote, run on
//! a buffer and read back. Run with `cargo test -p wrela-gpu -- --ignored` (under the machine's
//! GPU lock: tools/gpulock.py).

use wrela_abi::manifest::{
    BindingKind, BindingStage, BufferSizes, Pipeline, ResourceBinding, Stage,
};
use wrela_gpu::*;
use wrela_ir::{BinOp, BuiltinInput, Scalar};
use wrela_shader::build::Builder;

/// `out[i] = i * 3 + 1` for each invocation `i`, 64 to a workgroup.
fn kernel() -> Builder {
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let u3 = b.vector(Scalar::U32, 3);
    let out = b.storage("out", 0, u, true);
    b.compute("triple", [64, 1, 1], &[BuiltinInput::GlobalInvocationId], |f| {
        let id = f.input(0, u3);
        let i = f.extract(u, id, 0);
        let three = f.u32(3);
        let one = f.u32(1);
        let t = f.binary(BinOp::Mul, i, three);
        let v = f.binary(BinOp::Add, t, one);
        f.store(Builder::element(out, i), v);
    });
    b
}

fn pipeline() -> Pipeline {
    Pipeline {
        name: "triple".into(),
        shader: "triple".into(),
        stage: Stage::Compute { entry: "triple".into(), workgroup_size: [64, 1, 1] },
        uniform: None,
        bindings: vec![ResourceBinding {
            binding: 0,
            kind: BindingKind::ReadWrite,
            stage: BindingStage::Both,
            format: None,
        }],
        debug_flag: None,
        requires: Vec::new(),
        conditional: false,
        standin: None,
        sizes: None,
        kept: 0,
        work: Vec::new(),
        unbounded: Vec::new(),
    }
}

#[test]
#[ignore = "needs a GPU"]
fn a_kernel_writes_a_buffer_the_cpu_reads_back() {
    let device = Device::open(&DeviceOptions { label: "test".into(), ..Default::default() })
        .expect("a device");
    let mut p = pipeline();
    let shader = kernel().write(&mut p, wrela_abi::Target::HERE).expect("the shader");
    let Stage::Compute { entry, .. } = &p.stage else { unreachable!() };
    let n = 256u32;
    let out =
        device.create_buffer(&BufferDesc { label: "out", size: u64::from(n) * 4, mappable: false });
    let read =
        device.create_buffer(&BufferDesc { label: "read", size: u64::from(n) * 4, mappable: true });
    // The layout: the storage buffer, and where Metal's shader reads buffers' sizes.
    let mut entries = vec![LayoutEntry {
        binding: 0,
        visibility: Stages::COMPUTE,
        ty: BindingType::Storage { read_only: false },
    }];
    let sizes_buffer = p.sizes.as_ref().map(|s: &BufferSizes| {
        entries.push(LayoutEntry {
            binding: s.binding,
            visibility: Stages::COMPUTE,
            ty: BindingType::Uniform { dynamic: false, size: 16 },
        });
        let b = device.create_buffer(&BufferDesc { label: "sizes", size: 16, mappable: false });
        let mut words = [0u32; 4];
        words[0] = n * 4;
        device.write_buffer(&b, 0, &words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>());
        (s.binding, b)
    });
    let layout = device.create_bind_group_layout(&entries);
    let mut group = vec![BindGroupEntry {
        binding: 0,
        resource: Resource::Buffer { buffer: &out, offset: 0, size: None },
    }];
    if let Some((binding, b)) = &sizes_buffer {
        group.push(BindGroupEntry {
            binding: *binding,
            resource: Resource::Buffer { buffer: b, offset: 0, size: None },
        });
    }
    let group = device.create_bind_group(&layout, &group);
    let module = device.create_shader_module("triple", &shader.bytes).expect("a module");
    let pipeline = device
        .create_compute_pipeline(&ComputePipelineDesc {
            label: "triple",
            layout: &layout,
            module: &module,
            entry,
            workgroup_size: [64, 1, 1],
        })
        .expect("a pipeline");
    let mut enc = device.create_command_encoder("triple");
    {
        let mut pass = enc.begin_compute_pass("triple", None);
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(&group, &[]);
        pass.dispatch(n / 64, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &read, 0, u64::from(n) * 4);
    let done = device.submit(vec![enc.finish()]);
    device.wait(done);
    assert!(device.take_errors().is_empty(), "{:?}", device.take_errors());
    let bytes = read.read(0, u64::from(n) * 4).expect("read back");
    let words: Vec<u32> =
        bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let want: Vec<u32> = (0..n).map(|i| i * 3 + 1).collect();
    assert_eq!(words, want);
}
