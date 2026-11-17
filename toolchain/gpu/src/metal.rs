//! The Metal backend (objc2-metal): Metal 3 as the floor, the M4's Metal as measured.
//!
//! - Resources are tracked (Metal's default): Metal orders a read after a write across
//!   encoders and command buffers, so nothing here inserts a barrier.
//! - Buffers the GPU alone uses are private; a mappable one (a read back) is shared, read in
//!   place once its submission is done. Queue writes go through a shared buffer of their own,
//!   copied by a blit encoder of a command buffer committed before the submission's own, as
//!   wgpu's did: what earlier submissions read keeps its contents.
//! - Every buffer and texture starts zeroed (WebGPU's promise, which programs keep to).
//! - Each pass is an encoder: a compute pass a compute encoder, a render pass a render encoder,
//!   copies a blit encoder; timestamps are counter samples at the encoder's boundaries (Apple's
//!   GPUs sample there), in nanoseconds.
//! - A command buffer's error names the encoder it happened in (`encoderExecutionStatus`), whose
//!   label is the pass's.

use crate::*;
use block2::RcBlock;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSError, NSRange, NSString, NSUInteger};
use objc2_metal::*;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

type Id<T> = Retained<ProtocolObject<T>>;

fn ns(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

fn error_text(e: &NSError) -> String {
    e.localizedDescription().to_string()
}

// ---- the device -------------------------------------------------------------------------------------

struct Shared {
    raw: Id<dyn MTLDevice>,
    queue: Id<dyn MTLCommandQueue>,
    info: AdapterInfo,
    limits: Limits,
    features: Features,
    /// The timestamp counter set, where the device samples at stage boundaries.
    timestamps: Option<Id<dyn MTLCounterSet>>,
    submitted: AtomicU64,
    completed: Arc<AtomicU64>,
    /// Command buffers committed and not yet seen complete, by submission.
    in_flight: Mutex<Vec<(SubmissionIndex, Id<dyn MTLCommandBuffer>)>>,
    /// Staged writes, copied before the next submission's own work.
    pending: Mutex<Vec<PendingWrite>>,
    /// Errors the GPU reported: a failed command buffer, naming its encoder.
    errors: Arc<Mutex<Vec<String>>>,
    /// A buffer of zeros, shared, as large as the largest texture level zeroed so far.
    zeros: Mutex<Option<Id<dyn MTLBuffer>>>,
}

// SAFETY: Metal's objects are thread-safe (MTLDevice, MTLCommandQueue and resources may be used
// from any thread); the host records on one thread at a time.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

enum PendingWrite {
    Buffer {
        staging: Id<dyn MTLBuffer>,
        dst: Id<dyn MTLBuffer>,
        offset: u64,
        len: u64,
    },
    Fill {
        dst: Id<dyn MTLBuffer>,
        len: u64,
    },
    Texture {
        staging: Id<dyn MTLBuffer>,
        staging_offset: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
        dst: Id<dyn MTLTexture>,
        level: u32,
        slice: u32,
        origin: [u32; 3],
        size: [u32; 3],
    },
}

/// The GPU: Metal's device and its one command queue.
#[derive(Clone)]
pub struct Device(Arc<Shared>);

impl Device {
    /// The machine's hardware GPU (or none: an error naming what was found).
    pub fn open(options: &DeviceOptions) -> Result<Device> {
        autoreleasepool(|_| {
            let raw = MTLCreateSystemDefaultDevice()
                .ok_or_else(|| Error("no hardware GPU through Metal: found none".into()))?;
            let name = raw.name().to_string();
            let queue = raw
                .newCommandQueueWithMaxCommandBufferCount(2048)
                .ok_or("Metal made no command queue")?;
            queue.setLabel(Some(&ns(&options.label)));
            let apple = raw.supportsFamily(MTLGPUFamily::Apple7);
            let info = AdapterInfo {
                name: name.clone(),
                backend: Backend::Metal,
                device_type: if raw.isLowPower() || raw.hasUnifiedMemory() {
                    DeviceType::IntegratedGpu
                } else {
                    DeviceType::DiscreteGpu
                },
                vendor: 0,
                device: 0,
                driver: String::new(),
                driver_info: String::new(),
                subgroup_min_size: if apple { 4 } else { 4 },
                subgroup_max_size: 64,
            };
            let limits = Limits {
                max_texture_dimension_2d: 16384,
                max_texture_dimension_3d: 2048,
                max_texture_array_layers: 2048,
                max_buffer_size: raw.maxBufferLength() as u64,
                // The largest a binding's offset and size may say (WebGPU caps it a word short).
                max_storage_buffer_binding_size: (raw.maxBufferLength() as u64)
                    .min(u64::from(u32::MAX))
                    & !3,
                max_uniform_buffer_binding_size: (raw.maxBufferLength() as u64)
                    .min(u64::from(u32::MAX)),
                // 31 buffer slots a stage, less the two the binding model sets aside (its sizes
                // and the immediates wgpu reserved): what the compiler counts against.
                max_storage_buffers_per_shader_stage: 29,
                max_sampled_textures_per_shader_stage: 96,
                max_samplers_per_shader_stage: 16,
                max_storage_textures_per_shader_stage: 32,
                max_color_attachments: 8,
                max_color_attachment_bytes_per_sample: 64,
                min_uniform_buffer_offset_alignment: 256,
                min_storage_buffer_offset_alignment: 32,
                max_compute_workgroup_storage_size: raw.maxThreadgroupMemoryLength() as u32,
                max_compute_invocations_per_workgroup: 1024,
                max_compute_workgroup_size: {
                    let m = raw.maxThreadsPerThreadgroup();
                    [m.width as u32, m.height as u32, m.depth as u32]
                },
                max_compute_workgroups_per_dimension: 65535,
            };
            // WebGPU's optional features this GPU has (the M4 has every one the host asks for).
            let mut features = vec![
                "shader-f16",
                "subgroups",
                "float32-filterable",
                "float32-blendable",
                "rg11b10ufloat-renderable",
                "bgra8unorm-storage",
                "indirect-first-instance",
                "chromium-experimental-multi-draw-indirect",
            ];
            if raw.supportsBCTextureCompression() {
                features.push("texture-compression-bc");
            }
            if raw.supportsFamily(MTLGPUFamily::Apple2) {
                features.push("texture-compression-etc2");
                features.push("texture-compression-astc");
            }
            let timestamps = timestamp_counters(&raw);
            if timestamps.is_some() {
                features.push("timestamp-query");
            }
            let features = match &options.features {
                Some(keep) => {
                    Features(features.into_iter().filter(|f| keep.iter().any(|k| k == f)).collect())
                }
                None => Features(features),
            };
            Ok(Device(Arc::new(Shared {
                raw,
                queue,
                info,
                limits,
                features,
                timestamps,
                submitted: AtomicU64::new(0),
                completed: Arc::new(AtomicU64::new(0)),
                in_flight: Mutex::new(Vec::new()),
                pending: Mutex::new(Vec::new()),
                errors: Arc::new(Mutex::new(Vec::new())),
                zeros: Mutex::new(None),
            })))
        })
    }

    /// The Metal device, for what reads it directly (a capture, a pipeline's report).
    pub fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.0.raw
    }

    pub fn info(&self) -> &AdapterInfo {
        &self.0.info
    }

    pub fn limits(&self) -> &Limits {
        &self.0.limits
    }

    pub fn features(&self) -> &Features {
        &self.0.features
    }

    /// Nanoseconds a timestamp tick is: Apple's GPUs count nanoseconds.
    pub fn timestamp_period(&self) -> f32 {
        1.0
    }

    /// What the device has said of itself, a line each, for `--adapter`.
    pub fn report(&self) -> String {
        let d = &self.0.raw;
        let mut s = String::new();
        let families = [
            ("Apple7", MTLGPUFamily::Apple7),
            ("Apple8", MTLGPUFamily::Apple8),
            ("Apple9", MTLGPUFamily::Apple9),
            ("Metal3", MTLGPUFamily::Metal3),
        ];
        let have: Vec<&str> =
            families.iter().filter(|(_, f)| d.supportsFamily(*f)).map(|(n, _)| *n).collect();
        s.push_str(&format!("families {}\n", have.join(" ")));
        s.push_str(&format!("unified memory {}\n", d.hasUnifiedMemory()));
        s.push_str(&format!("max buffer length {}\n", d.maxBufferLength()));
        s.push_str(&format!("max threadgroup memory {}\n", d.maxThreadgroupMemoryLength()));
        s.push_str(&format!("recommended working set {}\n", d.recommendedMaxWorkingSetSize()));
        s
    }

    fn take_pending(&self) -> Vec<PendingWrite> {
        std::mem::take(&mut *lock(&self.0.pending))
    }

    pub fn create_buffer(&self, desc: &BufferDesc<'_>) -> Buffer {
        let size = desc.size.max(4);
        let options = if desc.mappable {
            MTLResourceOptions::StorageModeShared
        } else {
            MTLResourceOptions::StorageModePrivate
        };
        let raw = self
            .0
            .raw
            .newBufferWithLength_options(size as NSUInteger, options)
            .expect("Metal made no buffer");
        raw.setLabel(Some(&ns(desc.label)));
        if !desc.mappable {
            lock(&self.0.pending).push(PendingWrite::Fill { dst: raw.clone(), len: size });
        }
        Buffer { raw, size: desc.size, mappable: desc.mappable }
    }

    pub fn create_texture(&self, desc: &TextureDesc<'_>) -> Texture {
        let d = MTLTextureDescriptor::new();
        let layers = desc.depth_or_layers.max(1);
        let ty = match desc.dimension {
            TextureDimension::D3 => MTLTextureType::Type3D,
            TextureDimension::D2 if layers > 1 => MTLTextureType::Type2DArray,
            TextureDimension::D2 => MTLTextureType::Type2D,
        };
        d.setTextureType(ty);
        d.setPixelFormat(pixel_format(desc.format));
        unsafe {
            d.setWidth(desc.width as NSUInteger);
            d.setHeight(desc.height as NSUInteger);
            match desc.dimension {
                TextureDimension::D3 => d.setDepth(layers as NSUInteger),
                TextureDimension::D2 => d.setArrayLength(layers as NSUInteger),
            }
            d.setMipmapLevelCount(desc.levels.max(1) as NSUInteger);
        }
        d.setStorageMode(MTLStorageMode::Private);
        let mut usage = MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView;
        if desc.usage.storage {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        if desc.usage.render {
            usage |= MTLTextureUsage::RenderTarget;
        }
        d.setUsage(usage);
        let raw = self.0.raw.newTextureWithDescriptor(&d).expect("Metal made no texture");
        raw.setLabel(Some(&ns(desc.label)));
        let t = Texture {
            raw,
            format: desc.format,
            width: desc.width,
            height: desc.height,
            depth_or_layers: layers,
            levels: desc.levels.max(1),
            dimension: desc.dimension,
        };
        self.zero_texture(&t);
        t
    }

    /// Every level and layer of `t` zeroed, before the next submission.
    fn zero_texture(&self, t: &Texture) {
        let f = t.format;
        let block = f.block();
        let (layers, depth) = match t.dimension {
            TextureDimension::D3 => (1, t.depth_or_layers),
            TextureDimension::D2 => (t.depth_or_layers, 1),
        };
        for level in 0..t.levels {
            let w = (t.width >> level).max(1);
            let h = (t.height >> level).max(1);
            let d = match t.dimension {
                TextureDimension::D3 => (depth >> level).max(1),
                TextureDimension::D2 => 1,
            };
            let row = f.bytes_of(w, 1);
            let rows = u64::from(h.div_ceil(block));
            let image = row * rows;
            let total = image * u64::from(d);
            let zeros = self.zeros(total);
            for slice in 0..layers {
                lock(&self.0.pending).push(PendingWrite::Texture {
                    staging: zeros.clone(),
                    staging_offset: 0,
                    bytes_per_row: row,
                    bytes_per_image: image,
                    dst: t.raw.clone(),
                    level,
                    slice,
                    origin: [0, 0, 0],
                    size: [w.div_ceil(block) * block, h.div_ceil(block) * block, d],
                });
            }
        }
    }

    /// A shared buffer of at least `len` zeros.
    fn zeros(&self, len: u64) -> Id<dyn MTLBuffer> {
        let mut z = lock(&self.0.zeros);
        if let Some(b) = z.as_ref().filter(|b| b.length() as u64 >= len) {
            return b.clone();
        }
        let b = self
            .0
            .raw
            .newBufferWithLength_options(
                len.max(256) as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("Metal made no buffer");
        *z = Some(b.clone());
        b
    }

    pub fn create_sampler(&self, desc: &SamplerDesc) -> Sampler {
        let d = MTLSamplerDescriptor::new();
        let filter = if desc.linear {
            MTLSamplerMinMagFilter::Linear
        } else {
            MTLSamplerMinMagFilter::Nearest
        };
        d.setMinFilter(filter);
        d.setMagFilter(filter);
        d.setMipFilter(if desc.mip_linear {
            MTLSamplerMipFilter::Linear
        } else {
            MTLSamplerMipFilter::Nearest
        });
        let address = if desc.repeat {
            MTLSamplerAddressMode::Repeat
        } else {
            MTLSamplerAddressMode::ClampToEdge
        };
        d.setSAddressMode(address);
        d.setTAddressMode(address);
        d.setRAddressMode(address);
        d.setMaxAnisotropy(desc.anisotropy.clamp(1, 16) as NSUInteger);
        d.setLodMinClamp(0.0);
        d.setLodMaxClamp(32.0);
        if let Some(c) = desc.compare {
            d.setCompareFunction(compare_function(c));
        }
        d.setSupportArgumentBuffers(true);
        let raw = self.0.raw.newSamplerStateWithDescriptor(&d).expect("Metal made no sampler");
        Sampler { raw }
    }

    pub fn create_query_set(&self, count: u32) -> Result<QuerySet> {
        let set = self.0.timestamps.as_ref().ok_or("this GPU samples no timestamps")?;
        let d = MTLCounterSampleBufferDescriptor::new();
        d.setCounterSet(Some(set));
        d.setStorageMode(MTLStorageMode::Shared);
        unsafe {
            d.setSampleCount(count as NSUInteger);
        }
        d.setLabel(&ns("timestamps"));
        let raw = self.0.raw.newCounterSampleBufferWithDescriptor_error(&d).map_err(|e| {
            Error(format!("Metal made no counter sample buffer: {}", error_text(&e)))
        })?;
        Ok(QuerySet { raw, count })
    }

    pub fn create_bind_group_layout(&self, entries: &[LayoutEntry]) -> BindGroupLayout {
        BindGroupLayout(Arc::new(Layout::of(entries)))
    }

    pub fn create_bind_group(
        &self,
        layout: &BindGroupLayout,
        entries: &[BindGroupEntry<'_>],
    ) -> BindGroup {
        let mut out = Vec::with_capacity(entries.len());
        for e in entries {
            let Some(k) = layout.0.entries.iter().position(|l| l.binding == e.binding) else {
                continue;
            };
            let bound = match &e.resource {
                Resource::Buffer { buffer, offset, .. } => {
                    Bound::Buffer(buffer.raw.clone(), *offset)
                }
                Resource::View(v) => Bound::Texture(v.raw.clone()),
                Resource::Sampler(s) => Bound::Sampler(s.raw.clone()),
            };
            out.push((k, bound));
        }
        BindGroup(Arc::new(Group { layout: layout.0.clone(), bound: out }))
    }

    /// A shader's library: its MSL compiled as a passthrough shader was (`MTLCompileOptions`'
    /// defaults: Metal's newest language, fast math, which the text states besides).
    pub fn create_shader_module(&self, label: &str, source: &[u8]) -> Result<ShaderModule> {
        let text = std::str::from_utf8(source).map_err(|_| Error("its MSL isn't UTF-8".into()))?;
        autoreleasepool(|_| {
            let options = MTLCompileOptions::new();
            let library = self
                .0
                .raw
                .newLibraryWithSource_options_error(&ns(text), Some(&options))
                .map_err(|e| Error(format!("MSL: {}", error_text(&e))))?;
            library.setLabel(Some(&ns(label)));
            Ok(ShaderModule { library })
        })
    }

    fn function(&self, module: &ShaderModule, entry: &str) -> Result<Id<dyn MTLFunction>> {
        module
            .library
            .newFunctionWithName(&ns(entry))
            .ok_or_else(|| Error(format!("the shader has no entry point `{entry}`")))
    }

    pub fn create_compute_pipeline(
        &self,
        desc: &ComputePipelineDesc<'_>,
    ) -> Result<ComputePipeline> {
        autoreleasepool(|_| {
            let function = self.function(desc.module, desc.entry)?;
            let d = MTLComputePipelineDescriptor::new();
            d.setComputeFunction(Some(&function));
            d.setLabel(Some(&ns(desc.label)));
            let raw = self
                .0
                .raw
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &d,
                    MTLPipelineOption::empty(),
                    None,
                )
                .map_err(|e| Error(format!("new compute pipeline state: {}", error_text(&e))))?;
            let [x, y, z] = desc.workgroup_size;
            Ok(ComputePipeline {
                raw,
                workgroup: MTLSize {
                    width: x as NSUInteger,
                    height: y as NSUInteger,
                    depth: z as NSUInteger,
                },
            })
        })
    }

    pub fn create_render_pipeline(&self, desc: &RenderPipelineDesc<'_>) -> Result<RenderPipeline> {
        autoreleasepool(|_| {
            let vertex = self.function(desc.module, desc.vertex_entry)?;
            let fragment = self.function(desc.module, desc.fragment_entry)?;
            let d = MTLRenderPipelineDescriptor::new();
            d.setVertexFunction(Some(&vertex));
            d.setFragmentFunction(Some(&fragment));
            d.setLabel(Some(&ns(desc.label)));
            unsafe {
                d.setInputPrimitiveTopology(MTLPrimitiveTopologyClass::Triangle);
                d.setRasterSampleCount(1);
            }
            let attachments = d.colorAttachments();
            for (k, c) in desc.colors.iter().enumerate() {
                let a = unsafe { attachments.objectAtIndexedSubscript(k as NSUInteger) };
                a.setPixelFormat(pixel_format(c.format));
                a.setWriteMask(MTLColorWriteMask::All);
                if let Some((color, alpha)) = c.blend.components() {
                    a.setBlendingEnabled(true);
                    a.setSourceRGBBlendFactor(blend_factor(color.0));
                    a.setDestinationRGBBlendFactor(blend_factor(color.1));
                    a.setRgbBlendOperation(blend_operation(color.2));
                    a.setSourceAlphaBlendFactor(blend_factor(alpha.0));
                    a.setDestinationAlphaBlendFactor(blend_factor(alpha.1));
                    a.setAlphaBlendOperation(blend_operation(alpha.2));
                }
            }
            let depth_state = match &desc.depth {
                Some(dd) => {
                    d.setDepthAttachmentPixelFormat(pixel_format(dd.format));
                    let s = MTLDepthStencilDescriptor::new();
                    s.setDepthCompareFunction(compare_function(dd.state.compare));
                    s.setDepthWriteEnabled(dd.state.write);
                    Some(
                        self.0
                            .raw
                            .newDepthStencilStateWithDescriptor(&s)
                            .ok_or("Metal made no depth state")?,
                    )
                }
                None => None,
            };
            let raw = self
                .0
                .raw
                .newRenderPipelineStateWithDescriptor_error(&d)
                .map_err(|e| Error(format!("new render pipeline state: {}", error_text(&e))))?;
            Ok(RenderPipeline {
                raw,
                depth_state,
                cull: match desc.cull {
                    Cull::None => MTLCullMode::None,
                    Cull::Front => MTLCullMode::Front,
                    Cull::Back => MTLCullMode::Back,
                },
                bias: desc.depth.map(|d| d.bias).unwrap_or_default(),
            })
        })
    }

    pub fn create_command_encoder(&self, label: &str) -> CommandEncoder {
        CommandEncoder { device: self.clone(), label: label.to_string(), cb: None, blit: None }
    }

    /// A command buffer of the queue's, reporting the encoder an error happened in.
    fn command_buffer(&self) -> Id<dyn MTLCommandBuffer> {
        let d = MTLCommandBufferDescriptor::new();
        d.setErrorOptions(MTLCommandBufferErrorOption::EncoderExecutionStatus);
        self.0.queue.commandBufferWithDescriptor(&d).expect("Metal made no command buffer")
    }

    /// Writes `data` at `offset` of `buffer` before the next submission's own work.
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let staging = self.staging(data);
        lock(&self.0.pending).push(PendingWrite::Buffer {
            staging,
            dst: buffer.raw.clone(),
            offset,
            len: data.len() as u64,
        });
    }

    /// Writes texels to a level and layer of `texture` before the next submission's own work:
    /// `data` holds `rows_per_image` rows (of blocks) of `bytes_per_row` each, a layer.
    pub fn write_texture(
        &self,
        at: TextureCopy<'_>,
        data: &[u8],
        bytes_per_row: u32,
        rows_per_image: u32,
        extent: [u32; 3],
    ) {
        if data.is_empty() {
            return;
        }
        let staging = self.staging(data);
        let (slice, z) = match at.texture.dimension {
            TextureDimension::D3 => (0, at.origin[2]),
            TextureDimension::D2 => (at.origin[2], 0),
        };
        lock(&self.0.pending).push(PendingWrite::Texture {
            staging,
            staging_offset: 0,
            bytes_per_row: u64::from(bytes_per_row),
            bytes_per_image: u64::from(bytes_per_row) * u64::from(rows_per_image),
            dst: at.texture.raw.clone(),
            level: at.level,
            slice,
            origin: [at.origin[0], at.origin[1], z],
            size: extent,
        });
    }

    fn staging(&self, data: &[u8]) -> Id<dyn MTLBuffer> {
        unsafe {
            self.0
                .raw
                .newBufferWithBytes_length_options(
                    NonNull::new(data.as_ptr() as *mut _).expect("data"),
                    data.len() as NSUInteger,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("Metal made no buffer")
        }
    }

    /// Commits the staged writes, then `buffers`, in order: their submission's index.
    pub fn submit(&self, buffers: Vec<CommandBuffer>) -> SubmissionIndex {
        autoreleasepool(|_| {
            let index = self.0.submitted.fetch_add(1, Ordering::AcqRel) + 1;
            let pending = self.take_pending();
            let mut cbs: Vec<Id<dyn MTLCommandBuffer>> = Vec::new();
            if !pending.is_empty() {
                let cb = self.command_buffer();
                cb.setLabel(Some(&ns("queue writes")));
                let blit = cb.blitCommandEncoder().expect("Metal made no blit encoder");
                blit.setLabel(Some(&ns("queue writes")));
                for w in pending {
                    encode_write(&blit, &w);
                }
                blit.endEncoding();
                cbs.push(cb);
            }
            cbs.extend(buffers.into_iter().map(|b| b.raw));
            if cbs.is_empty() {
                // Nothing to run: a command buffer of its own marks the submission done.
                let cb = self.command_buffer();
                cbs.push(cb);
            }
            let last = cbs.len() - 1;
            for (k, cb) in cbs.iter().enumerate() {
                let errors = self.0.errors.clone();
                let completed = self.0.completed.clone();
                let is_last = k == last;
                let block =
                    RcBlock::new(move |cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                        let cb = unsafe { cb.as_ref() };
                        if cb.status() == MTLCommandBufferStatus::Error {
                            lock(&errors).push(command_buffer_error(cb));
                        }
                        if is_last {
                            completed.fetch_max(index, Ordering::AcqRel);
                        }
                    });
                unsafe {
                    cb.addCompletedHandler(RcBlock::as_ptr(&block));
                }
                cb.commit();
            }
            lock(&self.0.in_flight).push((index, cbs[last].clone()));
            index
        })
    }

    /// The submissions done so far: every one up to this index.
    pub fn completed(&self) -> SubmissionIndex {
        self.0.completed.load(Ordering::Acquire)
    }

    pub fn submitted(&self) -> SubmissionIndex {
        self.0.submitted.load(Ordering::Acquire)
    }

    /// Waits for submission `index` (and every one before it) to be done.
    pub fn wait(&self, index: SubmissionIndex) {
        let waiting: Vec<Id<dyn MTLCommandBuffer>> = {
            let mut f = lock(&self.0.in_flight);
            let w = f.iter().filter(|(i, _)| *i <= index).map(|(_, cb)| cb.clone()).collect();
            f.retain(|(i, _)| *i > index);
            w
        };
        for cb in waiting {
            cb.waitUntilCompleted();
        }
        // The handlers run on another thread, after `waitUntilCompleted` returns.
        while self.completed() < index.min(self.submitted()) {
            std::thread::yield_now();
        }
    }

    /// Submits the staged writes, if any, and waits for every submission to be done.
    pub fn wait_idle(&self) {
        if !lock(&self.0.pending).is_empty() {
            self.submit(Vec::new());
        }
        self.wait(self.submitted());
    }

    /// The errors the GPU reported since the last take.
    pub fn take_errors(&self) -> Vec<String> {
        std::mem::take(&mut *lock(&self.0.errors))
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What a failed command buffer says: its error, and the encoders that faulted (their labels
/// are the passes').
fn command_buffer_error(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> String {
    let label = cb.label().map(|l| l.to_string()).unwrap_or_default();
    let what = cb.error().map(|e| error_text(&e)).unwrap_or_else(|| "an error".into());
    let mut out = format!("the GPU failed command buffer `{label}`: {what}");
    if let Some(e) = cb.error() {
        let info = e.userInfo();
        let key = unsafe { MTLCommandBufferEncoderInfoErrorKey };
        if let Some(list) = info.objectForKey(key) {
            if let Ok(list) = list.downcast::<NSArray>() {
                for item in list.iter() {
                    let item: Retained<ProtocolObject<dyn MTLCommandBufferEncoderInfo>> =
                        unsafe { Retained::cast_unchecked(item) };
                    if item.errorState() == MTLCommandEncoderErrorState::Faulted {
                        out.push_str(&format!("; it faulted in `{}`", item.label()));
                    }
                }
            }
        }
    }
    out
}

fn encode_write(blit: &ProtocolObject<dyn MTLBlitCommandEncoder>, w: &PendingWrite) {
    unsafe {
        match w {
            PendingWrite::Buffer { staging, dst, offset, len } => {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    staging,
                    0,
                    dst,
                    *offset as NSUInteger,
                    *len as NSUInteger,
                );
            }
            PendingWrite::Fill { dst, len } => {
                blit.fillBuffer_range_value(dst, NSRange::new(0, *len as NSUInteger), 0);
            }
            PendingWrite::Texture {
                staging,
                staging_offset,
                bytes_per_row,
                bytes_per_image,
                dst,
                level,
                slice,
                origin,
                size,
            } => {
                blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    staging,
                    *staging_offset as NSUInteger,
                    *bytes_per_row as NSUInteger,
                    *bytes_per_image as NSUInteger,
                    MTLSize { width: size[0] as NSUInteger, height: size[1] as NSUInteger, depth: size[2] as NSUInteger },
                    dst,
                    *slice as NSUInteger,
                    *level as NSUInteger,
                    MTLOrigin { x: origin[0] as NSUInteger, y: origin[1] as NSUInteger, z: origin[2] as NSUInteger },
                );
            }
        }
    }
}

/// The device's timestamp counter set, where it samples at stage boundaries.
fn timestamp_counters(d: &ProtocolObject<dyn MTLDevice>) -> Option<Id<dyn MTLCounterSet>> {
    if !d.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
        return None;
    }
    let sets = d.counterSets()?;
    let common = unsafe { MTLCommonCounterSetTimestamp };
    sets.iter().find(|s| s.name().isEqualToString(common))
}

// ---- resources ----------------------------------------------------------------------------------------

pub struct Buffer {
    raw: Id<dyn MTLBuffer>,
    size: u64,
    mappable: bool,
}

impl Buffer {
    pub fn size(&self) -> u64 {
        self.size
    }

    /// A mappable buffer's bytes `offset..offset + len`, read in place: its submission must be
    /// done.
    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        if !self.mappable {
            return Err("the buffer isn't mappable".into());
        }
        if offset + len > self.raw.length() as u64 {
            return Err("a read past the buffer's end".into());
        }
        let p = self.raw.contents().as_ptr() as *const u8;
        Ok(unsafe { std::slice::from_raw_parts(p.add(offset as usize), len as usize) }.to_vec())
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Buffer {
        Buffer { raw: self.raw.clone(), size: self.size, mappable: self.mappable }
    }
}

pub struct Texture {
    raw: Id<dyn MTLTexture>,
    format: TextureFormat,
    width: u32,
    height: u32,
    depth_or_layers: u32,
    levels: u32,
    dimension: TextureDimension,
}

impl Texture {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn format(&self) -> TextureFormat {
        self.format
    }

    pub fn levels(&self) -> u32 {
        self.levels
    }

    pub fn depth_or_layers(&self) -> u32 {
        self.depth_or_layers
    }

    /// The Metal texture, for a capture.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLTexture> {
        &self.raw
    }

    pub fn create_view(&self, desc: &ViewDesc) -> TextureView {
        let levels = desc.level_count.unwrap_or(self.levels - desc.base_level);
        let all_layers = match self.dimension {
            TextureDimension::D3 => 1,
            TextureDimension::D2 => self.depth_or_layers,
        };
        let layers = desc.layer_count.unwrap_or(all_layers - desc.base_layer);
        let ty = match desc.dimension {
            ViewDimension::D2 => MTLTextureType::Type2D,
            ViewDimension::D2Array => MTLTextureType::Type2DArray,
            ViewDimension::Cube => MTLTextureType::TypeCube,
            ViewDimension::D3 => MTLTextureType::Type3D,
        };
        let raw = unsafe {
            self.raw.newTextureViewWithPixelFormat_textureType_levels_slices(
                pixel_format(self.format),
                ty,
                NSRange::new(desc.base_level as NSUInteger, levels as NSUInteger),
                NSRange::new(desc.base_layer as NSUInteger, layers as NSUInteger),
            )
        }
        .expect("Metal made no texture view");
        TextureView { raw }
    }
}

#[derive(Clone)]
pub struct TextureView {
    raw: Id<dyn MTLTexture>,
}

#[derive(Clone)]
pub struct Sampler {
    raw: Id<dyn MTLSamplerState>,
}

pub struct QuerySet {
    raw: Id<dyn MTLCounterSampleBuffer>,
    count: u32,
}

impl QuerySet {
    pub fn count(&self) -> u32 {
        self.count
    }
}

pub struct ShaderModule {
    library: Id<dyn MTLLibrary>,
}

pub struct ComputePipeline {
    raw: Id<dyn MTLComputePipelineState>,
    workgroup: MTLSize,
}

impl ComputePipeline {
    /// The pipeline state, for a pipeline's report.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.raw
    }
}

pub struct RenderPipeline {
    raw: Id<dyn MTLRenderPipelineState>,
    depth_state: Option<Id<dyn MTLDepthStencilState>>,
    cull: MTLCullMode,
    bias: DepthBias,
}

// ---- bindings -----------------------------------------------------------------------------------------

/// A layout's entries, and each one's slot in each stage's table (`wrela_abi::metal::slots`'
/// rule: a stage's own counter for each table, in the entries' order, counting only what the
/// stage sees).
struct Layout {
    entries: Vec<LayoutEntry>,
    /// Each entry's slot in the vertex, fragment and compute stage, where it sees it.
    slots: Vec<[Option<u8>; 3]>,
    /// The entry that takes a dynamic offset.
    dynamic: Vec<usize>,
}

impl Layout {
    fn of(entries: &[LayoutEntry]) -> Layout {
        let mut counters = [[0u8; 3]; 3];
        let mut slots = Vec::with_capacity(entries.len());
        let mut dynamic = Vec::new();
        for (k, e) in entries.iter().enumerate() {
            let table = e.ty.table() as usize;
            let mut s = [None; 3];
            for (stage, sees) in [e.visibility.vertex, e.visibility.fragment, e.visibility.compute]
                .into_iter()
                .enumerate()
            {
                if sees {
                    s[stage] = Some(counters[stage][table]);
                    counters[stage][table] += 1;
                }
            }
            slots.push(s);
            if matches!(e.ty, BindingType::Uniform { dynamic: true, .. }) {
                dynamic.push(k);
            }
        }
        Layout { entries: entries.to_vec(), slots, dynamic }
    }
}

#[derive(Clone)]
pub struct BindGroupLayout(Arc<Layout>);

enum Bound {
    Buffer(Id<dyn MTLBuffer>, u64),
    Texture(Id<dyn MTLTexture>),
    Sampler(Id<dyn MTLSamplerState>),
}

struct Group {
    layout: Arc<Layout>,
    bound: Vec<(usize, Bound)>,
}

#[derive(Clone)]
pub struct BindGroup(Arc<Group>);

// SAFETY: as `Shared`'s: Metal's objects may be used from any thread.
unsafe impl Send for Group {}
unsafe impl Sync for Group {}
unsafe impl Send for Layout {}
unsafe impl Sync for Layout {}

impl BindGroup {
    /// Each bound resource with its slot in `stage` (0 vertex, 1 fragment, 2 compute), its
    /// buffer offset moved by the dynamic offsets in the layout's order.
    fn each(&self, stage: usize, offsets: &[u32], mut f: impl FnMut(&Bound, u64, u8)) {
        let g = &self.0;
        for (k, b) in &g.bound {
            let Some(slot) = g.layout.slots[*k][stage] else { continue };
            let extra = g
                .layout
                .dynamic
                .iter()
                .position(|d| d == k)
                .and_then(|i| offsets.get(i))
                .map_or(0, |o| u64::from(*o));
            f(b, extra, slot);
        }
    }
}

// ---- recording ----------------------------------------------------------------------------------------

/// A command buffer being recorded.
pub struct CommandEncoder {
    device: Device,
    label: String,
    cb: Option<Id<dyn MTLCommandBuffer>>,
    /// The blit encoder copies go into, while copies follow one another.
    blit: Option<Id<dyn MTLBlitCommandEncoder>>,
}

pub struct CommandBuffer {
    raw: Id<dyn MTLCommandBuffer>,
}

impl CommandEncoder {
    fn cb(&mut self) -> &Id<dyn MTLCommandBuffer> {
        if self.cb.is_none() {
            let cb = self.device.command_buffer();
            cb.setLabel(Some(&ns(&self.label)));
            self.cb = Some(cb);
        }
        self.cb.as_ref().expect("made above")
    }

    fn end_blit(&mut self) {
        if let Some(b) = self.blit.take() {
            b.endEncoding();
        }
    }

    fn blit(&mut self) -> &Id<dyn MTLBlitCommandEncoder> {
        if self.blit.is_none() {
            let b = self.cb().blitCommandEncoder().expect("Metal made no blit encoder");
            b.setLabel(Some(&ns("copies")));
            self.blit = Some(b);
        }
        self.blit.as_ref().expect("made above")
    }

    pub fn copy_buffer_to_buffer(
        &mut self,
        src: &Buffer,
        src_offset: u64,
        dst: &Buffer,
        dst_offset: u64,
        size: u64,
    ) {
        let b = self.blit();
        unsafe {
            b.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src.raw,
                src_offset as NSUInteger,
                &dst.raw,
                dst_offset as NSUInteger,
                size as NSUInteger,
            );
        }
    }

    pub fn copy_buffer_to_texture(
        &mut self,
        src: BufferCopy<'_>,
        dst: TextureCopy<'_>,
        extent: [u32; 3],
    ) {
        let (slice, z) = slice_of(&dst);
        let b = self.blit();
        unsafe {
            b.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &src.buffer.raw,
                src.offset as NSUInteger,
                src.bytes_per_row as NSUInteger,
                (u64::from(src.bytes_per_row) * u64::from(src.rows_per_image)) as NSUInteger,
                size(extent),
                &dst.texture.raw,
                slice,
                dst.level as NSUInteger,
                MTLOrigin { x: dst.origin[0] as NSUInteger, y: dst.origin[1] as NSUInteger, z },
            );
        }
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        src: TextureCopy<'_>,
        dst: BufferCopy<'_>,
        extent: [u32; 3],
    ) {
        let (slice, z) = slice_of(&src);
        let b = self.blit();
        unsafe {
            b.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                &src.texture.raw,
                slice,
                src.level as NSUInteger,
                MTLOrigin { x: src.origin[0] as NSUInteger, y: src.origin[1] as NSUInteger, z },
                size(extent),
                &dst.buffer.raw,
                dst.offset as NSUInteger,
                dst.bytes_per_row as NSUInteger,
                (u64::from(dst.bytes_per_row) * u64::from(dst.rows_per_image)) as NSUInteger,
            );
        }
    }

    pub fn copy_texture_to_texture(
        &mut self,
        src: TextureCopy<'_>,
        dst: TextureCopy<'_>,
        extent: [u32; 3],
    ) {
        let (ss, sz) = slice_of(&src);
        let (ds, dz) = slice_of(&dst);
        let b = self.blit();
        unsafe {
            b.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &src.texture.raw,
                ss,
                src.level as NSUInteger,
                MTLOrigin { x: src.origin[0] as NSUInteger, y: src.origin[1] as NSUInteger, z: sz },
                size(extent),
                &dst.texture.raw,
                ds,
                dst.level as NSUInteger,
                MTLOrigin { x: dst.origin[0] as NSUInteger, y: dst.origin[1] as NSUInteger, z: dz },
            );
        }
    }

    /// Copies `queries`' timestamps `first..first + count` into `dst` at `offset`, as `u64`s
    /// (nanoseconds).
    pub fn resolve_query_set(
        &mut self,
        queries: &QuerySet,
        first: u32,
        count: u32,
        dst: &Buffer,
        offset: u64,
    ) {
        let b = self.blit();
        unsafe {
            b.resolveCounters_inRange_destinationBuffer_destinationOffset(
                &queries.raw,
                NSRange::new(first as NSUInteger, count as NSUInteger),
                &dst.raw,
                offset as NSUInteger,
            );
        }
    }

    pub fn begin_compute_pass(
        &mut self,
        label: &str,
        timestamps: Option<PassTimestamps<'_>>,
    ) -> ComputePass<'_> {
        self.end_blit();
        let d = MTLComputePassDescriptor::new();
        if let Some(t) = timestamps {
            let a = unsafe { d.sampleBufferAttachments().objectAtIndexedSubscript(0) };
            a.setSampleBuffer(Some(&t.queries.raw));
            unsafe {
                a.setStartOfEncoderSampleIndex(t.begin as NSUInteger);
                a.setEndOfEncoderSampleIndex(t.end as NSUInteger);
            }
        }
        let encoder = self
            .cb()
            .computeCommandEncoderWithDescriptor(&d)
            .expect("Metal made no compute encoder");
        encoder.setLabel(Some(&ns(label)));
        ComputePass { encoder, pipeline: None, _cb: std::marker::PhantomData }
    }

    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc<'_>) -> RenderPass<'_> {
        self.end_blit();
        let d = MTLRenderPassDescriptor::new();
        let colors = d.colorAttachments();
        for (k, c) in desc.colors.iter().enumerate() {
            let a = unsafe { colors.objectAtIndexedSubscript(k as NSUInteger) };
            a.setTexture(Some(&c.view.raw));
            match c.load {
                LoadOp::Load => a.setLoadAction(MTLLoadAction::Load),
                LoadOp::Clear([r, g, b, alpha]) => {
                    a.setLoadAction(MTLLoadAction::Clear);
                    a.setClearColor(MTLClearColor { red: r, green: g, blue: b, alpha });
                }
            }
            a.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(depth) = &desc.depth {
            let a = d.depthAttachment();
            a.setTexture(Some(&depth.view.raw));
            match depth.load {
                LoadOp::Load => a.setLoadAction(MTLLoadAction::Load),
                LoadOp::Clear(z) => {
                    a.setLoadAction(MTLLoadAction::Clear);
                    a.setClearDepth(f64::from(z));
                }
            }
            a.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(t) = desc.timestamps {
            let a = unsafe { d.sampleBufferAttachments().objectAtIndexedSubscript(0) };
            a.setSampleBuffer(Some(&t.queries.raw));
            unsafe {
                a.setStartOfVertexSampleIndex(t.begin as NSUInteger);
                a.setEndOfVertexSampleIndex(MTLCounterDontSample);
                a.setStartOfFragmentSampleIndex(MTLCounterDontSample);
                a.setEndOfFragmentSampleIndex(t.end as NSUInteger);
            }
        }
        let encoder =
            self.cb().renderCommandEncoderWithDescriptor(&d).expect("Metal made no render encoder");
        encoder.setLabel(Some(&ns(desc.label)));
        encoder.setFrontFacingWinding(MTLWinding::CounterClockwise);
        RenderPass { encoder, index: None, _cb: std::marker::PhantomData }
    }

    pub fn finish(mut self) -> CommandBuffer {
        self.end_blit();
        CommandBuffer { raw: self.cb().clone() }
    }
}

fn size(e: [u32; 3]) -> MTLSize {
    MTLSize {
        width: e[0] as NSUInteger,
        height: e[1] as NSUInteger,
        depth: e[2].max(1) as NSUInteger,
    }
}

/// A copy's slice and depth: a 2D texture's layer is its slice; a 3D texture's is its depth.
fn slice_of(c: &TextureCopy<'_>) -> (NSUInteger, NSUInteger) {
    match c.texture.dimension {
        TextureDimension::D3 => (0, c.origin[2] as NSUInteger),
        TextureDimension::D2 => (c.origin[2] as NSUInteger, 0),
    }
}

pub struct ComputePass<'a> {
    encoder: Id<dyn MTLComputeCommandEncoder>,
    pipeline: Option<MTLSize>,
    _cb: std::marker::PhantomData<&'a mut CommandEncoder>,
}

impl ComputePass<'_> {
    pub fn set_pipeline(&mut self, p: &ComputePipeline) {
        self.encoder.setComputePipelineState(&p.raw);
        self.pipeline = Some(p.workgroup);
    }

    pub fn set_bind_group(&mut self, group: &BindGroup, offsets: &[u32]) {
        let e = &self.encoder;
        group.each(2, offsets, |b, extra, slot| unsafe {
            match b {
                Bound::Buffer(buf, offset) => e.setBuffer_offset_atIndex(
                    Some(buf),
                    (offset + extra) as NSUInteger,
                    slot as NSUInteger,
                ),
                Bound::Texture(t) => e.setTexture_atIndex(Some(t), slot as NSUInteger),
                Bound::Sampler(s) => e.setSamplerState_atIndex(Some(s), slot as NSUInteger),
            }
        });
    }

    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        let wg = self.pipeline.expect("a pipeline is set");
        self.encoder.dispatchThreadgroups_threadsPerThreadgroup(size([x, y, z]), wg);
    }

    pub fn dispatch_indirect(&mut self, buffer: &Buffer, offset: u64) {
        let wg = self.pipeline.expect("a pipeline is set");
        unsafe {
            self.encoder
                .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    &buffer.raw,
                    offset as NSUInteger,
                    wg,
                );
        }
    }
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        self.encoder.endEncoding();
    }
}

pub struct RenderPass<'a> {
    encoder: Id<dyn MTLRenderCommandEncoder>,
    /// The index buffer and its offset, for the indexed draws after it.
    index: Option<(Id<dyn MTLBuffer>, u64)>,
    _cb: std::marker::PhantomData<&'a mut CommandEncoder>,
}

impl RenderPass<'_> {
    pub fn set_pipeline(&mut self, p: &RenderPipeline) {
        let e = &self.encoder;
        e.setRenderPipelineState(&p.raw);
        if let Some(s) = &p.depth_state {
            e.setDepthStencilState(Some(s));
        }
        e.setCullMode(p.cull);
        e.setDepthBias_slopeScale_clamp(p.bias.constant as f32, p.bias.slope_scale, p.bias.clamp);
    }

    pub fn set_bind_group(&mut self, group: &BindGroup, offsets: &[u32]) {
        let e = &self.encoder;
        for stage in 0..2 {
            group.each(stage, offsets, |b, extra, slot| unsafe {
                let slot = slot as NSUInteger;
                match (stage, b) {
                    (0, Bound::Buffer(buf, offset)) => e.setVertexBuffer_offset_atIndex(
                        Some(buf),
                        (offset + extra) as NSUInteger,
                        slot,
                    ),
                    (_, Bound::Buffer(buf, offset)) => e.setFragmentBuffer_offset_atIndex(
                        Some(buf),
                        (offset + extra) as NSUInteger,
                        slot,
                    ),
                    (0, Bound::Texture(t)) => e.setVertexTexture_atIndex(Some(t), slot),
                    (_, Bound::Texture(t)) => e.setFragmentTexture_atIndex(Some(t), slot),
                    (0, Bound::Sampler(s)) => e.setVertexSamplerState_atIndex(Some(s), slot),
                    (_, Bound::Sampler(s)) => e.setFragmentSamplerState_atIndex(Some(s), slot),
                }
            });
        }
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min: f32, max: f32) {
        self.encoder.setViewport(MTLViewport {
            originX: f64::from(x),
            originY: f64::from(y),
            width: f64::from(w),
            height: f64::from(h),
            znear: f64::from(min),
            zfar: f64::from(max),
        });
    }

    pub fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        self.encoder.setScissorRect(MTLScissorRect {
            x: x as NSUInteger,
            y: y as NSUInteger,
            width: w as NSUInteger,
            height: h as NSUInteger,
        });
    }

    /// `u32` indices from `offset` of `buffer`.
    pub fn set_index_buffer(&mut self, buffer: &Buffer, offset: u64) {
        self.index = Some((buffer.raw.clone(), offset));
    }

    pub fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        unsafe {
            self.encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                vertices.start as NSUInteger,
                vertices.len() as NSUInteger,
                instances.len() as NSUInteger,
                instances.start as NSUInteger,
            );
        }
    }

    pub fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        let (buffer, offset) = self.index.clone().expect("an index buffer is set");
        unsafe {
            self.encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                MTLPrimitiveType::Triangle,
                indices.len() as NSUInteger,
                MTLIndexType::UInt32,
                &buffer,
                (offset + u64::from(indices.start) * 4) as NSUInteger,
                instances.len() as NSUInteger,
                base_vertex as isize,
                instances.start as NSUInteger,
            );
        }
    }

    pub fn draw_indirect(&mut self, buffer: &Buffer, offset: u64) {
        unsafe {
            self.encoder.drawPrimitives_indirectBuffer_indirectBufferOffset(
                MTLPrimitiveType::Triangle,
                &buffer.raw,
                offset as NSUInteger,
            );
        }
    }

    pub fn draw_indexed_indirect(&mut self, buffer: &Buffer, offset: u64) {
        let (index, index_offset) = self.index.clone().expect("an index buffer is set");
        unsafe {
            self.encoder.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                MTLPrimitiveType::Triangle,
                MTLIndexType::UInt32,
                &index,
                index_offset as NSUInteger,
                &buffer.raw,
                offset as NSUInteger,
            );
        }
    }

    /// `count` indirect draws, their arguments 16 bytes apart from `offset`.
    pub fn multi_draw_indirect(&mut self, buffer: &Buffer, offset: u64, count: u32) {
        for k in 0..u64::from(count) {
            self.draw_indirect(buffer, offset + 16 * k);
        }
    }

    /// `count` indexed indirect draws, their arguments 20 bytes apart from `offset`.
    pub fn multi_draw_indexed_indirect(&mut self, buffer: &Buffer, offset: u64, count: u32) {
        for k in 0..u64::from(count) {
            self.draw_indexed_indirect(buffer, offset + 20 * k);
        }
    }
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        self.encoder.endEncoding();
    }
}

// ---- conversions --------------------------------------------------------------------------------------

pub fn pixel_format(f: TextureFormat) -> MTLPixelFormat {
    match f {
        TextureFormat::Rgba8 => MTLPixelFormat::RGBA8Unorm,
        TextureFormat::Rgba16Float => MTLPixelFormat::RGBA16Float,
        TextureFormat::Depth32Float => MTLPixelFormat::Depth32Float,
        TextureFormat::R16Float => MTLPixelFormat::R16Float,
        TextureFormat::Rg16Float => MTLPixelFormat::RG16Float,
        TextureFormat::R32Float => MTLPixelFormat::R32Float,
        TextureFormat::R32Uint => MTLPixelFormat::R32Uint,
        TextureFormat::R8 => MTLPixelFormat::R8Unorm,
        TextureFormat::Rg8 => MTLPixelFormat::RG8Unorm,
        TextureFormat::Rgb10a2 => MTLPixelFormat::RGB10A2Unorm,
        TextureFormat::Rg32Float => MTLPixelFormat::RG32Float,
        TextureFormat::Rgba32Float => MTLPixelFormat::RGBA32Float,
        TextureFormat::Rg32Uint => MTLPixelFormat::RG32Uint,
        TextureFormat::Rg11b10Float => MTLPixelFormat::RG11B10Float,
        TextureFormat::Bc1 => MTLPixelFormat::BC1_RGBA,
        TextureFormat::Bc4 => MTLPixelFormat::BC4_RUnorm,
        TextureFormat::Bc5 => MTLPixelFormat::BC5_RGUnorm,
        TextureFormat::Bc7 => MTLPixelFormat::BC7_RGBAUnorm,
        TextureFormat::Etc2Rgb8 => MTLPixelFormat::ETC2_RGB8,
        TextureFormat::Astc4x4 => MTLPixelFormat::ASTC_4x4_LDR,
    }
}

fn compare_function(c: Compare) -> MTLCompareFunction {
    match c {
        Compare::Less => MTLCompareFunction::Less,
        Compare::LessEqual => MTLCompareFunction::LessEqual,
        Compare::Greater => MTLCompareFunction::Greater,
        Compare::GreaterEqual => MTLCompareFunction::GreaterEqual,
        Compare::Equal => MTLCompareFunction::Equal,
        Compare::NotEqual => MTLCompareFunction::NotEqual,
        Compare::Always => MTLCompareFunction::Always,
        Compare::Never => MTLCompareFunction::Never,
    }
}

fn blend_factor(name: &str) -> MTLBlendFactor {
    match name {
        "zero" => MTLBlendFactor::Zero,
        "one" => MTLBlendFactor::One,
        "src-alpha" => MTLBlendFactor::SourceAlpha,
        "one-minus-src-alpha" => MTLBlendFactor::OneMinusSourceAlpha,
        "dst" => MTLBlendFactor::DestinationColor,
        "dst-alpha" => MTLBlendFactor::DestinationAlpha,
        other => unreachable!("Blend::components names no factor `{other}`"),
    }
}

fn blend_operation(name: &str) -> MTLBlendOperation {
    match name {
        "add" => MTLBlendOperation::Add,
        "min" => MTLBlendOperation::Min,
        "max" => MTLBlendOperation::Max,
        other => unreachable!("Blend::components names no operation `{other}`"),
    }
}

// ---- what reads Metal directly ------------------------------------------------------------------------

/// What Metal says of a compute kernel: its SIMD width, the most threads a threadgroup of it may
/// have, and its static threadgroup memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelReport {
    pub simd: u64,
    pub max_threads: u64,
    pub threadgroup_memory: u64,
}

impl ComputePipeline {
    pub fn report(&self) -> KernelReport {
        KernelReport {
            simd: self.raw.threadExecutionWidth() as u64,
            max_threads: self.raw.maxTotalThreadsPerThreadgroup() as u64,
            threadgroup_memory: self.raw.staticThreadgroupMemoryLength() as u64,
        }
    }
}

impl Device {
    /// `msl`'s kernel `entry` compiled on its own, as the pipelines are: what Metal says of it.
    pub fn kernel_report(&self, msl: &str, entry: &str) -> Result<KernelReport> {
        let module = self.create_shader_module("report", msl.as_bytes())?;
        let layout = self.create_bind_group_layout(&[]);
        let p = self.create_compute_pipeline(&ComputePipelineDesc {
            label: "report",
            layout: &layout,
            module: &module,
            entry,
            workgroup_size: [1, 1, 1],
        })?;
        Ok(p.report())
    }

    /// Starts capturing this device's work to the GPU trace `path` (a new `.gputrace`), which
    /// Xcode opens. Metal captures to a file only in a process started with
    /// `MTL_CAPTURE_ENABLED=1`.
    pub fn start_capture(&self, path: &std::path::Path) -> std::result::Result<(), String> {
        use objc2_foundation::NSURL;
        autoreleasepool(|_| {
            let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
            if !manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument) {
                return Err("Metal won't capture to a file in this process: start it with MTL_CAPTURE_ENABLED=1 in its environment".into());
            }
            let d = MTLCaptureDescriptor::new();
            unsafe { d.setCaptureObject(Some(self.0.raw.as_ref())) };
            d.setDestination(MTLCaptureDestination::GPUTraceDocument);
            let url = NSURL::fileURLWithPath(&ns(&path.to_string_lossy()));
            d.setOutputURL(Some(&url));
            manager
                .startCaptureWithDescriptor_error(&d)
                .map_err(|e| format!("Metal couldn't start the capture: {}", error_text(&e)))
        })
    }

    pub fn stop_capture(&self) {
        autoreleasepool(|_| {
            let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
            if manager.isCapturing() {
                manager.stopCapture();
            }
        });
    }
}
