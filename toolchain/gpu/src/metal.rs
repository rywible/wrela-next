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
//! - The table (`wrela_abi::table`): every texture view and sampler has an index in a shared
//!   buffer of `MTLResourceID`s, bound at buffer slot `METAL_TABLE` of an encoder's stages once a
//!   bind group needs it; a bind group's indices go in by `setBytes` at `METAL_BLOCK`. What a
//!   command reaches through the table it declares with `useResources` (residency, and Metal's
//!   hazard tracking), and the command buffer keeps its groups alive until it's done (Metal
//!   retains only what's bound directly).

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
    /// Each recent submission's span on the GPU's clock (seconds): the first of its command
    /// buffers' starts and the last of their ends, newest last ([`Device::gpu_ms`]); and how
    /// long the GPU had nothing of this device's before it ([`Device::idle_before_ms`]).
    clock: Arc<Mutex<std::collections::VecDeque<(SubmissionIndex, f64, f64, f64)>>>,
    /// Command buffers committed and not yet seen complete, by submission, with the bind groups
    /// their commands set, kept alive until they're done (what the table names in them).
    in_flight: Mutex<Vec<(SubmissionIndex, Id<dyn MTLCommandBuffer>, Vec<Arc<Group>>)>>,
    /// Staged writes, copied before the next submission's own work.
    pending: Mutex<Vec<PendingWrite>>,
    /// Errors the GPU reported: a failed command buffer, naming its encoder.
    errors: Arc<Mutex<Vec<String>>>,
    /// A buffer of zeros, shared, as large as the largest texture level zeroed so far.
    zeros: Mutex<Option<Id<dyn MTLBuffer>>>,
    /// Shared buffers staged writes are copied into, reused once their submission is done.
    belt: Mutex<Belt>,
    /// The table of texture views and samplers.
    table: Arc<Table>,
}

/// How many views and samplers the table holds at once.
const TABLE_LEN: u32 = 1 << 16;

/// The device's table (`wrela_abi::table`): a shared buffer of `MTLResourceID`s, and its free
/// indices (those let go, and the next never used).
struct Table {
    buffer: Id<dyn MTLBuffer>,
    free: Mutex<(Vec<u32>, u32)>,
}

// SAFETY: as `Shared`'s.
unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Table {
    /// An index for `held`'s `id`, written into the table.
    fn take(self: &Arc<Table>, id: MTLResourceID, held: Held) -> Arc<Slot> {
        let index = {
            let mut f = lock(&self.free);
            match f.0.pop() {
                Some(i) => i,
                None => {
                    let i = f.1;
                    assert!(i < TABLE_LEN, "the table of views and samplers is full");
                    f.1 += 1;
                    i
                }
            }
        };
        // SAFETY: `MTLResourceID` is one 64-bit word; the shared buffer holds `TABLE_LEN` of
        // them, and no command in flight reads a free index.
        unsafe {
            let raw: u64 = std::mem::transmute(id);
            let p = self.buffer.contents().as_ptr() as *mut u64;
            p.add(index as usize).write(raw);
        }
        Arc::new(Slot { table: self.clone(), index, _held: held })
    }
}

/// An index of the table, given back when the last view or sampler holding it goes (after every
/// command buffer that reached it: their bind groups hold it too). It holds the texture or
/// sampler its entry names: Metal gives a released object's `MTLResourceID` to the next one
/// made, so an entry a command in flight reads must not outlive its object (a kernel bound to a
/// sampler the program let go of before the GPU ran it read the next sampler made).
struct Slot {
    table: Arc<Table>,
    index: u32,
    _held: Held,
}

/// What a table entry names, held (never read: holding it is its job).
#[allow(dead_code)]
enum Held {
    Texture(Id<dyn MTLTexture>),
    Sampler(Id<dyn MTLSamplerState>),
}

impl Drop for Slot {
    fn drop(&mut self) {
        lock(&self.table.free).0.push(self.index);
    }
}

/// The staging belt: the buffer this submission's writes go into (and how much of it they
/// fill), the full ones this submission also uses, and those earlier submissions use or used.
#[derive(Default)]
struct Belt {
    current: Option<(Id<dyn MTLBuffer>, u64)>,
    filled: Vec<Id<dyn MTLBuffer>>,
    in_flight: Vec<(SubmissionIndex, Id<dyn MTLBuffer>)>,
    free: Vec<Id<dyn MTLBuffer>>,
}

/// The least a staging buffer holds.
const BELT_CHUNK: u64 = 4 << 20;

// SAFETY: Metal's objects are thread-safe (MTLDevice, MTLCommandQueue and resources may be used
// from any thread); the host records on one thread at a time.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

enum PendingWrite {
    Buffer {
        staging: Id<dyn MTLBuffer>,
        staging_offset: u64,
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
                // Metal's compiler ships with the OS: its version is the driver's.
                driver: "Metal".into(),
                driver_info: objc2_foundation::NSProcessInfo::processInfo()
                    .operatingSystemVersionString()
                    .to_string(),
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
            let table = raw
                .newBufferWithLength_options(
                    (TABLE_LEN as usize) * 8,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or("Metal made no table")?;
            table.setLabel(Some(&ns("table")));
            let table = Arc::new(Table { buffer: table, free: Mutex::new((Vec::new(), 0)) });
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
                clock: Arc::new(Mutex::new(std::collections::VecDeque::new())),
                in_flight: Mutex::new(Vec::new()),
                table,
                pending: Mutex::new(Vec::new()),
                errors: Arc::new(Mutex::new(Vec::new())),
                zeros: Mutex::new(None),
                belt: Mutex::new(Belt::default()),
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
        // No `PixelFormatView`: every view keeps its texture's format, and the flag would give up
        // the GPU's lossless compression of what's drawn into it.
        let mut usage = MTLTextureUsage::ShaderRead;
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
            table: self.0.table.clone(),
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
        let slot = self.0.table.take(raw.gpuResourceID(), Held::Sampler(raw.clone()));
        Sampler { raw, slot }
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
        let mut block = vec![0u8; layout.0.block];
        let (mut reads, mut writes, mut slots) = (Vec::new(), Vec::new(), Vec::new());
        for e in entries {
            let Some(k) = layout.0.entries.iter().position(|l| l.binding == e.binding) else {
                continue;
            };
            if let Some(at) = layout.0.entries[k].index_at {
                let slot = match &e.resource {
                    Resource::View(v) => {
                        if matches!(layout.0.entries[k].ty, BindingType::StorageTexture { .. }) {
                            writes.push(v.raw.clone());
                        } else {
                            reads.push(v.raw.clone());
                        }
                        v.slot.clone().expect("a view in the table")
                    }
                    Resource::Sampler(s) => s.slot.clone(),
                    Resource::Buffer { .. } => panic!("a buffer isn't in the table"),
                };
                let at = at as usize;
                block[at..at + 4].copy_from_slice(&slot.index.to_le_bytes());
                slots.push(slot);
                continue;
            }
            let bound = match &e.resource {
                Resource::Buffer { buffer, offset, .. } => {
                    Bound::Buffer(buffer.raw.clone(), *offset)
                }
                Resource::View(v) => Bound::Texture(v.raw.clone()),
                Resource::Sampler(s) => Bound::Sampler(s.raw.clone()),
            };
            out.push((k, bound));
        }
        BindGroup(Arc::new(Group {
            layout: layout.0.clone(),
            bound: out,
            block,
            reads,
            writes,
            _slots: slots,
        }))
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
        CommandEncoder {
            device: self.clone(),
            label: label.to_string(),
            cb: None,
            blit: None,
            keep: Vec::new(),
        }
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
        let (staging, staging_offset) = self.staging(data, 4);
        lock(&self.0.pending).push(PendingWrite::Buffer {
            staging,
            staging_offset,
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
        let (staging, staging_offset) = self.staging(data, 256);
        let (slice, z) = match at.texture.dimension {
            TextureDimension::D3 => (0, at.origin[2]),
            TextureDimension::D2 => (at.origin[2], 0),
        };
        lock(&self.0.pending).push(PendingWrite::Texture {
            staging,
            staging_offset,
            bytes_per_row: u64::from(bytes_per_row),
            bytes_per_image: u64::from(bytes_per_row) * u64::from(rows_per_image),
            dst: at.texture.raw.clone(),
            level: at.level,
            slice,
            origin: [at.origin[0], at.origin[1], z],
            size: extent,
        });
    }

    /// `data` copied into the staging belt at an offset aligned to `align`: the buffer and the
    /// offset, which the next submission copies from.
    fn staging(&self, data: &[u8], align: u64) -> (Id<dyn MTLBuffer>, u64) {
        let need = data.len() as u64;
        let mut belt = lock(&self.0.belt);
        let fits = |b: &Id<dyn MTLBuffer>, used: u64| {
            used.next_multiple_of(align) + need <= b.length() as u64
        };
        let (buffer, at) = match belt.current.take() {
            Some((b, used)) if fits(&b, used) => (b, used.next_multiple_of(align)),
            other => {
                if let Some((b, _)) = other {
                    belt.filled.push(b);
                }
                let free = belt.free.iter().position(|b| b.length() as u64 >= need);
                let b = match free {
                    Some(k) => belt.free.swap_remove(k),
                    None => self
                        .0
                        .raw
                        .newBufferWithLength_options(
                            need.max(BELT_CHUNK).next_power_of_two() as NSUInteger,
                            MTLResourceOptions::StorageModeShared,
                        )
                        .expect("Metal made no buffer"),
                };
                (b, 0)
            }
        };
        // SAFETY: the buffer is shared and holds `at + need` bytes; no submission uses it yet.
        unsafe {
            let p = buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), p.add(at as usize), data.len());
        }
        belt.current = Some((buffer.clone(), at + need));
        (buffer, at)
    }

    /// The staging belt's buffers this submission uses go in flight with it; those of
    /// submissions done are free again.
    fn turn_belt(&self, index: SubmissionIndex) {
        let done = self.completed();
        let mut belt = lock(&self.0.belt);
        let current = belt.current.take().map(|c| c.0);
        let used: Vec<_> = belt.filled.drain(..).chain(current).collect();
        belt.in_flight.extend(used.into_iter().map(|b| (index, b)));
        let (ready, busy): (Vec<_>, Vec<_>) =
            belt.in_flight.drain(..).partition(|(i, _)| *i <= done);
        belt.in_flight = busy;
        belt.free.extend(ready.into_iter().map(|(_, b)| b));
        belt.free.truncate(8);
    }

    /// Commits the staged writes, then `buffers`, in order: their submission's index.
    pub fn submit(&self, buffers: Vec<CommandBuffer>) -> SubmissionIndex {
        autoreleasepool(|_| {
            let index = self.0.submitted.fetch_add(1, Ordering::AcqRel) + 1;
            let pending = self.take_pending();
            self.turn_belt(index);
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
            let mut kept = Vec::new();
            for b in buffers {
                cbs.push(b.raw);
                kept.extend(b.keep);
            }
            if cbs.is_empty() {
                // Nothing to run: a command buffer of its own marks the submission done.
                let cb = self.command_buffer();
                cbs.push(cb);
            }
            let last = cbs.len() - 1;
            for (k, cb) in cbs.iter().enumerate() {
                let errors = self.0.errors.clone();
                let completed = self.0.completed.clone();
                let clock = self.0.clock.clone();
                let is_last = k == last;
                let committed = std::time::Instant::now();
                let block =
                    RcBlock::new(move |cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                        let cb = unsafe { cb.as_ref() };
                        if cb.status() == MTLCommandBufferStatus::Error {
                            let mut e = command_buffer_error(cb);
                            e.push_str(&format!(
                                " (submission {index}, buffer {} of {}; failed {:.0} ms after its \
                                 commit{})",
                                k + 1,
                                last + 1,
                                committed.elapsed().as_secs_f64() * 1e3,
                                match (cb.GPUStartTime(), cb.GPUEndTime()) {
                                    (s, e) if s > 0.0 && e > s =>
                                        format!(", {:.1} ms on the GPU's clock", (e - s) * 1e3),
                                    _ => String::new(),
                                }
                            ));
                            lock(&errors).push(e);
                        }
                        // The submission's span: its buffers complete in order, each before
                        // the next's handler runs. A command buffer's start is near its commit,
                        // not when the work before it let it run (a pass of 40 draws, each its
                        // own submission, timed 1.4, 2.8, 4.3 ms...): it starts no earlier than
                        // the submission before it ended.
                        let (start, end) = (cb.GPUStartTime(), cb.GPUEndTime());
                        let mut c = lock(&clock);
                        match c.back_mut() {
                            // A later buffer of the same submission starts after its first.
                            Some((i, _, e, _)) if *i == index => *e = e.max(end),
                            back => {
                                let before = back.map(|(_, _, e, _)| *e);
                                let idle = before.map_or(0.0, |e| (start - e).max(0.0));
                                let start = before.map_or(start, |e| start.max(e));
                                c.push_back((index, start, end, idle));
                                while c.len() > CLOCK_KEPT {
                                    c.pop_front();
                                }
                            }
                        }
                        drop(c);
                        if is_last {
                            completed.fetch_max(index, Ordering::AcqRel);
                        }
                    });
                unsafe {
                    cb.addCompletedHandler(RcBlock::as_ptr(&block));
                }
                cb.commit();
            }
            let mut f = lock(&self.0.in_flight);
            // What's done lets go of what it kept.
            let done = self.completed();
            f.retain(|(i, ..)| *i > done);
            f.push((index, cbs[last].clone(), kept));
            index
        })
    }

    /// The submissions done so far: every one up to this index.
    pub fn completed(&self) -> SubmissionIndex {
        self.0.completed.load(Ordering::Acquire)
    }

    /// Whether it times its submissions ([`Device::gpu_ms`]): a Metal device always does.
    pub fn has_clock(&self) -> bool {
        true
    }

    /// How long submission `index` held the GPU (ms), from its first command buffer's start (or
    /// the end of the submission before it, if later) to its last one's end on the GPU's clock: once it's done, while it's among the last
    /// [`CLOCK_KEPT`] submissions.
    pub fn gpu_ms(&self, index: SubmissionIndex) -> Option<f64> {
        if index > self.completed() {
            return None;
        }
        let c = lock(&self.0.clock);
        let k = c.binary_search_by_key(&index, |(i, ..)| *i).ok()?;
        let (_, start, end, _) = c[k];
        (end >= start && start > 0.0).then_some((end - start) * 1e3)
    }

    /// How long the GPU had none of this device's work before submission `index` began (ms): from
    /// the end of the submission before it to its start, on the GPU's clock; once it's done,
    /// while it's among the last [`CLOCK_KEPT`] submissions.
    pub fn idle_before_ms(&self, index: SubmissionIndex) -> Option<f64> {
        if index > self.completed() {
            return None;
        }
        let c = lock(&self.0.clock);
        let k = c.binary_search_by_key(&index, |(i, ..)| *i).ok()?;
        (k > 0).then(|| c[k].3 * 1e3)
    }

    pub fn submitted(&self) -> SubmissionIndex {
        self.0.submitted.load(Ordering::Acquire)
    }

    /// Waits for submission `index` (and every one before it) to be done.
    pub fn wait(&self, index: SubmissionIndex) {
        #[allow(clippy::type_complexity)]
        let waiting: Vec<(SubmissionIndex, Id<dyn MTLCommandBuffer>, Vec<Arc<Group>>)> = {
            let mut f = lock(&self.0.in_flight);
            let (w, rest) = std::mem::take(&mut *f).into_iter().partition(|(i, ..)| *i <= index);
            *f = rest;
            w
        };
        for (_, cb, _) in &waiting {
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

    /// Reports the device lost, as a failed command buffer is reported: what the host's
    /// handling of a loss is tested by (`WRELA_GPU_LOSE_AT`).
    pub fn lose(&self) {
        lock(&self.0.errors)
            .push("the GPU failed command buffer `frame`: the device was lost: simulated (WRELA_GPU_LOSE_AT)".into());
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What a failed command buffer says: its error, and each of its encoders (their labels are the
/// passes') with how far it got: done, faulted, hit by the fault (affected), not begun (pending),
/// or not known. A hang names no faulted encoder; the one it was in is among the affected.
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
                    let state = item.errorState();
                    if state == MTLCommandEncoderErrorState::Faulted {
                        out.push_str(&format!("; it faulted in `{}`", item.label()));
                    } else {
                        let what = match state {
                            MTLCommandEncoderErrorState::Completed => "done",
                            MTLCommandEncoderErrorState::Affected => "affected",
                            MTLCommandEncoderErrorState::Pending => "not begun",
                            _ => "not known",
                        };
                        out.push_str(&format!("; `{}` {what}", item.label()));
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
            PendingWrite::Buffer { staging, staging_offset, dst, offset, len } => {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    staging,
                    *staging_offset as NSUInteger,
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

    /// Whether it and `other` are the same buffer (clones of one).
    pub fn same(&self, other: &Buffer) -> bool {
        std::ptr::eq(&*self.raw, &*other.raw)
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

#[derive(Clone)]
pub struct Texture {
    /// The device's table, where each view of it gets an index.
    table: Arc<Table>,
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
        let slot = Some(self.table.take(raw.gpuResourceID(), Held::Texture(raw.clone())));
        TextureView { raw, slot }
    }
}

#[derive(Clone)]
pub struct TextureView {
    raw: Id<dyn MTLTexture>,
    /// Its index in the device's table (none for a frame of a surface, only drawn into).
    slot: Option<Arc<Slot>>,
}

#[derive(Clone)]
pub struct Sampler {
    raw: Id<dyn MTLSamplerState>,
    slot: Arc<Slot>,
}

#[derive(Clone)]
pub struct QuerySet {
    raw: Id<dyn MTLCounterSampleBuffer>,
    count: u32,
}

impl QuerySet {
    pub fn count(&self) -> u32 {
        self.count
    }
}

#[derive(Clone)]
pub struct ShaderModule {
    library: Id<dyn MTLLibrary>,
}

#[derive(Clone)]
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

#[derive(Clone)]
pub struct RenderPipeline {
    raw: Id<dyn MTLRenderPipelineState>,
    depth_state: Option<Id<dyn MTLDepthStencilState>>,
    cull: MTLCullMode,
    bias: DepthBias,
}

// ---- bindings -----------------------------------------------------------------------------------------

/// A layout's entries, and each one's slot in each stage's table (`wrela_abi::metal::slots`'
/// rule: a stage's own counter for each table, in the entries' order, counting only what the
/// stage sees; an entry the device's table holds takes none).
struct Layout {
    entries: Vec<LayoutEntry>,
    /// Each entry's slot in the vertex, fragment and compute stage, where it sees it.
    slots: Vec<[Option<u8>; 3]>,
    /// The entry that takes a dynamic offset.
    dynamic: Vec<usize>,
    /// The bytes of its block of indices: past the last entry's index.
    block: usize,
}

impl Layout {
    fn of(entries: &[LayoutEntry]) -> Layout {
        let mut counters = [[0u8; 3]; 3];
        let mut slots = Vec::with_capacity(entries.len());
        let mut dynamic = Vec::new();
        let mut block = 0;
        for (k, e) in entries.iter().enumerate() {
            let table = e.ty.table() as usize;
            let mut s = [None; 3];
            if let Some(at) = e.index_at {
                block = block.max(at as usize + 4);
                slots.push(s);
                continue;
            }
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
            if e.ty.dynamic() {
                dynamic.push(k);
            }
        }
        Layout { entries: entries.to_vec(), slots, dynamic, block }
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
    /// Its block of indices in the device's table (empty where the table holds none of it).
    block: Vec<u8>,
    /// What it reaches through the table, as Metal declares it: textures read, and written.
    reads: Vec<Id<dyn MTLTexture>>,
    writes: Vec<Id<dyn MTLTexture>>,
    /// The indices it names, alive while it is.
    _slots: Vec<Arc<Slot>>,
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

/// What group `g` reaches through the table, as `useResources` takes it: the textures it reads,
/// and those it writes.
#[allow(clippy::type_complexity)]
fn declared(
    g: &Group,
) -> (Vec<NonNull<ProtocolObject<dyn MTLResource>>>, Vec<NonNull<ProtocolObject<dyn MTLResource>>>)
{
    let each = |ts: &[Id<dyn MTLTexture>]| {
        ts.iter()
            .map(|t| {
                let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**t);
                NonNull::from(r)
            })
            .collect()
    };
    (each(&g.reads), each(&g.writes))
}

// ---- recording ----------------------------------------------------------------------------------------

/// A command buffer being recorded.
pub struct CommandEncoder {
    device: Device,
    label: String,
    cb: Option<Id<dyn MTLCommandBuffer>>,
    /// The blit encoder copies go into, while copies follow one another.
    blit: Option<Id<dyn MTLBlitCommandEncoder>>,
    /// The bind groups its commands set, which what they reach through the table needs alive.
    keep: Vec<Arc<Group>>,
}

pub struct CommandBuffer {
    raw: Id<dyn MTLCommandBuffer>,
    keep: Vec<Arc<Group>>,
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
        // An empty copy is nothing (WebGPU allows one; Vulkan doesn't).
        if size == 0 {
            return;
        }
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
        if extent.contains(&0) {
            return;
        }
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
        if extent.contains(&0) {
            return;
        }
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
        if extent.contains(&0) {
            return;
        }
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
        let table = self.device.0.table.clone();
        ComputePass { encoder, pipeline: None, keep: &mut self.keep, table, table_set: false }
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
        let table = self.device.0.table.clone();
        RenderPass { encoder, index: None, keep: &mut self.keep, table, table_set: false }
    }

    pub fn finish(mut self) -> CommandBuffer {
        self.end_blit();
        let keep = std::mem::take(&mut self.keep);
        CommandBuffer { raw: self.cb().clone(), keep }
    }
}

/// An encoder dropped unfinished (its recording failed part way) ends its open blit encoder:
/// Metal asserts on an encoder released without `endEncoding`, which ended the whole process
/// (the GPU tests' run stopped at the first test whose recording failed after a copy).
impl Drop for CommandEncoder {
    fn drop(&mut self) {
        self.end_blit();
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
    keep: &'a mut Vec<Arc<Group>>,
    table: Arc<Table>,
    /// Whether the table is bound in this encoder yet.
    table_set: bool,
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
        let g = &group.0;
        if g.block.is_empty() {
            return;
        }
        // SAFETY: the block's bytes are copied as the call is made; the table, and what the
        // group names in it, are alive until the command buffer is done (`keep`).
        unsafe {
            if !self.table_set {
                let at = wrela_abi::table::METAL_TABLE as NSUInteger;
                e.setBuffer_offset_atIndex(Some(&self.table.buffer), 0, at);
                self.table_set = true;
            }
            let at = wrela_abi::table::METAL_BLOCK as NSUInteger;
            e.setBytes_length_atIndex(NonNull::from(&g.block[..]).cast(), g.block.len(), at);
            let (reads, writes) = declared(g);
            if !reads.is_empty() {
                e.useResources_count_usage(
                    NonNull::from(&reads[..]).cast(),
                    reads.len() as NSUInteger,
                    MTLResourceUsage::Read,
                );
            }
            if !writes.is_empty() {
                e.useResources_count_usage(
                    NonNull::from(&writes[..]).cast(),
                    writes.len() as NSUInteger,
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
            }
        }
        self.keep.push(g.clone());
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
    keep: &'a mut Vec<Arc<Group>>,
    table: Arc<Table>,
    /// Whether the table is bound in this encoder's stages yet.
    table_set: bool,
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
        let g = &group.0;
        if g.block.is_empty() {
            return;
        }
        let stages = MTLRenderStages::Vertex | MTLRenderStages::Fragment;
        // SAFETY: as `ComputePass::set_bind_group`'s.
        unsafe {
            if !self.table_set {
                let at = wrela_abi::table::METAL_TABLE as NSUInteger;
                e.setVertexBuffer_offset_atIndex(Some(&self.table.buffer), 0, at);
                e.setFragmentBuffer_offset_atIndex(Some(&self.table.buffer), 0, at);
                self.table_set = true;
            }
            let at = wrela_abi::table::METAL_BLOCK as NSUInteger;
            let bytes = NonNull::from(&g.block[..]).cast();
            e.setVertexBytes_length_atIndex(bytes, g.block.len(), at);
            e.setFragmentBytes_length_atIndex(bytes, g.block.len(), at);
            let (reads, writes) = declared(g);
            if !reads.is_empty() {
                e.useResources_count_usage_stages(
                    NonNull::from(&reads[..]).cast(),
                    reads.len() as NSUInteger,
                    MTLResourceUsage::Read,
                    stages,
                );
            }
            if !writes.is_empty() {
                e.useResources_count_usage_stages(
                    NonNull::from(&writes[..]).cast(),
                    writes.len() as NSUInteger,
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                    stages,
                );
            }
        }
        self.keep.push(g.clone());
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

// ---- a window's surface -------------------------------------------------------------------------

/// A window's `CAMetalLayer`, which Metal draws a frame into and the window shows. Its frames are
/// half floats (`RGBA16Float`), which an extended-range display shows past white, and which hold
/// the screen's 8-bit values exactly.
pub struct Surface {
    layer: objc2::rc::Retained<objc2_quartz_core::CAMetalLayer>,
    /// The window's view: its window's screen says the display's headroom.
    view: objc2::rc::Retained<objc2_app_kit::NSView>,
    /// The window, alive while its layer is drawn into.
    _window: Arc<dyn Window>,
    /// When each frame presented reached the display, by its id ([`Surface::present`]), until
    /// taken; and the next id.
    presented: Arc<Mutex<Vec<(u64, std::time::Instant)>>>,
    next_id: u64,
    /// The device's table, which a frame's texture carries as every texture does.
    table: Arc<Table>,
}

// SAFETY: the layer is made and used on the thread that runs the window (macOS's main thread);
// the host keeps its surface there.
unsafe impl Send for Surface {}

/// A frame of a surface to draw into, then [`Surface::present`].
pub struct SurfaceTexture {
    drawable: objc2::rc::Retained<ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>>,
    pub texture: Texture,
    pub view: TextureView,
}

impl Device {
    /// A surface of `window`, which must be an AppKit window's, made on the main thread.
    pub fn create_surface(&self, window: Arc<dyn Window>) -> Result<Surface> {
        use objc2_quartz_core::CAMetalLayer;
        use raw_window_handle::RawWindowHandle;
        let mtm = objc2::MainThreadMarker::new()
            .ok_or("a window's Metal layer is made on the main thread")?;
        let _ = mtm;
        let handle = window.window_handle().map_err(|e| Error(format!("the window: {e}")))?;
        let RawWindowHandle::AppKit(h) = handle.as_raw() else {
            return Err("Metal draws only into an AppKit window".into());
        };
        // SAFETY: the handle's view is the window's, alive while the window is (the surface keeps
        // it).
        let view: &objc2_app_kit::NSView = unsafe { h.ns_view.cast().as_ref() };
        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&self.0.raw));
        layer.setPixelFormat(MTLPixelFormat::RGBA16Float);
        layer.setFramebufferOnly(true);
        view.setWantsLayer(true);
        view.setLayer(Some(&layer));
        let view = objc2::Message::retain(view);
        let mut surface = Surface {
            layer,
            view,
            _window: window,
            presented: Arc::new(Mutex::new(Vec::new())),
            next_id: 0,
            table: self.0.table.clone(),
        };
        surface.set_space(false, false);
        Ok(surface)
    }

    /// Submits `buffers` for `frame`: on Metal, as any other submission.
    pub fn submit_frame(
        &self,
        buffers: Vec<CommandBuffer>,
        frame: &SurfaceTexture,
    ) -> SubmissionIndex {
        let _ = frame;
        self.submit(buffers)
    }
}

impl Surface {
    /// Its frames' format.
    pub fn format(&self) -> TextureFormat {
        TextureFormat::Rgba16Float
    }

    pub fn configure(&mut self, device: &Device, c: &SurfaceConfig) -> Result<()> {
        let _ = device;
        self.layer.setDrawableSize(objc2_core_foundation::CGSize {
            width: f64::from(c.width),
            height: f64::from(c.height),
        });
        self.layer.setMaximumDrawableCount(c.frames.clamp(2, 3) as NSUInteger);
        // Every mode waits for the display's refresh (Fifo): twice, a window run that turned
        // display sync off (`immediate`, once in full screen, once in a window) was followed by
        // WindowServer hanging and the Mac needing a restart (2026-11-16). Pacing, not tearing,
        // is how a frame gets to the display sooner.
        let _ = c.present;
        self.layer.setDisplaySyncEnabled(true);
        Ok(())
    }

    /// Shows its frames as `hdr` (linear values, 1.0 SDR white, brighter where the display has
    /// headroom: EDR) or not (sRGB's or Display P3's encoded values), in Display P3 (`p3`) or
    /// sRGB: what it does of that, as `(hdr, p3)` (Metal does all of it).
    pub fn set_space(&mut self, hdr: bool, p3: bool) -> (bool, bool) {
        use objc2_core_graphics::{
            CGColorSpace, kCGColorSpaceDisplayP3, kCGColorSpaceExtendedLinearDisplayP3,
            kCGColorSpaceExtendedLinearSRGB, kCGColorSpaceSRGB,
        };
        // SAFETY: the names are CoreGraphics' own constants.
        let name = unsafe {
            match (hdr, p3) {
                (true, true) => kCGColorSpaceExtendedLinearDisplayP3,
                (true, false) => kCGColorSpaceExtendedLinearSRGB,
                (false, true) => kCGColorSpaceDisplayP3,
                (false, false) => kCGColorSpaceSRGB,
            }
        };
        let space = CGColorSpace::with_name(Some(name));
        self.layer.setColorspace(space.as_deref());
        self.layer.setWantsExtendedDynamicRangeContent(hdr);
        (hdr, p3)
    }

    /// The display's headroom now: the most a value may be over SDR white and still be brighter
    /// (its screen's `maximumExtendedDynamicRangeColorComponentValue`; 1 where it has none).
    pub fn headroom(&self) -> f32 {
        let Some(mtm) = objc2::MainThreadMarker::new() else { return 1.0 };
        let _ = mtm;
        self.view
            .window()
            .and_then(|w| w.screen())
            .map_or(1.0, |s| s.maximumExtendedDynamicRangeColorComponentValue() as f32)
    }

    /// The next frame to draw into (waits for one the display is done with).
    pub fn acquire(&mut self) -> Result<SurfaceTexture> {
        use objc2_quartz_core::CAMetalDrawable;
        let drawable = self.layer.nextDrawable().ok_or("the layer gave no drawable")?;
        let raw = drawable.texture();
        let (width, height) = (raw.width() as u32, raw.height() as u32);
        let texture = Texture {
            table: self.table.clone(),
            raw: raw.clone(),
            format: TextureFormat::Rgba16Float,
            width,
            height,
            depth_or_layers: 1,
            levels: 1,
            dimension: TextureDimension::D2,
        };
        Ok(SurfaceTexture { drawable, texture, view: TextureView { raw, slot: None } })
    }

    /// Shows `frame` once the work submitted for it is done: its id, which
    /// [`Surface::take_presented`] gives with the time it reached the display.
    pub fn present(&mut self, frame: SurfaceTexture) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        let sink = self.presented.clone();
        // Metal says when the frame was shown (`presentedTime`, Core Animation's clock), or 0 if
        // it never was (a frame replaced before its refresh).
        let block = RcBlock::new(move |d: NonNull<ProtocolObject<dyn MTLDrawable>>| {
            // SAFETY: Metal passes the drawable that was presented.
            let shown = unsafe { d.as_ref() }.presentedTime();
            if shown > 0.0 {
                let ago = (objc2_quartz_core::CACurrentMediaTime() - shown).max(0.0);
                let at = std::time::Instant::now() - std::time::Duration::from_secs_f64(ago);
                lock(&sink).push((id, at));
            }
        });
        // SAFETY: a handler of the drawable's own, which Metal copies.
        unsafe { frame.drawable.addPresentedHandler(RcBlock::as_ptr(&block)) };
        frame.drawable.present();
        Ok(id)
    }

    /// The frames shown since the last take: each id and when it reached the display.
    pub fn take_presented(&mut self) -> Vec<(u64, std::time::Instant)> {
        std::mem::take(&mut *lock(&self.presented))
    }
}
