//! The native host's GPU, straight on the machine's own API: Metal on Apple's machines
//! ([`metal`], objc2-metal), Vulkan everywhere else ([`vulkan`], ash). One small API in front of
//! both, cut to what the host does (`wrela_host`'s `gpu.rs`): a device and its limits; buffers,
//! textures and samplers; bind groups laid out as the compiler binds them
//! ([`wrela_abi::metal`], [`wrela_abi::vulkan`]); compute and render pipelines from the
//! compiler's own shaders (MSL text, SPIR-V words); command encoders with compute and render
//! passes, copies and timestamps; submissions, their completion, and reads back.
//!
//! Its shape is wgpu's, which the host ran on before (toolchain 0.33's backend change), so the
//! host's logic moved over unchanged: queue writes are staged and land before the next
//! submission, as wgpu's do; a render pass's draws are recorded where it ends. What wgpu decided
//! for the host (hazards and barriers, memory, the binding model) is decided here, and said:
//!
//! - **The table.** Every texture view and sampler has an index in the device's one table
//!   (`wrela_abi::table`), given when it's made. A bind group's textures and samplers that its
//!   layout says the table holds are not bound: their indices go into a block of bytes the
//!   group sets with its buffers (push constants, `setBytes`), and the group says what they are
//!   for the hazards between commands (and, on Metal, their residency).
//! - **Hazards.** Metal's resources are tracked: Metal orders what reads after what writes.
//!   Vulkan's aren't: a tracker of each buffer's and each texture subresource's last use
//!   inserts the barriers (and image layouts) each command needs.
//! - **Memory.** Both machines are unified: a mapped buffer is the GPU's own, read in place once
//!   its submission is done.
//!
//! The one crate that may use `unsafe`: every call into the GPU's API is here.

use std::fmt;

pub use wrela_abi::manifest::{Blend, Cull, DepthBias, DepthState};
pub use wrela_abi::stream::{Compare, TextureFormat};

#[cfg(target_vendor = "apple")]
pub mod metal;
#[cfg(target_vendor = "apple")]
pub use metal as backend;

#[cfg(not(target_vendor = "apple"))]
pub mod vulkan;
#[cfg(not(target_vendor = "apple"))]
pub use vulkan as backend;

pub use backend::{
    BindGroup, BindGroupLayout, Buffer, CommandBuffer, CommandEncoder, ComputePass,
    ComputePipeline, Device, QuerySet, RenderPass, RenderPipeline, Sampler, ShaderModule, Surface,
    SurfaceTexture, Texture, TextureView,
};

/// How many of the last submissions' GPU times a device keeps (`Device::gpu_ms`).
pub const CLOCK_KEPT: usize = 4096;

/// How a surface's frames reach the display.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// On Metal every mode presents as `Fifo`: turning display sync off was followed twice by
/// WindowServer hanging (see `metal.rs`).
pub enum PresentMode {
    /// In order, one each refresh: no tearing, a frame may wait for a refresh (vsync).
    Fifo,
    /// As `Fifo`, but a late frame goes at once rather than waiting a refresh (it may tear).
    FifoRelaxed,
    /// The newest frame done at each refresh; the rest are dropped: no tearing, no waiting.
    Mailbox,
    /// At once (it may tear): the least latency.
    Immediate,
}

impl PresentMode {
    pub fn parse(s: &str) -> Option<PresentMode> {
        Some(match s {
            "fifo" => PresentMode::Fifo,
            "fifo-relaxed" => PresentMode::FifoRelaxed,
            "mailbox" => PresentMode::Mailbox,
            "immediate" => PresentMode::Immediate,
            _ => return None,
        })
    }
}

/// A surface's swapchain: its size, how it presents, and how many frames it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceConfig {
    pub width: u32,
    pub height: u32,
    pub present: PresentMode,
    /// Frames in the chain: 2 or 3.
    pub frames: u32,
}

/// A window a surface draws into: its handles, alive as long as the surface.
pub trait Window:
    raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle + Send + Sync
{
}

impl<T: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle + Send + Sync>
    Window for T
{
}

/// What went wrong: a message for the host's error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(s: String) -> Error {
        Error(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Error {
        Error(s.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// The screen's format: `wrela_abi::manifest::SCREEN_FORMAT`, rgba8unorm.
pub const SCREEN_FORMAT: TextureFormat = TextureFormat::Rgba8;

/// The alignment of a buffer-to-texture copy's rows, in bytes (WebGPU's, which the host pads
/// its uploads to: both APIs take it).
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// The alignment of a buffer copy's offsets and sizes, in bytes.
pub const COPY_BUFFER_ALIGNMENT: u64 = 4;

/// What kind of GPU a device is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceType {
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    /// A GPU emulated on the CPU (Mesa's llvmpipe): run on only where allowed.
    Cpu,
    Other,
}

/// Which API a device is driven through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Metal,
    Vulkan,
}

/// What a device says of itself.
#[derive(Clone, Debug)]
pub struct AdapterInfo {
    pub name: String,
    pub backend: Backend,
    pub device_type: DeviceType,
    /// PCI vendor and device IDs (0 where the API gives none).
    pub vendor: u32,
    pub device: u32,
    pub driver: String,
    pub driver_info: String,
    /// The least and most invocations a subgroup has.
    pub subgroup_min_size: u32,
    pub subgroup_max_size: u32,
}

/// A device's limits, by WebGPU's names (what the program sees, `wrela_abi::Limits`, comes from
/// these).
#[derive(Clone, Debug, PartialEq)]
pub struct Limits {
    pub max_texture_dimension_2d: u32,
    pub max_texture_dimension_3d: u32,
    pub max_texture_array_layers: u32,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_uniform_buffer_binding_size: u64,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_sampled_textures_per_shader_stage: u32,
    pub max_samplers_per_shader_stage: u32,
    pub max_storage_textures_per_shader_stage: u32,
    pub max_color_attachments: u32,
    pub max_color_attachment_bytes_per_sample: u32,
    pub min_uniform_buffer_offset_alignment: u32,
    pub min_storage_buffer_offset_alignment: u32,
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size: [u32; 3],
    pub max_compute_workgroups_per_dimension: u32,
}

impl Limits {
    /// WebGPU's default limits (the WebGPU specification's limits table): what a browser's
    /// device has unless it asks for more.
    pub fn webgpu_defaults() -> Limits {
        Limits {
            max_texture_dimension_2d: 8192,
            max_texture_dimension_3d: 2048,
            max_texture_array_layers: 256,
            max_buffer_size: 256 << 20,
            max_storage_buffer_binding_size: 128 << 20,
            max_uniform_buffer_binding_size: 64 << 10,
            max_storage_buffers_per_shader_stage: 8,
            max_sampled_textures_per_shader_stage: 16,
            max_samplers_per_shader_stage: 16,
            max_storage_textures_per_shader_stage: 4,
            max_color_attachments: 8,
            max_color_attachment_bytes_per_sample: 32,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 256,
            max_compute_workgroup_storage_size: 16384,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size: [256, 256, 64],
            max_compute_workgroups_per_dimension: 65535,
        }
    }
}

/// The optional features a device has, by WebGPU's names (`wrela_abi::FEATURES`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Features(pub Vec<&'static str>);

impl Features {
    pub fn has(&self, name: &str) -> bool {
        self.0.contains(&name)
    }

    /// These, but only those `keep` says.
    pub fn filtered(&self, keep: impl Fn(&str) -> bool) -> Features {
        Features(self.0.iter().copied().filter(|f| keep(f)).collect())
    }
}

/// How a device is opened.
#[derive(Clone, Debug, Default)]
pub struct DeviceOptions {
    /// A label, for captures and validation messages.
    pub label: String,
    /// Run on a software device (a GPU on the CPU) if there's no hardware one.
    pub allow_software: bool,
    /// The optional features to open it with where it has them (all of them, unless masked).
    pub features: Option<Vec<String>>,
}

/// A buffer: `size` bytes, read and written by the GPU (a storage, uniform, index or indirect
/// buffer, a copy's source or destination), or `mappable`: a copy's destination the CPU reads.
#[derive(Clone, Debug)]
pub struct BufferDesc<'a> {
    pub label: &'a str,
    pub size: u64,
    pub mappable: bool,
}

/// A texture's shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureDimension {
    /// 2D: `depth_or_layers` layers (6 for a cube).
    D2,
    D3,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TextureUsage {
    pub sampled: bool,
    pub storage: bool,
    pub render: bool,
    pub copy_src: bool,
    pub copy_dst: bool,
}

#[derive(Clone, Debug)]
pub struct TextureDesc<'a> {
    pub label: &'a str,
    pub width: u32,
    pub height: u32,
    pub depth_or_layers: u32,
    pub levels: u32,
    pub dimension: TextureDimension,
    pub format: TextureFormat,
    pub usage: TextureUsage,
    /// Samples per pixel: `1` (normal) or `4` (multisample attachment).
    pub samples: u8,
    /// Attachment-only memoryless/transient backing (never read or stored across passes).
    pub transient: bool,
}

/// How a view sees a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewDimension {
    D2,
    D2Array,
    Cube,
    D3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewDesc {
    pub dimension: ViewDimension,
    pub base_level: u32,
    /// `None`: every level from `base_level` on.
    pub level_count: Option<u32>,
    pub base_layer: u32,
    /// `None`: every layer from `base_layer` on.
    pub layer_count: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerDesc {
    pub linear: bool,
    pub repeat: bool,
    pub mip_linear: bool,
    /// 1 to 16.
    pub anisotropy: u16,
    pub compare: Option<Compare>,
}

/// The shader stages that see a binding.
pub use wrela_abi::metal::Stages;

/// How a shader reads a sampled texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleType {
    Float { filterable: bool },
    Uint,
    Depth,
}

/// What a bind group layout's entry binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BindingType {
    /// A uniform buffer: at a dynamic offset (given when the group is set) or not.
    Uniform {
        dynamic: bool,
        size: u64,
    },
    /// A storage buffer: at a dynamic offset, `size` bytes of it (a uniform block held in
    /// storage), or not.
    Storage {
        read_only: bool,
        dynamic: bool,
        size: u64,
    },
    Texture {
        sample_type: SampleType,
        dimension: ViewDimension,
    },
    /// A texture a kernel writes texels of.
    StorageTexture {
        format: TextureFormat,
        dimension: ViewDimension,
    },
    Sampler {
        comparison: bool,
        filtering: bool,
    },
}

impl BindingType {
    /// Whether it's bound at a dynamic offset, given when its group is set.
    pub fn dynamic(self) -> bool {
        matches!(
            self,
            BindingType::Uniform { dynamic: true, .. } | BindingType::Storage { dynamic: true, .. }
        )
    }

    /// Which of a Metal stage's argument tables it takes a slot of: 0 buffers, 1 textures, 2
    /// samplers (`wrela_abi::metal::Table`'s order).
    pub fn table(self) -> u8 {
        match self {
            BindingType::Uniform { .. } | BindingType::Storage { .. } => 0,
            BindingType::Texture { .. } | BindingType::StorageTexture { .. } => 1,
            BindingType::Sampler { .. } => 2,
        }
    }
}

/// An entry of bind group 0's layout: the pipeline's binding number, the stages that see it, and
/// what it is. A layout's entries come in ascending binding order: Vulkan numbers each by its
/// place among them (`wrela_abi::vulkan::descriptor`), Metal gives each a slot of its table in
/// each stage that sees it (`wrela_abi::metal::slots`). A texture or sampler reached through the
/// device's table (`wrela_abi::table`) takes no slot or descriptor: its index goes into the
/// group's block at `index_at`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayoutEntry {
    pub binding: u32,
    pub visibility: Stages,
    pub ty: BindingType,
    /// The byte of the block of indices that holds its index in the device's table, where the
    /// table holds it.
    pub index_at: Option<u32>,
}

/// What a bind group binds at one of its layout's bindings.
#[derive(Clone)]
pub enum Resource<'a> {
    /// `size` bytes from `offset` (`None`: to the buffer's end).
    Buffer {
        buffer: &'a Buffer,
        offset: u64,
        size: Option<u64>,
    },
    View(&'a TextureView),
    Sampler(&'a Sampler),
}

#[derive(Clone)]
pub struct BindGroupEntry<'a> {
    pub binding: u32,
    pub resource: Resource<'a>,
}

/// A compute pipeline: its shader's entry point, its workgroup's size and its layout.
pub struct ComputePipelineDesc<'a> {
    pub label: &'a str,
    pub layout: &'a BindGroupLayout,
    pub module: &'a ShaderModule,
    pub entry: &'a str,
    pub workgroup_size: [u32; 3],
}

/// A colour target of a render pipeline: its format and how a fragment blends into it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorTarget {
    pub format: TextureFormat,
    pub blend: Blend,
}

/// A render pipeline's depth: its target's format, its test and write, and its bias.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthDesc {
    pub format: TextureFormat,
    pub state: DepthState,
    pub bias: DepthBias,
}

/// A render pipeline: triangle lists, counter-clockwise front faces (WebGPU's defaults), no
/// vertex buffers (shaders pull their vertices).
pub struct RenderPipelineDesc<'a> {
    pub label: &'a str,
    pub layout: &'a BindGroupLayout,
    pub module: &'a ShaderModule,
    pub vertex_entry: &'a str,
    pub fragment_entry: &'a str,
    pub colors: &'a [ColorTarget],
    pub depth: Option<DepthDesc>,
    pub cull: Cull,
    /// Samples per pixel: `1` or `4`.
    pub samples: u8,
}

/// A copy's place in a buffer: rows `bytes_per_row` apart from `offset`, `rows_per_image` of
/// them a layer.
#[derive(Clone, Copy)]
pub struct BufferCopy<'a> {
    pub buffer: &'a Buffer,
    pub offset: u64,
    pub bytes_per_row: u32,
    pub rows_per_image: u32,
}

/// A copy's place in a texture: a level, a layer (or a 3D texture's slice) and a corner.
#[derive(Clone, Copy)]
pub struct TextureCopy<'a> {
    pub texture: &'a Texture,
    pub level: u32,
    pub origin: [u32; 3],
    /// The depth of a depth-and-stencil format (no format here has stencil: always its whole).
    pub depth_only: bool,
}

/// What a pass does with a target as it begins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadOp<T> {
    Load,
    Clear(T),
}

pub struct ColorAttachment<'a> {
    pub view: &'a TextureView,
    pub resolve: Option<&'a TextureView>,
    pub load: LoadOp<[f64; 4]>,
    pub store: bool,
}

pub struct DepthAttachment<'a> {
    pub view: &'a TextureView,
    pub resolve: Option<&'a TextureView>,
    pub load: LoadOp<f32>,
    pub store: bool,
}

/// Timestamps written as a pass begins and ends, at two indices of a query set.
#[derive(Clone, Copy)]
pub struct PassTimestamps<'a> {
    pub queries: &'a QuerySet,
    pub begin: u32,
    pub end: u32,
}

pub struct RenderPassDesc<'a> {
    pub label: &'a str,
    pub colors: &'a [ColorAttachment<'a>],
    pub depth: Option<DepthAttachment<'a>>,
    pub timestamps: Option<PassTimestamps<'a>>,
}

/// Each submission's place in the device's order: its work is done once the device's
/// completed count is past it.
pub type SubmissionIndex = u64;
