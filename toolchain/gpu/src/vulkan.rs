//! The GPU through Vulkan 1.3 (ash), for every machine but Apple's: RADV on the Linux machine.
//!
//! What wgpu decided for the host is decided here, and kept simple until a measurement says
//! otherwise:
//!
//! - **Submissions** are counted by one timeline semaphore: submission `n` signals `n`.
//! - **Memory**: one allocation a resource, from the first memory type that fits (device-local
//!   for the GPU's own, host-visible and coherent for what the CPU maps). Every resource starts
//!   zeroed, as WebGPU's do: a fill or a clear staged before the next submission's own work.
//!   Staged writes go through a belt of host-visible buffers reused once their submission is done.
//! - **Images** rest in a home layout by their use (`GENERAL` where a kernel writes one, else
//!   read-only optimal), and a pass or a copy moves what it uses out and back: in `GENERAL` (or
//!   made `MUTABLE_FORMAT`) RADV keeps no compression of a target, which cost the opening 10%.
//! - **Hazards**: a dispatch or a copy waits on a full memory barrier only when it reads or
//!   writes what a command since the last barrier wrote, or writes what one read (each
//!   encoder's sets of handles); a pass begins and ends with one.
//! - **Bindings**: set 0, each layout entry numbered by its place among the layout's entries
//!   (`wrela_abi::vulkan::descriptor`), a uniform at a dynamic offset as
//!   `UNIFORM_BUFFER_DYNAMIC`.
//! - **Validation**: `WRELA_VULKAN_VALIDATION=1` turns on Khronos's layer, and every message it
//!   reports is an error the host reads after the frame (`take_errors`).

use crate::*;
use ash::vk;
use std::ffi::{CStr, CString, c_void};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ---- the device ---------------------------------------------------------------------------------

struct Shared {
    entry: ash::Entry,
    /// Whether it can present to a window (the surface and swapchain extensions are on).
    swapchain: bool,
    instance: ash::Instance,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    physical: vk::PhysicalDevice,
    raw: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    memory: vk::PhysicalDeviceMemoryProperties,
    info: AdapterInfo,
    limits: Limits,
    features: Features,
    timestamp_period: f32,
    /// The timeline semaphore: submission `n` signals `n`.
    timeline: vk::Semaphore,
    submitted: AtomicU64,
    /// Staged writes, recorded before the next submission's own work.
    pending: Mutex<Vec<Pending>>,
    /// Command pools and buffers of submissions not yet seen done, with what they hold alive.
    in_flight: Mutex<Vec<(SubmissionIndex, vk::CommandPool, Vec<Keep>)>>,
    /// Errors: the validation layer's messages, a lost device.
    errors: Arc<Mutex<Vec<String>>>,
    /// Pipelines and layouts made for each bind group layout, freed with the device.
    descriptor_pools: Mutex<Vec<vk::DescriptorPool>>,
    /// Host-visible buffers staged writes are copied into, reused once their submission is done.
    belt: Mutex<Belt>,
}

/// The staging belt: the buffer this submission's writes go into (and how much of it they
/// fill), the full ones this submission also uses, and those earlier submissions use or used.
#[derive(Default)]
struct Belt {
    current: Option<(Arc<RawBuffer>, u64)>,
    filled: Vec<Arc<RawBuffer>>,
    in_flight: Vec<(SubmissionIndex, Arc<RawBuffer>)>,
    free: Vec<Arc<RawBuffer>>,
}

/// The least a staging buffer holds.
const BELT_CHUNK: u64 = 4 << 20;

// SAFETY: Vulkan's handles may be used from any thread when externally synchronized; the queue
// and the pending list are behind mutexes, and the host records on one thread at a time.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// What a submission holds alive until it's done: what its commands use (held, never read).
#[allow(dead_code)]
enum Keep {
    Buffer(Arc<RawBuffer>),
    Group(Arc<Group>),
    Pipeline(Arc<RawPipeline>),
    Held(Held),
    Queries(Arc<RawQueries>),
    Image(Arc<RawImage>),
}

enum Pending {
    /// Bytes from a staging buffer (at `at`) into a buffer.
    Buffer { staging: Arc<RawBuffer>, at: u64, dst: Arc<RawBuffer>, offset: u64, len: u64 },
    /// Zeros over a whole buffer.
    Fill { dst: Arc<RawBuffer> },
    /// Texels from a staging buffer into a level and layer of an image.
    Texture {
        staging: Arc<RawBuffer>,
        at: u64,
        bytes_per_row: u32,
        rows_per_image: u32,
        dst: Arc<RawImage>,
        level: u32,
        layer: u32,
        origin: [u32; 3],
        extent: [u32; 3],
    },
    /// An image cleared to zero (every level and layer), and put in `GENERAL`.
    Clear { dst: Arc<RawImage> },
}

/// The GPU: a Vulkan device and its one queue.
#[derive(Clone)]
pub struct Device(Arc<Shared>);

unsafe extern "system" fn on_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        || severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
    {
        // SAFETY: the layer passes its message, and `user` is the errors list it was given,
        // which the device keeps alive while the messenger exists.
        let message = unsafe {
            let d = &*data;
            if d.p_message.is_null() {
                String::from("(no message)")
            } else {
                CStr::from_ptr(d.p_message).to_string_lossy().into_owned()
            }
        };
        let errors = unsafe { &*(user as *const Mutex<Vec<String>>) };
        lock(errors).push(format!("Vulkan validation: {message}"));
    }
    vk::FALSE
}

impl Device {
    /// The machine's GPU: a discrete one first, then an integrated one; a GPU on the CPU only if
    /// `allow_software`.
    pub fn open(options: &DeviceOptions) -> Result<Device> {
        // SAFETY: every call below is Vulkan's, with structures built here and kept alive past
        // the calls that read them.
        unsafe {
            let entry = ash::Entry::load().map_err(|e| Error(format!("no Vulkan loader: {e}")))?;
            let validate = std::env::var("WRELA_VULKAN_VALIDATION").is_ok_and(|v| v == "1");
            let app_name = CString::new("wrela").expect("a name");
            let app = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .engine_name(&app_name)
                .api_version(vk::API_VERSION_1_3);
            let layer = CString::new("VK_LAYER_KHRONOS_validation").expect("a name");
            let layers: Vec<*const i8> = if validate { vec![layer.as_ptr()] } else { Vec::new() };
            let mut extensions: Vec<*const i8> =
                if validate { vec![ash::ext::debug_utils::NAME.as_ptr()] } else { Vec::new() };
            // A window's surface, wherever the loader has its kind (a headless run uses none).
            let have = entry.enumerate_instance_extension_properties(None).unwrap_or_default();
            let has = |name: &CStr| have.iter().any(|e| e.extension_name_as_c_str() == Ok(name));
            let surfaces = [
                ash::khr::surface::NAME,
                ash::khr::wayland_surface::NAME,
                ash::khr::xlib_surface::NAME,
                ash::khr::xcb_surface::NAME,
            ];
            let windowed = has(ash::khr::surface::NAME);
            if windowed {
                extensions.extend(surfaces.iter().filter(|n| has(n)).map(|n| n.as_ptr()));
            }
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default()
                        .application_info(&app)
                        .enabled_layer_names(&layers)
                        .enabled_extension_names(&extensions),
                    None,
                )
                .map_err(|e| Error(format!("Vulkan made no instance: {e}")))?;
            let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
            let debug = if validate {
                let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(on_message))
                    .user_data(Arc::as_ptr(&errors) as *mut c_void);
                let m = du
                    .create_debug_utils_messenger(&info, None)
                    .map_err(|e| Error(format!("no validation messenger: {e}")))?;
                Some((du, m))
            } else {
                None
            };
            let devices = instance
                .enumerate_physical_devices()
                .map_err(|e| Error(format!("Vulkan lists no GPU: {e}")))?;
            let rank = |p: &vk::PhysicalDevice| match instance
                .get_physical_device_properties(*p)
                .device_type
            {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU if options.allow_software => 3,
                _ => 9,
            };
            let mut found: Vec<String> = Vec::new();
            for p in &devices {
                let props = instance.get_physical_device_properties(*p);
                found.push(cstr(&props.device_name));
            }
            let physical =
                devices.iter().copied().filter(|p| rank(p) < 9).min_by_key(rank).ok_or_else(
                    || Error(format!("no hardware GPU through Vulkan: found {}", found.join(", "))),
                )?;
            let props = instance.get_physical_device_properties(physical);
            if props.api_version < vk::API_VERSION_1_3 {
                return Err(Error(format!(
                    "{} has Vulkan {}.{}; the host needs 1.3",
                    cstr(&props.device_name),
                    vk::api_version_major(props.api_version),
                    vk::api_version_minor(props.api_version)
                )));
            }
            let families = instance.get_physical_device_queue_family_properties(physical);
            let queue_family = families
                .iter()
                .position(|f| {
                    f.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                })
                .ok_or("the GPU has no queue that both draws and computes")?
                as u32;
            let timestamps_valid = families[queue_family as usize].timestamp_valid_bits > 0;

            // What it has.
            let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut all = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut f11)
                .push_next(&mut f12)
                .push_next(&mut f13);
            instance.get_physical_device_features2(physical, &mut all);
            let base = all.features;
            let (f11, f12, f13) = (f11, f12, f13);
            for (have, what) in [
                (f12.timeline_semaphore, "timeline semaphores"),
                (f13.synchronization2, "synchronization2"),
                (f13.dynamic_rendering, "dynamic rendering"),
            ] {
                if have == vk::FALSE {
                    return Err(Error(format!("the GPU lacks {what}, which the host needs")));
                }
            }
            let mut sub = vk::PhysicalDeviceSubgroupProperties::default();
            let mut sub13 = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            let mut driver = vk::PhysicalDeviceDriverProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut sub)
                .push_next(&mut sub13)
                .push_next(&mut driver);
            instance.get_physical_device_properties2(physical, &mut props2);
            let l = props.limits;

            // The features WebGPU names that it has.
            let f16 = f12.shader_float16 == vk::TRUE
                && f11.storage_buffer16_bit_access == vk::TRUE
                && f11.uniform_and_storage_buffer16_bit_access == vk::TRUE;
            let subgroups = sub.supported_stages.contains(vk::ShaderStageFlags::COMPUTE)
                && sub.supported_operations.contains(
                    vk::SubgroupFeatureFlags::BASIC
                        | vk::SubgroupFeatureFlags::VOTE
                        | vk::SubgroupFeatureFlags::ARITHMETIC
                        | vk::SubgroupFeatureFlags::BALLOT
                        | vk::SubgroupFeatureFlags::SHUFFLE,
                );
            let format_has = |f: vk::Format, want: vk::FormatFeatureFlags| {
                instance
                    .get_physical_device_format_properties(physical, f)
                    .optimal_tiling_features
                    .contains(want)
            };
            let mut features: Vec<&'static str> = Vec::new();
            if f16 {
                features.push("shader-f16");
            }
            if subgroups {
                features.push("subgroups");
            }
            if format_has(
                vk::Format::R32_SFLOAT,
                vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
            ) {
                features.push("float32-filterable");
            }
            if format_has(vk::Format::R32_SFLOAT, vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND) {
                features.push("float32-blendable");
            }
            if format_has(
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::FormatFeatureFlags::COLOR_ATTACHMENT,
            ) {
                features.push("rg11b10ufloat-renderable");
            }
            features.push("indirect-first-instance");
            if base.multi_draw_indirect == vk::TRUE {
                features.push("chromium-experimental-multi-draw-indirect");
            }
            if base.texture_compression_bc == vk::TRUE {
                features.push("texture-compression-bc");
            }
            if base.texture_compression_etc2 == vk::TRUE {
                features.push("texture-compression-etc2");
            }
            if base.texture_compression_astc_ldr == vk::TRUE {
                features.push("texture-compression-astc");
            }
            if timestamps_valid && l.timestamp_compute_and_graphics == vk::TRUE {
                features.push("timestamp-query");
            }
            let features = match &options.features {
                Some(keep) => {
                    Features(features.into_iter().filter(|f| keep.iter().any(|k| k == f)).collect())
                }
                None => Features(features),
            };

            // The device, with what it has of what the host uses.
            let priorities = [1.0f32];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)];
            let mut on11 = vk::PhysicalDeviceVulkan11Features::default()
                .storage_buffer16_bit_access(f16)
                .uniform_and_storage_buffer16_bit_access(f16);
            let mut on12 = vk::PhysicalDeviceVulkan12Features::default()
                .timeline_semaphore(true)
                .shader_float16(f16)
                .host_query_reset(f12.host_query_reset == vk::TRUE)
                .shader_int8(f12.shader_int8 == vk::TRUE);
            let mut on13 = vk::PhysicalDeviceVulkan13Features::default()
                .synchronization2(true)
                .dynamic_rendering(true)
                .maintenance4(f13.maintenance4 == vk::TRUE)
                .subgroup_size_control(f13.subgroup_size_control == vk::TRUE)
                .shader_demote_to_helper_invocation(
                    f13.shader_demote_to_helper_invocation == vk::TRUE,
                );
            let on = vk::PhysicalDeviceFeatures::default()
                .multi_draw_indirect(base.multi_draw_indirect == vk::TRUE)
                .draw_indirect_first_instance(base.draw_indirect_first_instance == vk::TRUE)
                .sampler_anisotropy(base.sampler_anisotropy == vk::TRUE)
                .texture_compression_bc(base.texture_compression_bc == vk::TRUE)
                .texture_compression_etc2(base.texture_compression_etc2 == vk::TRUE)
                .texture_compression_astc_ldr(base.texture_compression_astc_ldr == vk::TRUE)
                .fragment_stores_and_atomics(base.fragment_stores_and_atomics == vk::TRUE)
                .shader_storage_image_write_without_format(
                    base.shader_storage_image_write_without_format == vk::TRUE,
                )
                .shader_storage_image_extended_formats(
                    base.shader_storage_image_extended_formats == vk::TRUE,
                )
                .independent_blend(base.independent_blend == vk::TRUE)
                .depth_clamp(base.depth_clamp == vk::TRUE)
                .depth_bias_clamp(base.depth_bias_clamp == vk::TRUE);
            let device_has = instance
                .enumerate_device_extension_properties(physical)
                .unwrap_or_default()
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(ash::khr::swapchain::NAME));
            let swapchain = windowed && device_has;
            let device_extensions: Vec<*const i8> =
                if swapchain { vec![ash::khr::swapchain::NAME.as_ptr()] } else { Vec::new() };
            let raw = instance
                .create_device(
                    physical,
                    &vk::DeviceCreateInfo::default()
                        .queue_create_infos(&queues)
                        .enabled_extension_names(&device_extensions)
                        .enabled_features(&on)
                        .push_next(&mut on11)
                        .push_next(&mut on12)
                        .push_next(&mut on13),
                    None,
                )
                .map_err(|e| Error(format!("Vulkan made no device: {e}")))?;
            let queue = raw.get_device_queue(queue_family, 0);
            let mut timeline_type = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let timeline = raw
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut timeline_type),
                    None,
                )
                .map_err(|e| Error(format!("Vulkan made no semaphore: {e}")))?;
            let memory = instance.get_physical_device_memory_properties(physical);
            let device_type = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => DeviceType::DiscreteGpu,
                vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceType::IntegratedGpu,
                vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceType::VirtualGpu,
                vk::PhysicalDeviceType::CPU => DeviceType::Cpu,
                _ => DeviceType::Other,
            };
            let info = AdapterInfo {
                name: cstr(&props.device_name),
                backend: Backend::Vulkan,
                device_type,
                vendor: props.vendor_id,
                device: props.device_id,
                driver: cstr(&driver.driver_name),
                driver_info: cstr(&driver.driver_info),
                subgroup_min_size: if sub13.min_subgroup_size > 0 {
                    sub13.min_subgroup_size
                } else {
                    sub.subgroup_size
                },
                subgroup_max_size: sub13.max_subgroup_size.max(sub.subgroup_size),
            };
            let storage_max = u64::from(l.max_storage_buffer_range).min(u64::from(u32::MAX)) & !3;
            let limits = Limits {
                max_texture_dimension_2d: l.max_image_dimension2_d,
                max_texture_dimension_3d: l.max_image_dimension3_d,
                max_texture_array_layers: l.max_image_array_layers,
                max_buffer_size: storage_max.max(u64::from(l.max_storage_buffer_range)),
                max_storage_buffer_binding_size: storage_max,
                max_uniform_buffer_binding_size: u64::from(l.max_uniform_buffer_range),
                max_storage_buffers_per_shader_stage: l.max_per_stage_descriptor_storage_buffers,
                max_sampled_textures_per_shader_stage: l.max_per_stage_descriptor_sampled_images,
                max_samplers_per_shader_stage: l.max_per_stage_descriptor_samplers,
                max_storage_textures_per_shader_stage: l.max_per_stage_descriptor_storage_images,
                max_color_attachments: l.max_color_attachments.min(8),
                max_color_attachment_bytes_per_sample: 64,
                min_uniform_buffer_offset_alignment: (l.min_uniform_buffer_offset_alignment as u32)
                    .max(256),
                min_storage_buffer_offset_alignment: (l.min_storage_buffer_offset_alignment as u32)
                    .max(32),
                max_compute_workgroup_storage_size: l.max_compute_shared_memory_size,
                max_compute_invocations_per_workgroup: l.max_compute_work_group_invocations,
                max_compute_workgroup_size: l.max_compute_work_group_size,
                max_compute_workgroups_per_dimension: l.max_compute_work_group_count[0]
                    .min(l.max_compute_work_group_count[1])
                    .min(l.max_compute_work_group_count[2]),
            };
            Ok(Device(Arc::new(Shared {
                entry,
                swapchain,
                instance,
                debug,
                physical,
                raw,
                queue,
                queue_family,
                memory,
                info,
                limits,
                features,
                timestamp_period: l.timestamp_period,
                timeline,
                submitted: AtomicU64::new(0),
                pending: Mutex::new(Vec::new()),
                in_flight: Mutex::new(Vec::new()),
                errors,
                descriptor_pools: Mutex::new(Vec::new()),
                belt: Mutex::new(Belt::default()),
            })))
        }
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

    /// Nanoseconds a timestamp tick.
    pub fn timestamp_period(&self) -> f32 {
        self.0.timestamp_period
    }

    pub fn report(&self) -> String {
        let i = &self.0.info;
        let mut s = String::new();
        s.push_str(&format!("driver {} ({})\n", i.driver, i.driver_info));
        s.push_str(&format!("subgroups {} to {}\n", i.subgroup_min_size, i.subgroup_max_size));
        for (k, h) in self.0.memory.memory_heaps_as_slice().iter().enumerate() {
            s.push_str(&format!("heap {k}: {} bytes, {:?}\n", h.size, h.flags));
        }
        s
    }

    fn raw(&self) -> &ash::Device {
        &self.0.raw
    }

    /// The first memory type of `bits` with `want`.
    fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        let m = &self.0.memory;
        (0..m.memory_type_count).find(|&i| {
            bits & (1 << i) != 0 && m.memory_types[i as usize].property_flags.contains(want)
        })
    }

    fn raw_buffer(&self, size: u64, mappable: bool, usage: vk::BufferUsageFlags) -> Arc<RawBuffer> {
        let size = size.max(4).next_multiple_of(4);
        // SAFETY: a buffer, its memory and its mapping, made and bound here.
        unsafe {
            let d = self.raw();
            let buffer = d
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .expect("Vulkan made no buffer");
            let req = d.get_buffer_memory_requirements(buffer);
            let want = if mappable {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            } else {
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            };
            let ty = self
                .memory_type(req.memory_type_bits, want)
                .or_else(|| {
                    self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::empty())
                })
                .expect("no memory type for a buffer");
            let memory = d
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(ty),
                    None,
                )
                .expect("Vulkan allocated no memory");
            d.bind_buffer_memory(buffer, memory, 0).expect("Vulkan bound no memory");
            let mapped = if mappable {
                d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .expect("Vulkan mapped no memory") as *mut u8
            } else {
                std::ptr::null_mut()
            };
            Arc::new(RawBuffer { device: self.0.clone(), buffer, memory, size, mapped })
        }
    }

    pub fn create_buffer(&self, desc: &BufferDesc<'_>) -> Buffer {
        let usage = if desc.mappable {
            vk::BufferUsageFlags::TRANSFER_DST
        } else {
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
        };
        let raw = self.raw_buffer(desc.size, desc.mappable, usage);
        if desc.mappable {
            // SAFETY: the mapping is the buffer's whole size.
            unsafe { std::ptr::write_bytes(raw.mapped, 0, raw.size as usize) };
        } else {
            lock(&self.0.pending).push(Pending::Fill { dst: raw.clone() });
        }
        Buffer { raw, size: desc.size, mappable: desc.mappable }
    }

    pub fn create_texture(&self, desc: &TextureDesc<'_>) -> Texture {
        let layers = desc.depth_or_layers.max(1);
        let levels = desc.levels.max(1);
        let format = vk_format(desc.format);
        let depth = desc.format == TextureFormat::Depth32Float;
        let mut usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        if desc.usage.storage {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        if desc.usage.render {
            usage |= if depth {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
            };
        }
        let (ty, extent, array) = match desc.dimension {
            TextureDimension::D3 => (
                vk::ImageType::TYPE_3D,
                vk::Extent3D { width: desc.width, height: desc.height, depth: layers },
                1,
            ),
            TextureDimension::D2 => (
                vk::ImageType::TYPE_2D,
                vk::Extent3D { width: desc.width, height: desc.height, depth: 1 },
                layers,
            ),
        };
        let cube =
            desc.dimension == TextureDimension::D2 && layers % 6 == 0 && desc.width == desc.height;
        // SAFETY: an image and its memory, made and bound here.
        let raw = unsafe {
            let d = self.raw();
            let image = d
                .create_image(
                    &vk::ImageCreateInfo::default()
                        // Never MUTABLE_FORMAT: every view keeps its image's format, and the
                        // flag would cost a target its compression (RADV drops DCC).
                        .flags(if cube {
                            vk::ImageCreateFlags::CUBE_COMPATIBLE
                        } else {
                            vk::ImageCreateFlags::empty()
                        })
                        .image_type(ty)
                        .format(format)
                        .extent(extent)
                        .mip_levels(levels)
                        .array_layers(array)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .expect("Vulkan made no image");
            let req = d.get_image_memory_requirements(image);
            let ty = self
                .memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .or_else(|| {
                    self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::empty())
                })
                .expect("no memory type for an image");
            let memory = d
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(ty),
                    None,
                )
                .expect("Vulkan allocated no memory");
            d.bind_image_memory(image, memory, 0).expect("Vulkan bound no memory");
            Arc::new(RawImage {
                device: self.0.clone(),
                image,
                memory,
                format,
                aspect: if depth {
                    vk::ImageAspectFlags::DEPTH
                } else {
                    vk::ImageAspectFlags::COLOR
                },
                swapchain: false,
                home: if desc.usage.storage {
                    vk::ImageLayout::GENERAL
                } else if depth {
                    vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                } else {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                },
                levels,
                layers: array,
            })
        };
        lock(&self.0.pending).push(Pending::Clear { dst: raw.clone() });
        Texture {
            raw,
            format: desc.format,
            width: desc.width,
            height: desc.height,
            depth_or_layers: layers,
            levels,
            dimension: desc.dimension,
        }
    }

    pub fn create_sampler(&self, desc: &SamplerDesc) -> Sampler {
        let filter = if desc.linear { vk::Filter::LINEAR } else { vk::Filter::NEAREST };
        let mode = if desc.repeat {
            vk::SamplerAddressMode::REPEAT
        } else {
            vk::SamplerAddressMode::CLAMP_TO_EDGE
        };
        let mut info = vk::SamplerCreateInfo::default()
            .mag_filter(filter)
            .min_filter(filter)
            .mipmap_mode(if desc.mip_linear {
                vk::SamplerMipmapMode::LINEAR
            } else {
                vk::SamplerMipmapMode::NEAREST
            })
            .address_mode_u(mode)
            .address_mode_v(mode)
            .address_mode_w(mode)
            .min_lod(0.0)
            .max_lod(32.0);
        if desc.anisotropy > 1 {
            info = info.anisotropy_enable(true).max_anisotropy(f32::from(desc.anisotropy));
        }
        if let Some(c) = desc.compare {
            info = info.compare_enable(true).compare_op(compare_op(c));
        }
        // SAFETY: a sampler.
        let raw =
            unsafe { self.raw().create_sampler(&info, None).expect("Vulkan made no sampler") };
        Sampler { raw: Arc::new(RawSampler { device: self.0.clone(), raw }) }
    }

    pub fn create_query_set(&self, count: u32) -> Result<QuerySet> {
        if !self.0.features.has("timestamp-query") {
            return Err("this GPU writes no timestamps".into());
        }
        // SAFETY: a query pool.
        let pool = unsafe {
            self.raw()
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(count),
                    None,
                )
                .map_err(|e| Error(format!("Vulkan made no query pool: {e}")))?
        };
        Ok(QuerySet { raw: Arc::new(RawQueries { device: self.0.clone(), pool, count }) })
    }

    pub fn create_bind_group_layout(&self, entries: &[LayoutEntry]) -> BindGroupLayout {
        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = entries
            .iter()
            .enumerate()
            .map(|(k, e)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(k as u32)
                    .descriptor_type(descriptor_type(e.ty))
                    .descriptor_count(1)
                    .stage_flags(stages(e.visibility))
            })
            .collect();
        // SAFETY: a set layout and a pipeline layout over it.
        unsafe {
            let d = self.raw();
            let set = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .expect("Vulkan made no set layout");
            let sets = [set];
            let pipeline = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&sets),
                    None,
                )
                .expect("Vulkan made no pipeline layout");
            BindGroupLayout(Arc::new(Layout {
                device: self.0.clone(),
                entries: entries.to_vec(),
                set,
                pipeline,
            }))
        }
    }

    pub fn create_bind_group(
        &self,
        layout: &BindGroupLayout,
        entries: &[BindGroupEntry<'_>],
    ) -> BindGroup {
        let l = &layout.0;
        let mut sizes: Vec<vk::DescriptorPoolSize> = Vec::new();
        for e in &l.entries {
            let ty = descriptor_type(e.ty);
            match sizes.iter_mut().find(|s| s.ty == ty) {
                Some(s) => s.descriptor_count += 1,
                None => sizes.push(vk::DescriptorPoolSize { ty, descriptor_count: 1 }),
            }
        }
        // SAFETY: a pool of one set, the set, and its descriptors written from live resources.
        unsafe {
            let d = self.raw();
            let pool = if sizes.is_empty() {
                vk::DescriptorPool::null()
            } else {
                d.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes),
                    None,
                )
                .expect("Vulkan made no descriptor pool")
            };
            let set = if pool == vk::DescriptorPool::null() {
                vk::DescriptorSet::null()
            } else {
                let layouts = [l.set];
                d.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
                .expect("Vulkan allocated no descriptor set")[0]
            };
            let mut buffers: Vec<[vk::DescriptorBufferInfo; 1]> = Vec::new();
            let mut images: Vec<[vk::DescriptorImageInfo; 1]> = Vec::new();
            let mut writes: Vec<(u32, vk::DescriptorType, bool, usize)> = Vec::new();
            let mut held = Vec::new();
            let mut uses: Vec<(u64, bool)> = Vec::new();
            for e in entries {
                let Some(k) = l.entries.iter().position(|x| x.binding == e.binding) else {
                    continue;
                };
                let ty = descriptor_type(l.entries[k].ty);
                match &e.resource {
                    Resource::Buffer { buffer, offset, size } => {
                        let range = match (l.entries[k].ty, size) {
                            // A dynamic uniform's range is its block, at offsets the pass gives.
                            (BindingType::Uniform { dynamic: true, size }, _)
                            | (BindingType::Storage { dynamic: true, size, .. }, _) => size,
                            (_, Some(s)) => *s,
                            (_, None) => vk::WHOLE_SIZE,
                        };
                        buffers.push([vk::DescriptorBufferInfo {
                            buffer: buffer.raw.buffer,
                            offset: *offset,
                            range,
                        }]);
                        writes.push((k as u32, ty, true, buffers.len() - 1));
                        held.push(Held::Buffer(buffer.raw.clone()));
                        uses.push((
                            vk::Handle::as_raw(buffer.raw.buffer),
                            matches!(
                                l.entries[k].ty,
                                BindingType::Storage { read_only: false, .. }
                            ),
                        ));
                    }
                    Resource::View(v) => {
                        images.push([vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: v.raw.view,
                            image_layout: v.raw.image.home,
                        }]);
                        writes.push((k as u32, ty, false, images.len() - 1));
                        held.push(Held::View(v.raw.clone()));
                        uses.push((
                            vk::Handle::as_raw(v.raw.image.image),
                            matches!(l.entries[k].ty, BindingType::StorageTexture { .. }),
                        ));
                    }
                    Resource::Sampler(s) => {
                        images.push([vk::DescriptorImageInfo {
                            sampler: s.raw.raw,
                            image_view: vk::ImageView::null(),
                            image_layout: vk::ImageLayout::UNDEFINED,
                        }]);
                        writes.push((k as u32, ty, false, images.len() - 1));
                        held.push(Held::Sampler(s.raw.clone()));
                    }
                }
            }
            let vk_writes: Vec<vk::WriteDescriptorSet<'_>> = writes
                .iter()
                .map(|&(binding, ty, buffer, at)| {
                    let w = vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding)
                        .descriptor_type(ty);
                    if buffer { w.buffer_info(&buffers[at]) } else { w.image_info(&images[at]) }
                })
                .collect();
            if !vk_writes.is_empty() {
                d.update_descriptor_sets(&vk_writes, &[]);
            }
            BindGroup(Arc::new(Group {
                device: self.0.clone(),
                layout: layout.0.clone(),
                pool,
                set,
                _held: held,
                uses,
            }))
        }
    }

    /// A shader module of SPIR-V words (little-endian bytes).
    pub fn create_shader_module(&self, label: &str, source: &[u8]) -> Result<ShaderModule> {
        if source.len() % 4 != 0 {
            return Err(Error(format!(
                "{label}: SPIR-V of {} bytes, not whole words",
                source.len()
            )));
        }
        let words: Vec<u32> =
            source.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        // SAFETY: a module of the words.
        let raw = unsafe {
            self.raw()
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(|e| Error(format!("{label}: Vulkan made no shader module: {e}")))?
        };
        Ok(ShaderModule { raw: Arc::new(RawModule { device: self.0.clone(), raw }) })
    }

    pub fn create_compute_pipeline(
        &self,
        desc: &ComputePipelineDesc<'_>,
    ) -> Result<ComputePipeline> {
        let name = CString::new(desc.entry).map_err(|_| "an entry point's name with a NUL")?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(desc.module.raw.raw)
            .name(&name);
        let info =
            vk::ComputePipelineCreateInfo::default().stage(stage).layout(desc.layout.0.pipeline);
        // SAFETY: a pipeline from a live module and layout.
        let raw = unsafe {
            self.raw().create_compute_pipelines(vk::PipelineCache::null(), &[info], None).map_err(
                |(_, e)| Error(format!("{}: Vulkan made no compute pipeline: {e}", desc.label)),
            )?[0]
        };
        Ok(ComputePipeline {
            raw: Arc::new(RawPipeline { device: self.0.clone(), raw }),
            layout: desc.layout.0.clone(),
        })
    }

    pub fn create_render_pipeline(&self, desc: &RenderPipelineDesc<'_>) -> Result<RenderPipeline> {
        let vs = CString::new(desc.vertex_entry).map_err(|_| "an entry point's name with a NUL")?;
        let fs =
            CString::new(desc.fragment_entry).map_err(|_| "an entry point's name with a NUL")?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(desc.module.raw.raw)
                .name(&vs),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(desc.module.raw.raw)
                .name(&fs),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport =
            vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let mut raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(match desc.cull {
                Cull::None => vk::CullModeFlags::NONE,
                Cull::Back => vk::CullModeFlags::BACK,
                Cull::Front => vk::CullModeFlags::FRONT,
            })
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        if let Some(d) = &desc.depth
            && (d.bias.constant != 0 || d.bias.slope_scale != 0.0)
        {
            raster = raster
                .depth_bias_enable(true)
                .depth_bias_constant_factor(d.bias.constant as f32)
                .depth_bias_slope_factor(d.bias.slope_scale)
                .depth_bias_clamp(d.bias.clamp);
        }
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_state = match &desc.depth {
            Some(d) => vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(d.state.write)
                .depth_compare_op(compare_op(d.state.compare)),
            None => vk::PipelineDepthStencilStateCreateInfo::default(),
        };
        let attachments: Vec<vk::PipelineColorBlendAttachmentState> =
            desc.colors.iter().map(|c| blend_state(c.blend)).collect();
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
        let formats: Vec<vk::Format> = desc.colors.iter().map(|c| vk_format(c.format)).collect();
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&formats)
            .depth_attachment_format(
                desc.depth.as_ref().map_or(vk::Format::UNDEFINED, |d| vk_format(d.format)),
            );
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_state)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(desc.layout.0.pipeline)
            .push_next(&mut rendering);
        // SAFETY: a pipeline from a live module and layout, its state built above.
        let raw = unsafe {
            self.raw().create_graphics_pipelines(vk::PipelineCache::null(), &[info], None).map_err(
                |(_, e)| Error(format!("{}: Vulkan made no render pipeline: {e}", desc.label)),
            )?[0]
        };
        Ok(RenderPipeline {
            raw: Arc::new(RawPipeline { device: self.0.clone(), raw }),
            layout: desc.layout.0.clone(),
        })
    }

    pub fn create_command_encoder(&self, label: &str) -> CommandEncoder {
        CommandEncoder::new(self.clone(), label)
    }

    /// Writes `data` at `offset` of `buffer` before the next submission's own work.
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let (staging, at) = self.staging(data, 4);
        lock(&self.0.pending).push(Pending::Buffer {
            staging,
            at,
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
        let (staging, staged_at) = self.staging(data, 256);
        let (layer, z) = match at.texture.dimension {
            TextureDimension::D3 => (0, at.origin[2]),
            TextureDimension::D2 => (at.origin[2], 0),
        };
        lock(&self.0.pending).push(Pending::Texture {
            staging,
            at: staged_at,
            bytes_per_row,
            rows_per_image,
            dst: at.texture.raw.clone(),
            level: at.level,
            layer,
            origin: [at.origin[0], at.origin[1], z],
            extent,
        });
    }

    /// `data` copied into the staging belt at an offset aligned to `align`: the buffer and the
    /// offset, which the next submission copies from.
    fn staging(&self, data: &[u8], align: u64) -> (Arc<RawBuffer>, u64) {
        let need = data.len() as u64;
        let mut belt = lock(&self.0.belt);
        let (buffer, at) = match belt.current.take() {
            Some((b, used)) if used.next_multiple_of(align) + need <= b.size => {
                let at = used.next_multiple_of(align);
                (b, at)
            }
            other => {
                if let Some((b, _)) = other {
                    belt.filled.push(b);
                }
                let free = belt.free.iter().position(|b| b.size >= need);
                let b = match free {
                    Some(k) => belt.free.swap_remove(k),
                    None => self.raw_buffer(
                        need.max(BELT_CHUNK).next_power_of_two(),
                        true,
                        vk::BufferUsageFlags::TRANSFER_SRC,
                    ),
                };
                (b, 0)
            }
        };
        // SAFETY: the mapping holds `at + need` bytes; no submission uses this part of it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.mapped.add(at as usize), data.len())
        };
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

    /// Records the staged writes, then submits them and `buffers`, in order: their submission's
    /// index.
    pub fn submit(&self, buffers: Vec<CommandBuffer>) -> SubmissionIndex {
        self.submit_waiting(buffers, None)
    }

    /// [`Device::submit`], first waiting on a binary semaphore and signalling another as it ends
    /// (a frame's image acquired, and rendered).
    fn submit_waiting(
        &self,
        buffers: Vec<CommandBuffer>,
        frame: Option<(vk::Semaphore, vk::Semaphore)>,
    ) -> SubmissionIndex {
        let pending = std::mem::take(&mut *lock(&self.0.pending));
        let index = self.0.submitted.fetch_add(1, Ordering::AcqRel) + 1;
        self.turn_belt(index);
        let mut keep: Vec<Keep> = Vec::new();
        let mut pools = Vec::new();
        let mut cbs = Vec::new();
        if !pending.is_empty() {
            let mut enc = CommandEncoder::new(self.clone(), "queue writes");
            for w in &pending {
                enc.record_pending(w, &mut keep);
            }
            let cb = enc.finish();
            pools.push(cb.pool);
            cbs.push(cb.raw);
            keep.extend(cb.keep);
        }
        for b in buffers {
            pools.push(b.pool);
            cbs.push(b.raw);
            keep.extend(b.keep);
        }
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.0.timeline)
            .value(index)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let cb_infos: Vec<vk::CommandBufferSubmitInfo<'_>> = cbs
            .iter()
            .map(|&cb| vk::CommandBufferSubmitInfo::default().command_buffer(cb))
            .collect();
        let mut signals = signal.to_vec();
        let mut waits = Vec::new();
        if let Some((acquired, rendered)) = frame {
            waits.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(acquired)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
            signals.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(rendered)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(&cb_infos)
            .signal_semaphore_infos(&signals);
        // SAFETY: recorded command buffers, submitted once, kept until the semaphore passes
        // `index`.
        let r = unsafe { self.raw().queue_submit2(self.0.queue, &[submit], vk::Fence::null()) };
        if let Err(e) = r {
            lock(&self.0.errors).push(format!("Vulkan refused a submission: {e}"));
        }
        let mut f = lock(&self.0.in_flight);
        for p in pools {
            f.push((index, p, Vec::new()));
        }
        if let Some(last) = f.last_mut() {
            last.2 = keep;
        } else {
            f.push((index, vk::CommandPool::null(), keep));
        }
        drop(f);
        self.reclaim();
        index
    }

    /// Frees what submissions now done held.
    fn reclaim(&self) {
        let done = self.completed();
        let freed: Vec<(SubmissionIndex, vk::CommandPool, Vec<Keep>)> = {
            let mut f = lock(&self.0.in_flight);
            let (gone, kept): (Vec<_>, Vec<_>) = f.drain(..).partition(|(i, _, _)| *i <= done);
            *f = kept;
            gone
        };
        for (_, pool, _keep) in freed {
            if pool != vk::CommandPool::null() {
                // SAFETY: its submission is done.
                unsafe { self.raw().destroy_command_pool(pool, None) };
            }
        }
    }

    /// The submissions done so far: every one up to this index.
    pub fn completed(&self) -> SubmissionIndex {
        // SAFETY: the timeline semaphore's value.
        match unsafe { self.raw().get_semaphore_counter_value(self.0.timeline) } {
            Ok(v) => v,
            Err(e) => {
                lock(&self.0.errors).push(format!("Vulkan lost the device: {e}"));
                self.submitted()
            }
        }
    }

    pub fn submitted(&self) -> SubmissionIndex {
        self.0.submitted.load(Ordering::Acquire)
    }

    /// Waits for submission `index` (and every one before it) to be done.
    pub fn wait(&self, index: SubmissionIndex) {
        let index = index.min(self.submitted());
        let sems = [self.0.timeline];
        let values = [index];
        let info = vk::SemaphoreWaitInfo::default().semaphores(&sems).values(&values);
        // SAFETY: waiting on the timeline semaphore.
        if let Err(e) = unsafe { self.raw().wait_semaphores(&info, u64::MAX) } {
            lock(&self.0.errors).push(format!("Vulkan lost the device: {e}"));
        }
        self.reclaim();
    }

    /// Submits the staged writes, if any, and waits for every submission to be done.
    pub fn wait_idle(&self) {
        if !lock(&self.0.pending).is_empty() {
            self.submit(Vec::new());
        }
        self.wait(self.submitted());
    }

    /// The errors reported since the last take.
    pub fn take_errors(&self) -> Vec<String> {
        std::mem::take(&mut *lock(&self.0.errors))
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: the device is idle before anything of it is destroyed; what's left of each
        // submission is freed, then the device and the instance.
        unsafe {
            let _ = self.raw.device_wait_idle();
            for (_, pool, _) in lock(&self.in_flight).drain(..) {
                if pool != vk::CommandPool::null() {
                    self.raw.destroy_command_pool(pool, None);
                }
            }
            lock(&self.pending).clear();
            for p in lock(&self.descriptor_pools).drain(..) {
                self.raw.destroy_descriptor_pool(p, None);
            }
            self.raw.destroy_semaphore(self.timeline, None);
            self.raw.destroy_device(None);
            if let Some((du, m)) = self.debug.take() {
                du.destroy_debug_utils_messenger(m, None);
            }
            self.instance.destroy_instance(None);
        }
        let _ = (self.physical, self.queue_family);
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn cstr(chars: &[std::ffi::c_char]) -> String {
    // SAFETY: Vulkan's fixed-size names end in a NUL.
    unsafe { CStr::from_ptr(chars.as_ptr()) }.to_string_lossy().into_owned()
}

// ---- resources ------------------------------------------------------------------------------------

struct RawBuffer {
    device: Arc<Shared>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    /// The host's view of a mappable buffer's memory; null otherwise.
    mapped: *mut u8,
}

// SAFETY: the mapping is only read or written by the host's one recording thread.
unsafe impl Send for RawBuffer {}
unsafe impl Sync for RawBuffer {}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        // SAFETY: nothing in flight holds it (submissions keep their resources alive).
        unsafe {
            self.device.raw.destroy_buffer(self.buffer, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}

pub struct Buffer {
    raw: Arc<RawBuffer>,
    size: u64,
    mappable: bool,
}

impl Buffer {
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Whether it and `other` are the same buffer (clones of one).
    pub fn same(&self, other: &Buffer) -> bool {
        Arc::ptr_eq(&self.raw, &other.raw)
    }

    /// A mappable buffer's bytes `offset..offset + len`, read in place: its submission must be
    /// done.
    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        if !self.mappable {
            return Err("the buffer isn't mappable".into());
        }
        if offset + len > self.raw.size {
            return Err("a read past the buffer's end".into());
        }
        // SAFETY: the mapping is coherent and holds the range.
        Ok(unsafe {
            std::slice::from_raw_parts(self.raw.mapped.add(offset as usize), len as usize)
        }
        .to_vec())
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Buffer {
        Buffer { raw: self.raw.clone(), size: self.size, mappable: self.mappable }
    }
}

struct RawImage {
    device: Arc<Shared>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    levels: u32,
    layers: u32,
    /// The layout it rests in between commands, by its use: `GENERAL` where a kernel writes it,
    /// else read-only optimal. A command that uses it otherwise (a pass drawing into it, a
    /// copy) moves what it uses out and back, so a compressed target stays compressed
    /// (RADV keeps no DCC or HTILE in `GENERAL`).
    home: vk::ImageLayout,
    /// A swapchain's image: not ours to free, and drawn whole each frame, so a pass takes it
    /// from `UNDEFINED`.
    swapchain: bool,
}

impl RawImage {
    /// Every level and layer.
    fn all(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        }
    }

    /// One level's layers `layer..layer + count`.
    fn part(&self, level: u32, layer: u32, count: u32) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect,
            base_mip_level: level,
            level_count: 1,
            base_array_layer: layer,
            layer_count: count,
        }
    }
}

/// `range` of `image` moved from layout `from` to `to`, everything before done and visible to
/// everything after.
fn transition(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    if from == to {
        return;
    }
    let barrier = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(from)
        .new_layout(to)
        .image(image)
        .subresource_range(range)];
    // SAFETY: a layout transition in a recording command buffer.
    unsafe {
        d.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&barrier))
    };
}

impl Drop for RawImage {
    fn drop(&mut self) {
        if self.swapchain {
            return;
        }
        // SAFETY: nothing in flight holds it.
        unsafe {
            self.device.raw.destroy_image(self.image, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}

#[derive(Clone)]
pub struct Texture {
    raw: Arc<RawImage>,
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

    pub fn create_view(&self, desc: &ViewDesc) -> TextureView {
        let ty = match desc.dimension {
            ViewDimension::D2 => vk::ImageViewType::TYPE_2D,
            ViewDimension::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
            ViewDimension::Cube => vk::ImageViewType::CUBE,
            ViewDimension::D3 => vk::ImageViewType::TYPE_3D,
        };
        let range = vk::ImageSubresourceRange {
            aspect_mask: self.raw.aspect,
            base_mip_level: desc.base_level,
            level_count: desc.level_count.unwrap_or(self.raw.levels - desc.base_level),
            base_array_layer: desc.base_layer,
            layer_count: desc.layer_count.unwrap_or(self.raw.layers - desc.base_layer),
        };
        // SAFETY: a view of a live image.
        let view = unsafe {
            self.raw
                .device
                .raw
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(self.raw.image)
                        .view_type(ty)
                        .format(self.raw.format)
                        .subresource_range(range),
                    None,
                )
                .expect("Vulkan made no image view")
        };
        let level = desc.base_level;
        TextureView {
            raw: Arc::new(RawView {
                image: self.raw.clone(),
                view,
                range,
                width: (self.width >> level).max(1),
                height: (self.height >> level).max(1),
            }),
        }
    }
}

struct RawView {
    image: Arc<RawImage>,
    view: vk::ImageView,
    /// Its levels and layers.
    range: vk::ImageSubresourceRange,
    width: u32,
    height: u32,
}

impl Drop for RawView {
    fn drop(&mut self) {
        // SAFETY: nothing in flight holds it.
        unsafe { self.image.device.raw.destroy_image_view(self.view, None) };
    }
}

#[derive(Clone)]
pub struct TextureView {
    raw: Arc<RawView>,
}

struct RawSampler {
    device: Arc<Shared>,
    raw: vk::Sampler,
}

impl Drop for RawSampler {
    fn drop(&mut self) {
        // SAFETY: nothing in flight holds it.
        unsafe { self.device.raw.destroy_sampler(self.raw, None) };
    }
}

#[derive(Clone)]
pub struct Sampler {
    raw: Arc<RawSampler>,
}

struct RawQueries {
    device: Arc<Shared>,
    pool: vk::QueryPool,
    count: u32,
}

impl Drop for RawQueries {
    fn drop(&mut self) {
        // SAFETY: nothing in flight holds it.
        unsafe { self.device.raw.destroy_query_pool(self.pool, None) };
    }
}

#[derive(Clone)]
pub struct QuerySet {
    raw: Arc<RawQueries>,
}

impl QuerySet {
    pub fn count(&self) -> u32 {
        self.raw.count
    }
}

struct RawModule {
    device: Arc<Shared>,
    raw: vk::ShaderModule,
}

impl Drop for RawModule {
    fn drop(&mut self) {
        // SAFETY: pipelines made of it don't need it.
        unsafe { self.device.raw.destroy_shader_module(self.raw, None) };
    }
}

#[derive(Clone)]
pub struct ShaderModule {
    raw: Arc<RawModule>,
}

struct RawPipeline {
    device: Arc<Shared>,
    raw: vk::Pipeline,
}

impl Drop for RawPipeline {
    fn drop(&mut self) {
        // SAFETY: nothing in flight holds it.
        unsafe { self.device.raw.destroy_pipeline(self.raw, None) };
    }
}

#[derive(Clone)]
pub struct ComputePipeline {
    raw: Arc<RawPipeline>,
    layout: Arc<Layout>,
}

#[derive(Clone)]
pub struct RenderPipeline {
    raw: Arc<RawPipeline>,
    layout: Arc<Layout>,
}

struct Layout {
    device: Arc<Shared>,
    entries: Vec<LayoutEntry>,
    set: vk::DescriptorSetLayout,
    pipeline: vk::PipelineLayout,
}

impl Drop for Layout {
    fn drop(&mut self) {
        // SAFETY: nothing in flight holds it.
        unsafe {
            self.device.raw.destroy_pipeline_layout(self.pipeline, None);
            self.device.raw.destroy_descriptor_set_layout(self.set, None);
        }
    }
}

#[derive(Clone)]
pub struct BindGroupLayout(Arc<Layout>);

/// A resource a bind group or a command holds alive (never read but to keep it).
#[allow(dead_code)]
enum Held {
    Buffer(Arc<RawBuffer>),
    Image(Arc<RawImage>),
    View(Arc<RawView>),
    Sampler(Arc<RawSampler>),
}

struct Group {
    device: Arc<Shared>,
    layout: Arc<Layout>,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// The resources it binds, alive while it is.
    _held: Vec<Held>,
    /// What it binds by its handle (a buffer's or an image's), and whether a shader may write
    /// it: what a command that sets it uses, for the encoder's hazards.
    uses: Vec<(u64, bool)>,
}

impl Drop for Group {
    fn drop(&mut self) {
        if self.pool != vk::DescriptorPool::null() {
            // SAFETY: nothing in flight holds it: a pass holds its groups until it's done.
            unsafe { self.device.raw.destroy_descriptor_pool(self.pool, None) };
        }
    }
}

#[derive(Clone)]
pub struct BindGroup(Arc<Group>);

// ---- recording ------------------------------------------------------------------------------------

pub struct CommandEncoder {
    device: Device,
    pool: vk::CommandPool,
    raw: vk::CommandBuffer,
    /// What its commands use, alive until its submission is done.
    keep: Vec<Keep>,
    groups: Vec<Arc<Group>>,
    pipelines: Vec<Arc<RawPipeline>>,
    held: Vec<Held>,
    /// The resources written, and read, since the last barrier, by handle.
    written: std::collections::HashSet<u64>,
    read: std::collections::HashSet<u64>,
}

pub struct CommandBuffer {
    pool: vk::CommandPool,
    raw: vk::CommandBuffer,
    keep: Vec<Keep>,
}

/// Everything before, done and visible to everything after: the full barrier between commands.
fn full_barrier(d: &ash::Device, cb: vk::CommandBuffer) {
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
    // SAFETY: a barrier in a recording command buffer.
    unsafe {
        d.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&barrier))
    };
}

impl CommandEncoder {
    fn new(device: Device, label: &str) -> CommandEncoder {
        let _ = label;
        // SAFETY: a pool of one command buffer, begun for one submission.
        unsafe {
            let d = device.raw();
            let pool = d
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(device.0.queue_family)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                    None,
                )
                .expect("Vulkan made no command pool");
            let raw = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("Vulkan allocated no command buffer")[0];
            d.begin_command_buffer(
                raw,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("Vulkan began no command buffer");
            // What the submissions before did, this one sees.
            full_barrier(d, raw);
            CommandEncoder {
                device,
                pool,
                raw,
                keep: Vec::new(),
                groups: Vec::new(),
                pipelines: Vec::new(),
                held: Vec::new(),
                written: std::collections::HashSet::new(),
                read: std::collections::HashSet::new(),
            }
        }
    }

    fn d(&self) -> &ash::Device {
        &self.device.0.raw
    }

    /// Before a command that reads `reads` and writes `writes`: a barrier, if it reads or
    /// writes what a command since the last one wrote, or writes what one read; then notes
    /// its own.
    fn hazard(&mut self, reads: &[u64], writes: &[u64]) {
        let conflict = reads.iter().any(|r| self.written.contains(r))
            || writes.iter().any(|w| self.written.contains(w) || self.read.contains(w));
        if conflict {
            self.barrier_all();
        }
        self.read.extend(reads);
        self.written.extend(writes);
    }

    /// The full barrier, and nothing pending after it.
    fn barrier_all(&mut self) {
        full_barrier(self.d(), self.raw);
        self.written.clear();
        self.read.clear();
    }

    /// One staged write.
    fn record_pending(&mut self, w: &Pending, keep: &mut Vec<Keep>) {
        let cb = self.raw;
        // SAFETY: copies, fills and clears in a recording command buffer, of live resources the
        // submission keeps.
        unsafe {
            let d = &self.device.0.raw;
            match w {
                Pending::Buffer { staging, at, dst, offset, len } => {
                    let region =
                        [vk::BufferCopy { src_offset: *at, dst_offset: *offset, size: *len }];
                    d.cmd_copy_buffer(cb, staging.buffer, dst.buffer, &region);
                    keep.push(Keep::Buffer(staging.clone()));
                    keep.push(Keep::Buffer(dst.clone()));
                }
                Pending::Fill { dst } => {
                    d.cmd_fill_buffer(cb, dst.buffer, 0, vk::WHOLE_SIZE, 0);
                    keep.push(Keep::Buffer(dst.clone()));
                }
                Pending::Texture {
                    staging,
                    at,
                    bytes_per_row,
                    rows_per_image,
                    dst,
                    level,
                    layer,
                    origin,
                    extent,
                } => {
                    let texel = format_texel(dst.format);
                    let region = [vk::BufferImageCopy {
                        buffer_offset: *at,
                        buffer_row_length: bytes_per_row / texel.0 * texel.1,
                        buffer_image_height: rows_per_image * texel.1,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: dst.aspect,
                            mip_level: *level,
                            base_array_layer: *layer,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D {
                            x: origin[0] as i32,
                            y: origin[1] as i32,
                            z: origin[2] as i32,
                        },
                        image_extent: vk::Extent3D {
                            width: extent[0],
                            height: extent[1],
                            depth: extent[2],
                        },
                    }];
                    let part = dst.part(*level, *layer, 1);
                    let to = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                    transition(d, cb, dst.image, part, dst.home, to);
                    d.cmd_copy_buffer_to_image(cb, staging.buffer, dst.image, to, &region);
                    transition(d, cb, dst.image, part, to, dst.home);
                    keep.push(Keep::Buffer(staging.clone()));
                    keep.push(Keep::Image(dst.clone()));
                }
                Pending::Clear { dst } => {
                    keep.push(Keep::Image(dst.clone()));
                    // From nothing to a copy's target, zeroed, then home.
                    let range = dst.all();
                    let to = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                    transition(d, cb, dst.image, range, vk::ImageLayout::UNDEFINED, to);
                    if dst.aspect == vk::ImageAspectFlags::DEPTH {
                        d.cmd_clear_depth_stencil_image(
                            cb,
                            dst.image,
                            to,
                            &vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                            &[range],
                        );
                    } else {
                        d.cmd_clear_color_image(
                            cb,
                            dst.image,
                            to,
                            &vk::ClearColorValue { uint32: [0; 4] },
                            &[range],
                        );
                    }
                    transition(d, cb, dst.image, range, to, dst.home);
                }
            }
            full_barrier(d, cb);
        }
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
        let region = [vk::BufferCopy { src_offset, dst_offset, size }];
        self.hazard(&[vk::Handle::as_raw(src.raw.buffer)], &[vk::Handle::as_raw(dst.raw.buffer)]);
        // SAFETY: a copy between live buffers the encoder keeps.
        unsafe { self.d().cmd_copy_buffer(self.raw, src.raw.buffer, dst.raw.buffer, &region) };
        self.held.push(Held::Buffer(src.raw.clone()));
        self.held.push(Held::Buffer(dst.raw.clone()));
    }

    fn buffer_image(
        b: &BufferCopy<'_>,
        t: &TextureCopy<'_>,
        extent: [u32; 3],
    ) -> vk::BufferImageCopy {
        let texel = format_texel(t.texture.raw.format);
        let (layer, z) = match t.texture.dimension {
            TextureDimension::D3 => (0, t.origin[2]),
            TextureDimension::D2 => (t.origin[2], 0),
        };
        let layers = match t.texture.dimension {
            TextureDimension::D3 => 1,
            TextureDimension::D2 => extent[2],
        };
        let depth = match t.texture.dimension {
            TextureDimension::D3 => extent[2],
            TextureDimension::D2 => 1,
        };
        vk::BufferImageCopy {
            buffer_offset: b.offset,
            buffer_row_length: b.bytes_per_row / texel.0 * texel.1,
            buffer_image_height: b.rows_per_image * texel.1,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: t.texture.raw.aspect,
                mip_level: t.level,
                base_array_layer: layer,
                layer_count: layers,
            },
            image_offset: vk::Offset3D {
                x: t.origin[0] as i32,
                y: t.origin[1] as i32,
                z: z as i32,
            },
            image_extent: vk::Extent3D { width: extent[0], height: extent[1], depth },
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
        let region = [Self::buffer_image(&src, &dst, extent)];
        self.hazard(
            &[vk::Handle::as_raw(src.buffer.raw.buffer)],
            &[vk::Handle::as_raw(dst.texture.raw.image)],
        );
        let image = &dst.texture.raw;
        let sub = region[0].image_subresource;
        let part = image.part(sub.mip_level, sub.base_array_layer, sub.layer_count);
        let to = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        transition(self.d(), self.raw, image.image, part, image.home, to);
        // SAFETY: a copy between live resources the encoder keeps.
        unsafe {
            self.d().cmd_copy_buffer_to_image(
                self.raw,
                src.buffer.raw.buffer,
                image.image,
                to,
                &region,
            )
        };
        transition(self.d(), self.raw, image.image, part, to, image.home);
        self.held.push(Held::Buffer(src.buffer.raw.clone()));
        self.held.push(Held::Image(image.clone()));
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
        let region = [Self::buffer_image(&dst, &src, extent)];
        self.hazard(
            &[vk::Handle::as_raw(src.texture.raw.image)],
            &[vk::Handle::as_raw(dst.buffer.raw.buffer)],
        );
        let image = &src.texture.raw;
        let sub = region[0].image_subresource;
        let part = image.part(sub.mip_level, sub.base_array_layer, sub.layer_count);
        let from = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        transition(self.d(), self.raw, image.image, part, image.home, from);
        // SAFETY: a copy between live resources the encoder keeps.
        unsafe {
            self.d().cmd_copy_image_to_buffer(
                self.raw,
                image.image,
                from,
                dst.buffer.raw.buffer,
                &region,
            )
        };
        transition(self.d(), self.raw, image.image, part, from, image.home);
        self.held.push(Held::Buffer(dst.buffer.raw.clone()));
        self.held.push(Held::Image(image.clone()));
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
        let layers_of = |t: &TextureCopy<'_>| match t.texture.dimension {
            TextureDimension::D3 => (0, t.origin[2], 1),
            TextureDimension::D2 => (t.origin[2], 0, extent[2]),
        };
        let (sl, sz, n) = layers_of(&src);
        let (dl, dz, _) = layers_of(&dst);
        let depth = match src.texture.dimension {
            TextureDimension::D3 => extent[2],
            TextureDimension::D2 => 1,
        };
        let region = [vk::ImageCopy {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: src.texture.raw.aspect,
                mip_level: src.level,
                base_array_layer: sl,
                layer_count: n,
            },
            src_offset: vk::Offset3D {
                x: src.origin[0] as i32,
                y: src.origin[1] as i32,
                z: sz as i32,
            },
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: dst.texture.raw.aspect,
                mip_level: dst.level,
                base_array_layer: dl,
                layer_count: n,
            },
            dst_offset: vk::Offset3D {
                x: dst.origin[0] as i32,
                y: dst.origin[1] as i32,
                z: dz as i32,
            },
            extent: vk::Extent3D { width: extent[0], height: extent[1], depth },
        }];
        self.hazard(
            &[vk::Handle::as_raw(src.texture.raw.image)],
            &[vk::Handle::as_raw(dst.texture.raw.image)],
        );
        let (a, b) = (&src.texture.raw, &dst.texture.raw);
        let pa = a.part(src.level, sl, n);
        let pb = b.part(dst.level, dl, n);
        // (One image to another place of itself: the general layout for both.)
        let same = a.image == b.image;
        let (from, to) = if same {
            (vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL)
        } else {
            (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        };
        let (d, cb) = (self.d(), self.raw);
        transition(d, cb, a.image, pa, a.home, from);
        transition(d, cb, b.image, pb, b.home, to);
        // SAFETY: a copy between live images.
        unsafe { d.cmd_copy_image(cb, a.image, from, b.image, to, &region) };
        transition(d, cb, a.image, pa, from, a.home);
        transition(d, cb, b.image, pb, to, b.home);
        let (a, b) = (a.clone(), b.clone());
        self.held.push(Held::Image(a));
        self.held.push(Held::Image(b));
    }

    /// Timestamps `first..first + count` as 64-bit ticks into `dst` at `offset`.
    pub fn resolve_query_set(
        &mut self,
        queries: &QuerySet,
        first: u32,
        count: u32,
        dst: &Buffer,
        offset: u64,
    ) {
        self.barrier_all();
        // SAFETY: results of written queries, into a live buffer.
        unsafe {
            self.d().cmd_copy_query_pool_results(
                self.raw,
                queries.raw.pool,
                first,
                count,
                dst.raw.buffer,
                offset,
                8,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        };
        self.held.push(Held::Buffer(dst.raw.clone()));
        self.barrier_all();
    }

    fn timestamp(&mut self, queries: &QuerySet, index: u32, end: bool) {
        // SAFETY: a timestamp write into a pool; the query was reset before its pass.
        unsafe {
            self.d().cmd_write_timestamp2(
                self.raw,
                if end {
                    vk::PipelineStageFlags2::ALL_COMMANDS
                } else {
                    vk::PipelineStageFlags2::TOP_OF_PIPE
                },
                queries.raw.pool,
                index,
            )
        };
    }

    fn reset(&mut self, t: Option<PassTimestamps<'_>>) {
        if let Some(t) = t {
            // SAFETY: resets outside any pass.
            unsafe {
                self.d().cmd_reset_query_pool(self.raw, t.queries.raw.pool, t.begin, 1);
                self.d().cmd_reset_query_pool(self.raw, t.queries.raw.pool, t.end, 1);
            }
            self.timestamp(t.queries, t.begin, false);
        }
    }

    pub fn begin_compute_pass(
        &mut self,
        label: &str,
        timestamps: Option<PassTimestamps<'_>>,
    ) -> ComputePass<'_> {
        let _ = label;
        self.reset(timestamps);
        let end = timestamps.map(|t| (t.queries.raw.clone(), t.end));
        ComputePass { encoder: self, layout: None, end, bound: None }
    }

    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc<'_>) -> RenderPass<'_> {
        self.barrier_all();
        self.reset(desc.timestamps);
        // Its targets, out of their homes into the layouts drawing takes.
        let mut targets: Vec<(Arc<RawView>, vk::ImageLayout)> = desc
            .colors
            .iter()
            .map(|c| (c.view.raw.clone(), vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL))
            .collect();
        if let Some(d) = &desc.depth {
            targets.push((d.view.raw.clone(), vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL));
        }
        for (v, to) in &targets {
            let from = if v.image.swapchain { vk::ImageLayout::UNDEFINED } else { v.image.home };
            transition(self.d(), self.raw, v.image.image, v.range, from, *to);
        }
        let colors: Vec<vk::RenderingAttachmentInfo<'_>> = desc
            .colors
            .iter()
            .map(|c| {
                let a = vk::RenderingAttachmentInfo::default()
                    .image_view(c.view.raw.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .store_op(vk::AttachmentStoreOp::STORE);
                match c.load {
                    LoadOp::Load => a.load_op(vk::AttachmentLoadOp::LOAD),
                    LoadOp::Clear(v) => {
                        a.load_op(vk::AttachmentLoadOp::CLEAR).clear_value(vk::ClearValue {
                            color: vk::ClearColorValue { float32: v.map(|x| x as f32) },
                        })
                    }
                }
            })
            .collect();
        let depth = desc.depth.as_ref().map(|d| {
            let a = vk::RenderingAttachmentInfo::default()
                .image_view(d.view.raw.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .store_op(vk::AttachmentStoreOp::STORE);
            match d.load {
                LoadOp::Load => a.load_op(vk::AttachmentLoadOp::LOAD),
                LoadOp::Clear(v) => {
                    a.load_op(vk::AttachmentLoadOp::CLEAR).clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue { depth: v, stencil: 0 },
                    })
                }
            }
        });
        let (w, h) = desc
            .colors
            .first()
            .map(|c| (c.view.raw.width, c.view.raw.height))
            .or_else(|| desc.depth.as_ref().map(|d| (d.view.raw.width, d.view.raw.height)))
            .unwrap_or((1, 1));
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width: w, height: h },
        };
        let mut info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(&colors);
        if let Some(d) = &depth {
            info = info.depth_attachment(d);
        }
        for c in desc.colors {
            self.held.push(Held::View(c.view.raw.clone()));
        }
        if let Some(d) = &desc.depth {
            self.held.push(Held::View(d.view.raw.clone()));
        }
        // SAFETY: rendering to live views in GENERAL; the viewport and scissor cover the target
        // until the pass sets its own.
        unsafe {
            let d = self.d();
            d.cmd_begin_rendering(self.raw, &info);
            d.cmd_set_viewport(self.raw, 0, &[viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0)]);
            d.cmd_set_scissor(self.raw, 0, &[area]);
        }
        let end = desc.timestamps.map(|t| (t.queries.raw.clone(), t.end));
        RenderPass { encoder: self, layout: None, index: None, end, targets }
    }

    pub fn finish(mut self) -> CommandBuffer {
        // SAFETY: the recording ends.
        unsafe { self.d().end_command_buffer(self.raw).expect("Vulkan ended no command buffer") };
        let mut keep = std::mem::take(&mut self.keep);
        // Groups, pipelines and resources stay alive until the submission's done.
        for g in self.groups.drain(..) {
            for h in &g._held {
                if let Held::Buffer(b) = h {
                    keep.push(Keep::Buffer(b.clone()));
                }
            }
            keep.push(Keep::Group(g));
        }
        for p in self.pipelines.drain(..) {
            keep.push(Keep::Pipeline(p));
        }
        for h in self.held.drain(..) {
            keep.push(Keep::Held(h));
        }
        let pool = std::mem::replace(&mut self.pool, vk::CommandPool::null());
        CommandBuffer { pool, raw: self.raw, keep }
    }
}

impl Drop for CommandEncoder {
    fn drop(&mut self) {
        if self.pool != vk::CommandPool::null() {
            // SAFETY: never submitted: nothing runs it.
            unsafe { self.device.0.raw.destroy_command_pool(self.pool, None) };
        }
    }
}

pub struct ComputePass<'a> {
    encoder: &'a mut CommandEncoder,
    layout: Option<vk::PipelineLayout>,
    end: Option<(Arc<RawQueries>, u32)>,
    /// The bind group set: what the next dispatch uses.
    bound: Option<Arc<Group>>,
}

impl ComputePass<'_> {
    pub fn set_pipeline(&mut self, p: &ComputePipeline) {
        // SAFETY: a live pipeline the encoder keeps.
        unsafe {
            self.encoder.d().cmd_bind_pipeline(
                self.encoder.raw,
                vk::PipelineBindPoint::COMPUTE,
                p.raw.raw,
            )
        };
        self.layout = Some(p.layout.pipeline);
        self.encoder.pipelines.push(p.raw.clone());
    }

    pub fn set_bind_group(&mut self, group: &BindGroup, offsets: &[u32]) {
        let layout = self.layout.unwrap_or(group.0.layout.pipeline);
        if group.0.set != vk::DescriptorSet::null() {
            // SAFETY: a live set the encoder keeps.
            unsafe {
                self.encoder.d().cmd_bind_descriptor_sets(
                    self.encoder.raw,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    &[group.0.set],
                    offsets,
                )
            };
        }
        self.encoder.groups.push(group.0.clone());
        self.bound = Some(group.0.clone());
    }

    /// The bound group's resources a dispatch reads and writes, with `extra` read too.
    fn uses(&self, extra: Option<u64>) -> (Vec<u64>, Vec<u64>) {
        let mut reads: Vec<u64> = extra.into_iter().collect();
        let mut writes = Vec::new();
        for &(h, w) in self.bound.iter().flat_map(|g| g.uses.iter()) {
            if w { writes.push(h) } else { reads.push(h) }
        }
        (reads, writes)
    }

    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        let (reads, writes) = self.uses(None);
        self.encoder.hazard(&reads, &writes);
        // SAFETY: a dispatch of the bound pipeline, after any barrier its uses need.
        unsafe { self.encoder.d().cmd_dispatch(self.encoder.raw, x, y, z) };
    }

    pub fn dispatch_indirect(&mut self, buffer: &Buffer, offset: u64) {
        let (reads, writes) = self.uses(Some(vk::Handle::as_raw(buffer.raw.buffer)));
        self.encoder.hazard(&reads, &writes);
        // SAFETY: an indirect dispatch of the bound pipeline, from a live buffer.
        unsafe {
            self.encoder.d().cmd_dispatch_indirect(self.encoder.raw, buffer.raw.buffer, offset)
        };
        self.encoder.held.push(Held::Buffer(buffer.raw.clone()));
    }
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        if let Some((q, end)) = self.end.take() {
            // SAFETY: the pass's end timestamp.
            unsafe {
                self.encoder.d().cmd_write_timestamp2(
                    self.encoder.raw,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    q.pool,
                    end,
                )
            };
            self.encoder.keep.push(Keep::Queries(q));
        }
    }
}

pub struct RenderPass<'a> {
    encoder: &'a mut CommandEncoder,
    /// Its targets, and the layouts they're in while it draws: home again as it ends.
    targets: Vec<(Arc<RawView>, vk::ImageLayout)>,
    layout: Option<vk::PipelineLayout>,
    /// The index buffer and its offset, for the indexed draws after it.
    index: Option<(vk::Buffer, u64)>,
    end: Option<(Arc<RawQueries>, u32)>,
}

impl RenderPass<'_> {
    pub fn set_pipeline(&mut self, p: &RenderPipeline) {
        // SAFETY: a live pipeline the encoder keeps.
        unsafe {
            self.encoder.d().cmd_bind_pipeline(
                self.encoder.raw,
                vk::PipelineBindPoint::GRAPHICS,
                p.raw.raw,
            )
        };
        self.layout = Some(p.layout.pipeline);
        self.encoder.pipelines.push(p.raw.clone());
    }

    pub fn set_bind_group(&mut self, group: &BindGroup, offsets: &[u32]) {
        let layout = self.layout.unwrap_or(group.0.layout.pipeline);
        if group.0.set != vk::DescriptorSet::null() {
            // SAFETY: a live set the encoder keeps.
            unsafe {
                self.encoder.d().cmd_bind_descriptor_sets(
                    self.encoder.raw,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &[group.0.set],
                    offsets,
                )
            };
        }
        self.encoder.groups.push(group.0.clone());
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min: f32, max: f32) {
        // SAFETY: dynamic state.
        unsafe {
            self.encoder.d().cmd_set_viewport(
                self.encoder.raw,
                0,
                &[viewport(x, y, w, h, min, max)],
            )
        };
    }

    pub fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        // SAFETY: dynamic state.
        unsafe {
            self.encoder.d().cmd_set_scissor(
                self.encoder.raw,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: x as i32, y: y as i32 },
                    extent: vk::Extent2D { width: w, height: h },
                }],
            )
        };
    }

    pub fn set_index_buffer(&mut self, buffer: &Buffer, offset: u64) {
        self.index = Some((buffer.raw.buffer, offset));
        self.encoder.held.push(Held::Buffer(buffer.raw.clone()));
        // SAFETY: a live buffer of u32 indices.
        unsafe {
            self.encoder.d().cmd_bind_index_buffer(
                self.encoder.raw,
                buffer.raw.buffer,
                offset,
                vk::IndexType::UINT32,
            )
        };
    }

    pub fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        // SAFETY: a draw of the bound pipeline.
        unsafe {
            self.encoder.d().cmd_draw(
                self.encoder.raw,
                vertices.len() as u32,
                instances.len() as u32,
                vertices.start,
                instances.start,
            )
        };
    }

    pub fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        let _ = self.index.expect("an index buffer is set");
        // SAFETY: an indexed draw of the bound pipeline.
        unsafe {
            self.encoder.d().cmd_draw_indexed(
                self.encoder.raw,
                indices.len() as u32,
                instances.len() as u32,
                indices.start,
                base_vertex,
                instances.start,
            )
        };
    }

    pub fn draw_indirect(&mut self, buffer: &Buffer, offset: u64) {
        self.multi_draw_indirect(buffer, offset, 1);
    }

    pub fn draw_indexed_indirect(&mut self, buffer: &Buffer, offset: u64) {
        self.multi_draw_indexed_indirect(buffer, offset, 1);
    }

    /// `count` indirect draws, their arguments 16 bytes apart from `offset`.
    pub fn multi_draw_indirect(&mut self, buffer: &Buffer, offset: u64, count: u32) {
        self.encoder.held.push(Held::Buffer(buffer.raw.clone()));
        // SAFETY: indirect draws from a live buffer.
        unsafe {
            self.encoder.d().cmd_draw_indirect(
                self.encoder.raw,
                buffer.raw.buffer,
                offset,
                count,
                16,
            )
        };
    }

    /// `count` indexed indirect draws, their arguments 20 bytes apart from `offset`.
    pub fn multi_draw_indexed_indirect(&mut self, buffer: &Buffer, offset: u64, count: u32) {
        let _ = self.index.expect("an index buffer is set");
        self.encoder.held.push(Held::Buffer(buffer.raw.clone()));
        // SAFETY: indexed indirect draws from a live buffer.
        unsafe {
            self.encoder.d().cmd_draw_indexed_indirect(
                self.encoder.raw,
                buffer.raw.buffer,
                offset,
                count,
                20,
            )
        };
    }
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        // SAFETY: the pass ends; what it wrote, the next command sees.
        unsafe { self.encoder.d().cmd_end_rendering(self.encoder.raw) };
        if let Some((q, end)) = self.end.take() {
            // SAFETY: the pass's end timestamp.
            unsafe {
                self.encoder.d().cmd_write_timestamp2(
                    self.encoder.raw,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    q.pool,
                    end,
                )
            };
            self.encoder.keep.push(Keep::Queries(q));
        }
        for (v, from) in std::mem::take(&mut self.targets) {
            transition(
                self.encoder.d(),
                self.encoder.raw,
                v.image.image,
                v.range,
                from,
                v.image.home,
            );
        }
        self.encoder.barrier_all();
    }
}

// ---- conversions ----------------------------------------------------------------------------------

/// A WebGPU viewport as Vulkan's: flipped, a negative height from its bottom edge, as wgpu-hal
/// does, since WebGPU's clip space has y up and Vulkan's down (and the compiler's SPIR-V writes
/// WebGPU's positions).
fn viewport(x: f32, y: f32, w: f32, h: f32, min: f32, max: f32) -> vk::Viewport {
    vk::Viewport { x, y: y + h, width: w, height: -h, min_depth: min, max_depth: max }
}

pub fn vk_format(f: TextureFormat) -> vk::Format {
    match f {
        TextureFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
        TextureFormat::R16Float => vk::Format::R16_SFLOAT,
        TextureFormat::Rg16Float => vk::Format::R16G16_SFLOAT,
        TextureFormat::R32Float => vk::Format::R32_SFLOAT,
        TextureFormat::R32Uint => vk::Format::R32_UINT,
        TextureFormat::R8 => vk::Format::R8_UNORM,
        TextureFormat::Rg8 => vk::Format::R8G8_UNORM,
        TextureFormat::Rgb10a2 => vk::Format::A2B10G10R10_UNORM_PACK32,
        TextureFormat::Rg32Float => vk::Format::R32G32_SFLOAT,
        TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        TextureFormat::Rg32Uint => vk::Format::R32G32_UINT,
        TextureFormat::Rg11b10Float => vk::Format::B10G11R11_UFLOAT_PACK32,
        TextureFormat::Bc1 => vk::Format::BC1_RGBA_UNORM_BLOCK,
        TextureFormat::Bc4 => vk::Format::BC4_UNORM_BLOCK,
        TextureFormat::Bc5 => vk::Format::BC5_UNORM_BLOCK,
        TextureFormat::Bc7 => vk::Format::BC7_UNORM_BLOCK,
        TextureFormat::Etc2Rgb8 => vk::Format::ETC2_R8G8B8_UNORM_BLOCK,
        TextureFormat::Astc4x4 => vk::Format::ASTC_4X4_UNORM_BLOCK,
    }
}

/// A format's bytes a block and its block's side in texels (1 for an uncompressed format), for
/// a copy's row length in texels.
fn format_texel(f: vk::Format) -> (u32, u32) {
    match f {
        vk::Format::R8_UNORM => (1, 1),
        vk::Format::R8G8_UNORM | vk::Format::R16_SFLOAT => (2, 1),
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R16G16_SFLOAT
        | vk::Format::R32_SFLOAT
        | vk::Format::R32_UINT
        | vk::Format::A2B10G10R10_UNORM_PACK32
        | vk::Format::B10G11R11_UFLOAT_PACK32
        | vk::Format::D32_SFLOAT => (4, 1),
        vk::Format::R16G16B16A16_SFLOAT | vk::Format::R32G32_SFLOAT | vk::Format::R32G32_UINT => {
            (8, 1)
        }
        vk::Format::R32G32B32A32_SFLOAT => (16, 1),
        vk::Format::BC1_RGBA_UNORM_BLOCK
        | vk::Format::BC4_UNORM_BLOCK
        | vk::Format::ETC2_R8G8B8_UNORM_BLOCK => (8, 4),
        _ => (16, 4),
    }
}

fn compare_op(c: Compare) -> vk::CompareOp {
    match c {
        Compare::Never => vk::CompareOp::NEVER,
        Compare::Less => vk::CompareOp::LESS,
        Compare::Equal => vk::CompareOp::EQUAL,
        Compare::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        Compare::Greater => vk::CompareOp::GREATER,
        Compare::NotEqual => vk::CompareOp::NOT_EQUAL,
        Compare::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        Compare::Always => vk::CompareOp::ALWAYS,
    }
}

fn descriptor_type(t: BindingType) -> vk::DescriptorType {
    match t {
        BindingType::Uniform { dynamic: true, .. } => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        BindingType::Uniform { dynamic: false, .. } => vk::DescriptorType::UNIFORM_BUFFER,
        BindingType::Storage { dynamic: true, .. } => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
        BindingType::Storage { dynamic: false, .. } => vk::DescriptorType::STORAGE_BUFFER,
        BindingType::Texture { .. } => vk::DescriptorType::SAMPLED_IMAGE,
        BindingType::StorageTexture { .. } => vk::DescriptorType::STORAGE_IMAGE,
        BindingType::Sampler { .. } => vk::DescriptorType::SAMPLER,
    }
}

fn stages(s: Stages) -> vk::ShaderStageFlags {
    let mut f = vk::ShaderStageFlags::empty();
    if s.vertex {
        f |= vk::ShaderStageFlags::VERTEX;
    }
    if s.fragment {
        f |= vk::ShaderStageFlags::FRAGMENT;
    }
    if s.compute {
        f |= vk::ShaderStageFlags::COMPUTE;
    }
    f
}

fn blend_state(b: Blend) -> vk::PipelineColorBlendAttachmentState {
    let state = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let Some((c, a)) = b.components() else { return state };
    state
        .blend_enable(true)
        .src_color_blend_factor(blend_factor(c.0))
        .dst_color_blend_factor(blend_factor(c.1))
        .color_blend_op(blend_op(c.2))
        .src_alpha_blend_factor(blend_factor(a.0))
        .dst_alpha_blend_factor(blend_factor(a.1))
        .alpha_blend_op(blend_op(a.2))
}

fn blend_factor(name: &str) -> vk::BlendFactor {
    match name {
        "zero" => vk::BlendFactor::ZERO,
        "one" => vk::BlendFactor::ONE,
        "src-alpha" => vk::BlendFactor::SRC_ALPHA,
        "one-minus-src-alpha" => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        "dst" => vk::BlendFactor::DST_COLOR,
        "dst-alpha" => vk::BlendFactor::DST_ALPHA,
        other => unreachable!("Blend::components names no factor `{other}`"),
    }
}

fn blend_op(name: &str) -> vk::BlendOp {
    match name {
        "add" => vk::BlendOp::ADD,
        "min" => vk::BlendOp::MIN,
        "max" => vk::BlendOp::MAX,
        other => unreachable!("Blend::components names no operation `{other}`"),
    }
}

// ---- a window's surface -------------------------------------------------------------------------

/// A window's surface and its swapchain. Its frames are `R8G8B8A8_UNORM`, the screen's format,
/// so presenting a frame is drawing the screen into it.
pub struct Surface {
    device: Device,
    surface: vk::SurfaceKHR,
    surface_fns: ash::khr::surface::Instance,
    swapchain_fns: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<Arc<RawView>>,
    /// Semaphores an acquisition signals, used in turn, each with the submission that last
    /// waited on it (reused once that's done).
    acquired: Vec<(vk::Semaphore, SubmissionIndex)>,
    next: usize,
    /// One a submission signals for each image, which its presentation waits on.
    rendered: Vec<vk::Semaphore>,
    /// The window, alive while its surface is.
    _window: Arc<dyn Window>,
}

/// A frame of a surface to draw into, then [`Surface::present`].
pub struct SurfaceTexture {
    index: u32,
    acquired: vk::Semaphore,
    rendered: vk::Semaphore,
    slot: usize,
    pub texture: Texture,
    pub view: TextureView,
}

impl Device {
    /// A surface of `window` (Wayland, X11).
    pub fn create_surface(&self, window: Arc<dyn Window>) -> Result<Surface> {
        if !self.0.swapchain {
            return Err("this Vulkan has no surface or swapchain extension".into());
        }
        let display = window.display_handle().map_err(|e| Error(format!("the window: {e}")))?;
        let handle = window.window_handle().map_err(|e| Error(format!("the window: {e}")))?;
        // SAFETY: the handles are the window's, alive while the surface is (it keeps the window).
        let surface = unsafe {
            ash_window::create_surface(
                &self.0.entry,
                &self.0.instance,
                display.as_raw(),
                handle.as_raw(),
                None,
            )
        }
        .map_err(|e| Error(format!("Vulkan made no surface: {e}")))?;
        let surface_fns = ash::khr::surface::Instance::new(&self.0.entry, &self.0.instance);
        // SAFETY: a query of the surface.
        let ok = unsafe {
            surface_fns.get_physical_device_surface_support(
                self.0.physical,
                self.0.queue_family,
                surface,
            )
        }
        .unwrap_or(false);
        if !ok {
            // SAFETY: the surface was made above and nothing uses it.
            unsafe { surface_fns.destroy_surface(surface, None) };
            return Err("the GPU's queue can't present to this window".into());
        }
        let swapchain_fns = ash::khr::swapchain::Device::new(&self.0.instance, &self.0.raw);
        Ok(Surface {
            device: self.clone(),
            surface,
            surface_fns,
            swapchain_fns,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            acquired: Vec::new(),
            next: 0,
            rendered: Vec::new(),
            _window: window,
        })
    }

    /// Submits `buffers` for `frame`: after its image is acquired, signalling what its
    /// presentation waits on.
    pub fn submit_frame(
        &self,
        buffers: Vec<CommandBuffer>,
        frame: &SurfaceTexture,
    ) -> SubmissionIndex {
        self.submit_waiting(buffers, Some((frame.acquired, frame.rendered)))
    }
}

impl Surface {
    /// Its frames' format.
    pub fn format(&self) -> TextureFormat {
        TextureFormat::Rgba8
    }

    pub fn configure(&mut self, device: &Device, c: &SurfaceConfig) -> Result<()> {
        let d = &device.0;
        // SAFETY: queries of the surface, then a swapchain of it (the old one retired once the
        // device is idle), and semaphores.
        unsafe {
            let caps = self
                .surface_fns
                .get_physical_device_surface_capabilities(d.physical, self.surface)
                .map_err(|e| Error(format!("the surface: {e}")))?;
            let formats = self
                .surface_fns
                .get_physical_device_surface_formats(d.physical, self.surface)
                .map_err(|e| Error(format!("the surface: {e}")))?;
            let format = formats
                .iter()
                .find(|f| f.format == vk::Format::R8G8B8A8_UNORM)
                .copied()
                .ok_or("the surface has no R8G8B8A8_UNORM format")?;
            let modes = self
                .surface_fns
                .get_physical_device_surface_present_modes(d.physical, self.surface)
                .map_err(|e| Error(format!("the surface: {e}")))?;
            let want = match c.present {
                PresentMode::Fifo => vk::PresentModeKHR::FIFO,
                PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
                PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
                PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
            };
            // FIFO is every surface's.
            let mode = if modes.contains(&want) { want } else { vk::PresentModeKHR::FIFO };
            let extent = if caps.current_extent.width == u32::MAX {
                vk::Extent2D { width: c.width, height: c.height }
            } else {
                caps.current_extent
            };
            let mut count = c.frames.clamp(2, 3).max(caps.min_image_count);
            if caps.max_image_count > 0 {
                count = count.min(caps.max_image_count);
            }
            let old = self.swapchain;
            let _ = d.raw.device_wait_idle();
            let swapchain = self
                .swapchain_fns
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .surface(self.surface)
                        .min_image_count(count)
                        .image_format(format.format)
                        .image_color_space(format.color_space)
                        .image_extent(extent)
                        .image_array_layers(1)
                        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .pre_transform(caps.current_transform)
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .present_mode(mode)
                        .clipped(true)
                        .old_swapchain(old),
                    None,
                )
                .map_err(|e| Error(format!("Vulkan made no swapchain: {e}")))?;
            self.release();
            if old != vk::SwapchainKHR::null() {
                self.swapchain_fns.destroy_swapchain(old, None);
            }
            self.swapchain = swapchain;
            let images = self
                .swapchain_fns
                .get_swapchain_images(swapchain)
                .map_err(|e| Error(format!("the swapchain: {e}")))?;
            for image in images {
                let raw = Arc::new(RawImage {
                    device: d.clone(),
                    image,
                    memory: vk::DeviceMemory::null(),
                    format: format.format,
                    aspect: vk::ImageAspectFlags::COLOR,
                    levels: 1,
                    layers: 1,
                    home: vk::ImageLayout::PRESENT_SRC_KHR,
                    swapchain: true,
                });
                let range = raw.part(0, 0, 1);
                let view = d
                    .raw
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format.format)
                            .subresource_range(range),
                        None,
                    )
                    .map_err(|e| Error(format!("Vulkan made no view: {e}")))?;
                self.images.push(Arc::new(RawView {
                    image: raw,
                    view,
                    range,
                    width: extent.width,
                    height: extent.height,
                }));
                self.rendered.push(
                    d.raw
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .map_err(|e| Error(format!("Vulkan made no semaphore: {e}")))?,
                );
            }
            for _ in 0..self.images.len() + 1 {
                let sem = d
                    .raw
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .map_err(|e| Error(format!("Vulkan made no semaphore: {e}")))?;
                self.acquired.push((sem, 0));
            }
        }
        Ok(())
    }

    /// The swapchain's images, views and semaphores let go (the device idle).
    fn release(&mut self) {
        let d = &self.device.0.raw;
        // SAFETY: the device is idle; nothing uses them.
        unsafe {
            let _ = d.device_wait_idle();
            for (s, _) in self.acquired.drain(..) {
                d.destroy_semaphore(s, None);
            }
            for s in self.rendered.drain(..) {
                d.destroy_semaphore(s, None);
            }
        }
        self.images.clear();
        self.next = 0;
    }

    /// The next frame to draw into (waits for an image the display is done with). An error
    /// when the window changed and the surface wants configuring again.
    pub fn acquire(&mut self) -> Result<SurfaceTexture> {
        let slot = self.next % self.acquired.len();
        self.next += 1;
        let (sem, used) = self.acquired[slot];
        // The semaphore's last wait must be done before it's signalled again.
        self.device.wait(used);
        // SAFETY: an acquisition signalling an unsignalled semaphore.
        let (index, _) = unsafe {
            self.swapchain_fns.acquire_next_image(self.swapchain, u64::MAX, sem, vk::Fence::null())
        }
        .map_err(|e| Error(format!("the swapchain: {e}")))?;
        let view = self.images[index as usize].clone();
        let texture = Texture {
            raw: view.image.clone(),
            format: TextureFormat::Rgba8,
            width: view.width,
            height: view.height,
            depth_or_layers: 1,
            levels: 1,
            dimension: TextureDimension::D2,
        };
        Ok(SurfaceTexture {
            index,
            acquired: sem,
            rendered: self.rendered[index as usize],
            slot,
            texture,
            view: TextureView { raw: view },
        })
    }

    /// Shows `frame` once the submission for it ([`Device::submit_frame`]) is done.
    pub fn present(&mut self, frame: SurfaceTexture) -> Result<()> {
        self.acquired[frame.slot].1 = self.device.submitted();
        let waits = [frame.rendered];
        let chains = [self.swapchain];
        let indices = [frame.index];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&chains)
            .image_indices(&indices);
        // SAFETY: the image was acquired and its rendering submitted, signalling `rendered`.
        unsafe { self.swapchain_fns.queue_present(self.device.0.queue, &info) }
            .map(|_| ())
            .map_err(|e| Error(format!("the swapchain: {e}")))
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        self.release();
        // SAFETY: the device is idle; the swapchain and then its surface go.
        unsafe {
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_fns.destroy_swapchain(self.swapchain, None);
            }
            self.surface_fns.destroy_surface(self.surface, None);
        }
    }
}
