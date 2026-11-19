//! The device's one table of textures and samplers: how a pipeline's textures and samplers reach
//! its shaders, the rule the compiler writes its shaders by and the host binds by, on every target.
//!
//! - **Every texture view and sampler has an index in one table a device**, given when it's made
//!   and taken back when it's let go. Vulkan: one descriptor set ([`TABLE_SET`]), its bindings
//!   arrays of every sampled image ([`SAMPLED`]), every storage image ([`STORAGE`]) and every
//!   sampler ([`SAMPLERS`]) at the same indices; a shader declares one array a type of image it
//!   reads, each aliasing its binding. Metal: one buffer of `MTLResourceID`s at buffer slot
//!   [`METAL_TABLE`] of every stage, a shader reading it as an array of the type it takes.
//! - **A dispatch or draw gets its textures' and samplers' indices in a block**: one `u32` each,
//!   in the manifest's order ([`indexed`]); push constants on Vulkan, `setBytes` at buffer slot
//!   [`METAL_BLOCK`] on Metal.
//! - **Buffers are not in the table**: they stay where [`crate::metal`] and [`crate::vulkan`] put
//!   them, bound a dispatch or draw (the uniform block at a dynamic offset). A buffer reached by
//!   its address cost a compute pass on the Radeon up to a quarter of its time (backend.md, 2c's
//!   third session), so buffers stay descriptors.
//!
//! Which pipelines reach their textures this way is [`by_table`]'s to say. The engine's case (many
//! 3D textures, one reached by an index a kernel reads from a buffer) is a read of the table at
//! that index: no binding a texture.

use crate::manifest::{BindingKind, Pipeline, Stage};

/// Vulkan: the table's descriptor set. Set 0 is the pipeline's own (its buffers).
pub const TABLE_SET: u32 = 1;
/// Vulkan: the table's binding of sampled images (every texture a shader samples or reads).
pub const SAMPLED: u32 = 0;
/// Vulkan: the table's binding of storage images (every texture a kernel writes).
pub const STORAGE: u32 = 1;
/// Vulkan: the table's binding of samplers.
pub const SAMPLERS: u32 = 2;
/// Metal: the buffer slot of a stage's block of indices (`setBytes`): the argument table's last
/// two slots are the block's and the table's, so a stage that reaches the table has 29 of
/// [`crate::TargetLimits::buffer_slots_per_stage`]'s 31 for its own buffers.
pub const METAL_BLOCK: u32 = 29;
/// Metal: the buffer slot of the table of resource IDs.
pub const METAL_TABLE: u32 = 30;
/// The most bytes a block of indices may be: Vulkan's least `maxPushConstantsSize`.
pub const BLOCK_BYTES: u32 = 128;

/// Whether pipeline `p` reaches its textures and samplers through the table (else each is bound
/// at a slot of its own, as [`crate::metal`] and [`crate::vulkan`] say): a kernel does; a draw
/// doesn't. Draws through the table cost the M4's opening 1.3% of its frame (its last pass, 3.7%:
/// a texture read through the table is a load more before the sample), past 2c's 1% line, so
/// draws keep their slots (backend.md, the sixth session).
pub fn by_table(p: &Pipeline) -> bool {
    matches!(p.stage, Stage::Compute { .. })
}

/// Whether a binding of `kind` is in the table: a texture or a sampler.
pub fn in_table(kind: BindingKind) -> bool {
    !kind.is_buffer()
}

/// The bindings of `p` it reaches through the table, in the manifest's order: binding
/// `indexed(p)[k]`'s index is the block's `u32` at byte `4 k`. Empty where [`by_table`] isn't.
pub fn indexed(p: &Pipeline) -> Vec<u32> {
    if !by_table(p) {
        return Vec::new();
    }
    p.bindings.iter().filter(|b| in_table(b.kind)).map(|b| b.binding).collect()
}

/// The byte of the block that holds `binding`'s index, if `p` reaches it through the table.
pub fn index_offset(p: &Pipeline, binding: u32) -> Option<u32> {
    let k = indexed(p).iter().position(|&b| b == binding)?;
    u32::try_from(4 * k).ok()
}

/// Which of Vulkan's table bindings a texture or sampler of `kind` is in; `None` for a buffer.
pub fn table_of(kind: BindingKind) -> Option<u32> {
    match kind {
        BindingKind::Read | BindingKind::ReadWrite => None,
        BindingKind::Sampler
        | BindingKind::ComparisonSampler
        | BindingKind::NonFilteringSampler => Some(SAMPLERS),
        k if k.is_storage() => Some(STORAGE),
        _ => Some(SAMPLED),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{BindingStage, ResourceBinding};

    fn binding(binding: u32, kind: BindingKind) -> ResourceBinding {
        ResourceBinding { binding, kind, stage: BindingStage::Both, format: None }
    }

    /// Textures and samplers in the manifest's order, a `u32` each; buffers not in the block.
    #[test]
    fn a_block_holds_each_texture_and_sampler_in_order() {
        let mut p = crate::vectors::sample_manifest().pipelines.remove(0);
        p.bindings = vec![
            binding(1, BindingKind::Read),
            binding(2, BindingKind::Texture3d),
            binding(3, BindingKind::ReadWrite),
            binding(4, BindingKind::Sampler),
            binding(5, BindingKind::StorageTexture),
        ];
        assert!(by_table(&p));
        assert_eq!(indexed(&p), [2, 4, 5]);
        assert_eq!(index_offset(&p, 2), Some(0));
        assert_eq!(index_offset(&p, 4), Some(4));
        assert_eq!(index_offset(&p, 5), Some(8));
        assert_eq!(index_offset(&p, 1), None);
    }

    #[test]
    fn textures_and_samplers_have_a_table_and_buffers_none() {
        assert_eq!(table_of(BindingKind::Read), None);
        assert_eq!(table_of(BindingKind::Texture3d), Some(SAMPLED));
        assert_eq!(table_of(BindingKind::DepthTextureArray), Some(SAMPLED));
        assert_eq!(table_of(BindingKind::StorageTexture3d), Some(STORAGE));
        assert_eq!(table_of(BindingKind::ComparisonSampler), Some(SAMPLERS));
    }
}
