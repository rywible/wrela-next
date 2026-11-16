//! How a pipeline's bindings reach a Metal shader: the one rule the compiler writes its Metal
//! shading language (MSL) by and the host binds by.
//!
//! The host hands wgpu each shader as finished MSL (`create_shader_module_passthrough`), so wgpu
//! no longer chooses where a binding goes in Metal's argument tables: the MSL names each slot
//! (`[[buffer(n)]]`, `[[texture(n)]]`, `[[sampler(n)]]`), and the host's bind group layout must
//! put each binding there. Both follow [`layout`] and [`slots`]:
//!
//! - [`layout`] is the pipeline's bind group 0 (its only one): each binding, what it is, and the
//!   shaders that see it. The host makes its `wgpu::BindGroupLayout` from it.
//! - [`slots`] is where wgpu 30's Metal backend puts those bindings in one shader stage's
//!   argument tables (`wgpu-hal` 30, `metal/device.rs`, `create_pipeline_layout`, read, and
//!   checked against what it wrote for WGSL): the stage's own counters of buffers, textures and
//!   samplers, each binding taking the next of its kind in ascending binding number, counted
//!   only in the stages that see it; no immediates; then, in a stage that sees a storage buffer
//!   (and in every vertex stage), one more buffer slot that wgpu-hal sets aside for the sizes of
//!   its runtime-sized buffers, which it fills only for shaders it translated itself.
//! - So a passthrough shader gets its buffers' sizes another way: a uniform buffer of `u32`s, the
//!   pipeline's [`crate::manifest::BufferSizes`], bound at a binding of its own after the rest.
//!   naga's MSL reads it as its `_mslBufferSizes` (one member for each global holding a
//!   runtime-sized array, in the module's order); the host writes each listed binding's bound
//!   size there when it makes a bind group.

use crate::manifest::{BindingKind, BindingStage, Pipeline, Stage, UniformSpace};

/// The shader stages that see a binding (wgpu's `ShaderStages`, which `wrela_abi` can't name).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Stages {
    pub vertex: bool,
    pub fragment: bool,
    pub compute: bool,
}

impl Stages {
    pub const COMPUTE: Stages = Stages { vertex: false, fragment: false, compute: true };
    pub const VERTEX: Stages = Stages { vertex: true, fragment: false, compute: false };
    pub const FRAGMENT: Stages = Stages { vertex: false, fragment: true, compute: false };
    pub const VERTEX_FRAGMENT: Stages = Stages { vertex: true, fragment: true, compute: false };

    /// Whether it holds `stage`.
    pub fn sees(self, stage: ShaderStage) -> bool {
        match stage {
            ShaderStage::Vertex => self.vertex,
            ShaderStage::Fragment => self.fragment,
            ShaderStage::Compute => self.compute,
        }
    }
}

/// One shader stage of a pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

/// What a binding of bind group 0 is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entry {
    /// The uniform block: a uniform buffer, or a read-only storage buffer, at a dynamic offset.
    Uniforms { space: UniformSpace, size: u32 },
    /// A debug build's flag: a read-write storage buffer the host makes.
    DebugFlag,
    /// One of the manifest's bindings.
    Binding(BindingKind),
    /// The buffers' sizes: a uniform buffer of `count` `u32`s the host makes for each bind group.
    Sizes { count: u32 },
}

/// A binding of bind group 0: its number, what it is and the shaders that see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutEntry {
    pub binding: u32,
    pub entry: Entry,
    pub visibility: Stages,
}

impl LayoutEntry {
    /// Which argument table it takes a slot of, and whether the shader may write what's there.
    fn class(&self) -> (Table, bool) {
        match self.entry {
            Entry::Uniforms { .. } => (Table::Buffer, false),
            Entry::DebugFlag => (Table::Buffer, true),
            Entry::Sizes { .. } => (Table::Buffer, false),
            Entry::Binding(kind) => match kind {
                BindingKind::Read => (Table::Buffer, false),
                BindingKind::ReadWrite => (Table::Buffer, true),
                BindingKind::Sampler
                | BindingKind::ComparisonSampler
                | BindingKind::NonFilteringSampler => (Table::Sampler, false),
                k if k.is_storage() => (Table::Texture, true),
                _ => (Table::Texture, false),
            },
        }
    }

    /// Whether it's a storage buffer (whose stages wgpu-hal sets a sizes slot aside in).
    fn is_storage_buffer(&self) -> bool {
        match self.entry {
            Entry::Uniforms { space, .. } => space == UniformSpace::Storage,
            Entry::DebugFlag => true,
            Entry::Binding(kind) => kind.is_buffer(),
            Entry::Sizes { .. } => false,
        }
    }
}

/// Metal's three argument tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Table {
    Buffer,
    Texture,
    Sampler,
}

/// The pipeline's bind group 0, in ascending binding number: the uniform block, the debug flag,
/// the manifest's bindings and the buffers' sizes, each seen by the shaders that read it. A
/// kernel's are its own; in a render pipeline a binding names the shaders that read it, but a
/// storage buffer the pipeline writes is the fragment shader's alone, and a debug flag is too
/// (a vertex shader doesn't write), and a storage texture only a kernel writes.
pub fn layout(p: &Pipeline) -> Vec<LayoutEntry> {
    let compute = matches!(p.stage, Stage::Compute { .. });
    let all = if compute { Stages::COMPUTE } else { Stages::VERTEX_FRAGMENT };
    let mut out = Vec::with_capacity(p.bindings.len() + 3);
    if let Some(u) = &p.uniform {
        out.push(LayoutEntry {
            binding: u.binding,
            entry: Entry::Uniforms { space: u.space, size: u.size },
            visibility: all,
        });
    }
    if let Some(binding) = p.debug_flag {
        let visibility = if compute { all } else { Stages::FRAGMENT };
        out.push(LayoutEntry { binding, entry: Entry::DebugFlag, visibility });
    }
    for b in &p.bindings {
        let seen = match b.stage {
            BindingStage::Both => all,
            BindingStage::Vertex => Stages::VERTEX,
            BindingStage::Fragment => Stages::FRAGMENT,
        };
        let visibility = match b.kind {
            BindingKind::ReadWrite if !compute => Stages::FRAGMENT,
            k if k.is_storage() => Stages::COMPUTE,
            _ => seen,
        };
        out.push(LayoutEntry { binding: b.binding, entry: Entry::Binding(b.kind), visibility });
    }
    if let Some(s) = &p.sizes {
        let count = s.buffers.len() as u32;
        out.push(LayoutEntry {
            binding: s.binding,
            entry: Entry::Sizes { count },
            visibility: all,
        });
    }
    out.sort_by_key(|e| e.binding);
    out
}

/// Where a binding is in one stage's argument tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub binding: u32,
    pub table: Table,
    pub index: u8,
    /// Whether the shader may write it (naga's `BindTarget::mutable`).
    pub mutable: bool,
}

/// One stage's slots: each binding it sees, and the buffer slot wgpu-hal sets aside for sizes
/// it fills only for its own translations (`None` where it sets none aside), which a
/// passthrough shader leaves alone.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StageSlots {
    pub slots: Vec<Slot>,
    pub set_aside: Option<u8>,
}

impl StageSlots {
    /// The slot of `binding`, if the stage sees it.
    pub fn of(&self, binding: u32) -> Option<&Slot> {
        self.slots.iter().find(|s| s.binding == binding)
    }

    /// How many buffer slots the stage uses, with the one set aside.
    pub fn buffers(&self) -> usize {
        self.slots.iter().filter(|s| s.table == Table::Buffer).count()
            + usize::from(self.set_aside.is_some())
    }
}

/// Where wgpu 30's Metal backend puts `layout`'s bindings in `stage`'s argument tables (the
/// module's header says how).
pub fn slots(layout: &[LayoutEntry], stage: ShaderStage) -> StageSlots {
    let mut counters = [0u8; 3];
    let mut out = StageSlots::default();
    let mut sizes = stage == ShaderStage::Vertex;
    for e in layout {
        if !e.visibility.sees(stage) {
            continue;
        }
        sizes |= e.is_storage_buffer();
        let (table, mutable) = e.class();
        let counter = &mut counters[table as usize];
        out.slots.push(Slot { binding: e.binding, table, index: *counter, mutable });
        *counter += 1;
    }
    if sizes {
        out.set_aside = Some(counters[Table::Buffer as usize]);
    }
    out
}

/// The stages of a pipeline: a kernel's, or a vertex and a fragment shader's.
pub fn stages(p: &Pipeline) -> &'static [ShaderStage] {
    match p.stage {
        Stage::Compute { .. } => &[ShaderStage::Compute],
        Stage::Render { .. } => &[ShaderStage::Vertex, ShaderStage::Fragment],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{BufferSizes, ResourceBinding, UniformBlock};

    fn binding(binding: u32, kind: BindingKind, stage: BindingStage) -> ResourceBinding {
        ResourceBinding { binding, kind, stage, format: None }
    }

    fn render(bindings: Vec<ResourceBinding>) -> Pipeline {
        let m = crate::vectors::sample_manifest();
        let mut p =
            m.pipelines.into_iter().find(|p| matches!(p.stage, Stage::Render { .. })).expect("one");
        p.bindings = bindings;
        p.debug_flag = None;
        p
    }

    /// A render pipeline's slots: each stage counts only what it sees, in binding order, each
    /// kind its own counter; a written buffer is the fragment shader's; the sizes binding and
    /// the slot wgpu-hal sets aside come after the rest.
    #[test]
    fn each_stage_counts_what_it_sees() {
        let mut p = render(vec![
            binding(1, BindingKind::Read, BindingStage::Both),
            binding(2, BindingKind::Texture, BindingStage::Fragment),
            binding(3, BindingKind::Sampler, BindingStage::Fragment),
            binding(4, BindingKind::ReadWrite, BindingStage::Both),
            binding(5, BindingKind::Read, BindingStage::Vertex),
        ]);
        p.uniform = Some(UniformBlock { binding: 0, size: 16, space: UniformSpace::Uniform });
        p.sizes = Some(BufferSizes { binding: 6, buffers: vec![1, 4, 5] });
        let l = layout(&p);
        let v = slots(&l, ShaderStage::Vertex);
        let at = |s: &StageSlots, b| s.of(b).map(|s| (s.table, s.index, s.mutable));
        assert_eq!(at(&v, 0), Some((Table::Buffer, 0, false)));
        assert_eq!(at(&v, 1), Some((Table::Buffer, 1, false)));
        assert_eq!(at(&v, 2), None);
        assert_eq!(at(&v, 4), None, "a written buffer is the fragment shader's");
        assert_eq!(at(&v, 5), Some((Table::Buffer, 2, false)));
        assert_eq!(at(&v, 6), Some((Table::Buffer, 3, false)));
        assert_eq!(v.set_aside, Some(4));
        let f = slots(&l, ShaderStage::Fragment);
        assert_eq!(at(&f, 1), Some((Table::Buffer, 1, false)));
        assert_eq!(at(&f, 2), Some((Table::Texture, 0, false)));
        assert_eq!(at(&f, 3), Some((Table::Sampler, 0, false)));
        assert_eq!(at(&f, 4), Some((Table::Buffer, 2, true)));
        assert_eq!(at(&f, 5), None);
        assert_eq!(at(&f, 6), Some((Table::Buffer, 3, false)));
        assert_eq!(f.set_aside, Some(4));
        assert_eq!(f.buffers(), 5);
    }

    /// A fragment shader that sees no storage buffer gets no slot set aside; a vertex shader
    /// always does.
    #[test]
    fn only_a_stage_with_storage_buffers_or_a_vertex_stage_sets_a_slot_aside() {
        let p = render(vec![binding(1, BindingKind::Texture, BindingStage::Fragment)]);
        let l = layout(&p);
        assert_eq!(slots(&l, ShaderStage::Fragment).set_aside, None);
        assert_eq!(slots(&l, ShaderStage::Vertex).set_aside, Some(0));
    }
}
