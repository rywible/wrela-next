/-
Sheet certificates, part 6: the sandwich. A certificate on the mesh, not in the cell.

The slab certificate (parts 1 and 5) holds the SURFACE in a cell to a slab and leaves the mesh's
place in it open (part 4). This one starts from the mesh: push a patch of it out along its normal
by `δ` and in by `δ`, and prove the field's sign on each copy (outside on the outer, inside on the
inner). Each proof is the field's range over a domain of two coordinates, the patch's own.

Results:

* `sandwich_zero`: between the two copies, along the normal through every point of the patch,
  the field has a zero: every point of the mesh is within `δ` of the surface.
* `sandwich_no_hole`, `sandwich_no_extra`: in the model of part 3, with the mesh's patch as the
  plane, a sandwich of `δ` less than the side rays' reach leaves no hole and no extra. The budget
  is `δ` to each side: the slab's rule spends the same reach on its whole width.

What the sign on the two copies does not say: that the field has no matter beyond the outer copy
(a part of the thing the mesh has no triangle for). `sandwich_no_hole` takes it as `houter`; in
the engine it is the thin cells' proof.
-/
module

public import Theory.Sheet.Hole
public import Mathlib.Topology.Order.IntermediateValue
public import Mathlib.Topology.Algebra.Module.Basic

@[expose] public section

namespace Theory.Sheet

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **Every point of the mesh is within `δ` of the surface.** A continuous field positive at
`m + δ n` and negative at `m - δ n` has a zero between them on that line. -/
theorem sandwich_zero {f : E → ℝ} (hf : Continuous f) {m n : E} {δ : ℝ} (hδ : 0 ≤ δ)
    (hout : 0 < f (m + δ • n)) (hin : f (m - δ • n) < 0) :
    ∃ t ∈ Set.Icc (-δ) δ, f (m + t • n) = 0 := by
  have hg : ContinuousOn (fun t : ℝ => f (m + t • n)) (Set.Icc (-δ) δ) :=
    (hf.comp (continuous_const.add (continuous_id.smul continuous_const))).continuousOn
  have h := intermediate_value_Icc (by linarith : -δ ≤ δ) hg
  have h0 : (0 : ℝ) ∈ Set.Icc (f (m + (-δ) • n)) (f (m + δ • n)) := by
    refine ⟨?_, hout.le⟩
    have e : m + (-δ) • n = m - δ • n := by rw [neg_smul, sub_eq_add_neg]
    rw [e]
    exact hin.le
  obtain ⟨t, ht, hzero⟩ := h h0
  exact ⟨t, ht, hzero⟩

/-- **No hole in a sandwich.** With the mesh's patch as the plane (all under it the mesh's
matter) and no matter of the field beyond the outer copy, `δ` out: a sandwich of `δ` less than
the side rays' reach covers the middle ray's point. -/
theorem sandwich_no_hole {o n : E} {δ reach : ℝ} {C F M : Set E}
    (houter : ∀ q ∈ C, q ∈ F → acrossE o n q ≤ δ)
    (hmesh : ∀ q ∈ C, acrossE o n q < 0 → q ∈ M)
    (hδ : δ < reach)
    {p p' : E} (hp : p ∈ C) (hp' : p' ∈ C) (hp'F : p' ∈ F)
    (hside : reach ≤ ⟪n, p' - p⟫) : p ∈ M :=
  covered houter hmesh (by linarith) hp hp' hp'F hside

/-- **No extra in a sandwich.** The mirror: the mesh has no matter above its patch, and all
under the inner copy, `δ` in, is the field's matter. -/
theorem sandwich_no_extra {o n : E} {δ reach : ℝ} {C F M : Set E}
    (hmesh : ∀ q ∈ C, q ∈ M → acrossE o n q ≤ 0)
    (hinner : ∀ q ∈ C, acrossE o n q < -δ → q ∈ F)
    (hδ : δ < reach)
    {p p' : E} (hp : p ∈ C) (hp' : p' ∈ C) (hp'M : p' ∈ M)
    (hside : reach ≤ ⟪n, p' - p⟫) : p ∈ F :=
  covered (F := M) (M := F) hmesh hinner (by linarith) hp hp' hp'M hside

end Theory.Sheet
