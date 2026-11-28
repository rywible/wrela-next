/-
Sheet certificates, part 3: how wide a slab may be before a hole fits in it.

Engine: `lib/engine/judge/audit.wrela` (a HOLE: the field meets matter along a pixel's middle ray
and along four rays half a pixel to each side, and the engine met nothing),
`lib/engine/mesh/dc.wrela` (`HOLE_PX`), `lib/engine/mesh/outline.wrela` (`cert_due`,
`certify_watched`: a certificate's slab is held to `HOLE_PX` pixels).

The model. A region `C` (the cells in question), the field's matter `F`, the mesh's matter `M`,
and a plane (`o`, `n`) with a slab `[a, b]` across it such that, in `C`:

* `hcert`: the field has no matter above the slab (its surface there is at or below `b`),
* `hmesh`: everything below the slab is the mesh's matter (its surface there is at or above `a`).

`p` is a point of the pixel's middle ray, `p'` a point where a ray beside it meets the field, and
`δ` how much further out across the slab `p'` stands than `p`.

Results:

* `no_hole`: if `p` is not the mesh's matter then `δ ≤ b - a`. So a slab narrower than `δ`
  leaves no hole: `covered`.
* `hole_fits`: this is tight. For every slab with `δ ≤ b - a` there is a field and a mesh that
  meet both hypotheses and have a hole. A slab of exactly the side rays' reach is too wide, and
  one of twice it more so.
* `side_reach`, `side_reach_diagonal`: the audit's four side rays stand half a pixel out along
  the picture's axes. Across an outline whose normal in the picture is `(x, y)` (a unit vector)
  the best of them reaches `max |x| |y|` of that: all of it for an outline along an axis, and
  `√2 / 2` of it for a diagonal one, never less. So the slab that leaves no hole for any outline
  is narrower than `√2 / 4` of a pixel (0.354), not half of one.
-/
module

public import Theory.Sheet.Hull
public import Mathlib.Analysis.SpecialFunctions.Sqrt

@[expose] public section

namespace Theory.Sheet

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **No hole in a narrow slab.** If the middle ray's point `p` is not the mesh's matter while
the field has matter at `p'`, at least `δ` further out across the slab, then the slab is at least
`δ` wide. -/
theorem no_hole {o n : E} {a b δ : ℝ} {C F M : Set E}
    (hcert : ∀ q ∈ C, q ∈ F → acrossE o n q ≤ b)
    (hmesh : ∀ q ∈ C, acrossE o n q < a → q ∈ M)
    {p p' : E} (hp : p ∈ C) (hpM : p ∉ M) (hp' : p' ∈ C) (hp'F : p' ∈ F)
    (hside : δ ≤ ⟪n, p' - p⟫) : δ ≤ b - a := by
  have h1 : acrossE o n p' ≤ b := hcert p' hp' hp'F
  have h2 : a ≤ acrossE o n p := by
    by_contra h
    exact hpM (hmesh p hp (not_le.mp h))
  have h3 : acrossE o n p' = acrossE o n p + ⟪n, p' - p⟫ := by
    rw [← acrossE_add]
    congr 1
    abel
  linarith

/-- A slab narrower than the side rays' reach covers the middle ray's point. -/
theorem covered {o n : E} {a b δ : ℝ} {C F M : Set E}
    (hcert : ∀ q ∈ C, q ∈ F → acrossE o n q ≤ b)
    (hmesh : ∀ q ∈ C, acrossE o n q < a → q ∈ M)
    (hwidth : b - a < δ)
    {p p' : E} (hp : p ∈ C) (hp' : p' ∈ C) (hp'F : p' ∈ F)
    (hside : δ ≤ ⟪n, p' - p⟫) : p ∈ M := by
  by_contra hpM
  exact absurd (no_hole hcert hmesh hp hpM hp' hp'F hside) (not_le.mpr hwidth)

/-- **The bound is tight.** For any slab at least as wide as the side rays' reach there is a field
and a mesh, each within the slab, with a hole: the field fills to the slab's top, the mesh to its
bottom, and the middle ray's point stands on the bottom. -/
theorem hole_fits {o n : E} (hn : ‖n‖ = 1) {a b δ : ℝ} (hδ : δ ≤ b - a) :
    ∃ (C F M : Set E) (p p' : E),
      (∀ q ∈ C, q ∈ F → acrossE o n q ≤ b) ∧
      (∀ q ∈ C, acrossE o n q < a → q ∈ M) ∧
      p ∈ C ∧ p ∉ M ∧ p' ∈ C ∧ p' ∈ F ∧ δ ≤ ⟪n, p' - p⟫ := by
  have hnn : ⟪n, n⟫ = 1 := by rw [real_inner_self_eq_norm_sq, hn]; norm_num
  have hat : ∀ t : ℝ, acrossE o n (o + t • n) = t := by
    intro t
    unfold acrossE
    rw [add_sub_cancel_left, inner_smul_right, hnn, mul_one]
  refine ⟨Set.univ, {q | acrossE o n q ≤ b}, {q | acrossE o n q < a},
    o + a • n, o + (a + δ) • n, ?_, ?_, Set.mem_univ _, ?_, Set.mem_univ _, ?_, ?_⟩
  · intro q _ hq
    exact hq
  · intro q _ hq
    exact hq
  · simp only [Set.mem_ofPred_eq, hat, lt_irrefl, not_false_eq_true]
  · simp only [Set.mem_ofPred_eq, hat]
    linarith
  · have : o + (a + δ) • n - (o + a • n) = δ • n := by
      rw [add_smul]
      abel
    rw [this, inner_smul_right, hnn, mul_one]

/-- **A mesh in the middle of the slab may have a slab twice as wide.** `no_hole` reads the
slab's bottom only as the level the mesh's matter reaches. If the mesh fills to the slab's middle
(within `ε` of it) while the field lies in `[a, b]`, a hole needs `δ ≤ (b - a) / 2 + ε`: the slab
may be nearly `2 δ` wide. A mesh whose vertices are on the surface can lie anywhere in the slab;
one whose vertices are at the slab's middle splits the error between the two sides. -/
theorem no_hole_centred {o n : E} {a b δ ε : ℝ} {C F M : Set E}
    (hcert : ∀ q ∈ C, q ∈ F → acrossE o n q ≤ b)
    (hmesh : ∀ q ∈ C, acrossE o n q < (a + b) / 2 - ε → q ∈ M)
    {p p' : E} (hp : p ∈ C) (hpM : p ∉ M) (hp' : p' ∈ C) (hp'F : p' ∈ F)
    (hside : δ ≤ ⟪n, p' - p⟫) : δ ≤ (b - a) / 2 + ε := by
  have h := no_hole hcert hmesh hp hpM hp' hp'F hside
  linarith

/-- The same for an extra (the mesh's matter where the field has none), with the two exchanged:
the mesh has no matter above the middle plus `ε`, all below the slab is the field's matter. -/
theorem no_extra_centred {o n : E} {a b δ ε : ℝ} {C F M : Set E}
    (hmesh : ∀ q ∈ C, q ∈ M → acrossE o n q ≤ (a + b) / 2 + ε)
    (hfield : ∀ q ∈ C, acrossE o n q < a → q ∈ F)
    {p p' : E} (hp : p ∈ C) (hpF : p ∉ F) (hp' : p' ∈ C) (hp'M : p' ∈ M)
    (hside : δ ≤ ⟪n, p' - p⟫) : δ ≤ (b - a) / 2 + ε := by
  have h := no_hole (F := M) (M := F) hmesh hfield hp hpF hp' hp'M hside
  linarith

/-! ### The audit's side rays

Four rays, half a pixel out along the picture's axes. In the picture's plane, with `d` that half
pixel's width where the rays meet the field, their offsets from the middle ray are `(±d, 0)` and
`(0, ±d)`, and an outline's normal is a unit vector `(x, y)`. -/

/-- The best of the four side rays reaches `d * max |x| |y|` across the outline. -/
theorem side_reach (d x y : ℝ) :
    ∃ s : ℝ × ℝ, (s = (d, 0) ∨ s = (-d, 0) ∨ s = (0, d) ∨ s = (0, -d)) ∧
      s.1 * x + s.2 * y = d * max |x| |y| := by
  rcases le_total |y| |x| with h | h
  · rw [max_eq_left h]
    rcases le_total 0 x with hx | hx
    · exact ⟨(d, 0), Or.inl rfl, by simp [abs_of_nonneg hx]⟩
    · exact ⟨(-d, 0), Or.inr (Or.inl rfl), by simp [abs_of_nonpos hx]⟩
  · rw [max_eq_right h]
    rcases le_total 0 y with hy | hy
    · exact ⟨(0, d), Or.inr (Or.inr (Or.inl rfl)), by simp [abs_of_nonneg hy]⟩
    · exact ⟨(0, -d), Or.inr (Or.inr (Or.inr rfl)), by simp [abs_of_nonpos hy]⟩

/-- Across any outline the best side ray reaches at least `√2 / 2` of the half pixel. -/
theorem side_reach_ge (x y : ℝ) (hxy : x ^ 2 + y ^ 2 = 1) : Real.sqrt 2 / 2 ≤ max |x| |y| := by
  by_contra h
  rw [not_le, max_lt_iff] at h
  have hs : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hs0 : 0 ≤ Real.sqrt 2 := Real.sqrt_nonneg 2
  have hx : x ^ 2 < 1 / 2 := by
    have := abs_nonneg x
    rw [← sq_abs]
    nlinarith [h.1]
  have hy : y ^ 2 < 1 / 2 := by
    have := abs_nonneg y
    rw [← sq_abs]
    nlinarith [h.2]
  linarith

/-- And a diagonal outline gives exactly that: no side ray reaches further across it. -/
theorem side_reach_diagonal :
    (Real.sqrt 2 / 2) ^ 2 + (Real.sqrt 2 / 2) ^ 2 = 1 ∧
      max |Real.sqrt 2 / 2| |Real.sqrt 2 / 2| = Real.sqrt 2 / 2 := by
  have hs : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hs0 : 0 ≤ Real.sqrt 2 := Real.sqrt_nonneg 2
  refine ⟨by nlinarith, ?_⟩
  rw [max_self, abs_of_nonneg (by positivity)]

end Theory.Sheet
