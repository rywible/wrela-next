/-
Sheet certificates, part 4: a certificate of every cell does not hold the mesh.

`certify_sheet` holds the SURFACE in ONE cell to a slab. A triangle of the mesh joins vertices of
different cells, and the part of it inside a cell is held by that cell's slab only if the other
cells' vertices lie in that slab too (`Theory.Sheet.Hull.gap_le_width`). Nothing checks that.

Two meshes in the plane (a section across an outline), heights measured straight up (the same
factor turns each height into a distance across a cell's plane, so the ratios stand):

* `Roof`: a ridge standing on the face between two cells. Each cell's surface is flat (a slab of
  width zero about its own plane), each vertex lies on the surface in its own cell, and the edge
  between them passes `m * a` under the ridge.
* `Bowl`: a surface of constant bend `κ`. Each cell `h` wide holds it in a slab `κ h² / 8` high;
  the edge between the vertices of two cells side by side stands `κ h² / 2` off it: four times
  what either cell's certificate says.
-/
module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Tactic

@[expose] public section

namespace Theory.Sheet

noncomputable section

/-! ### A ridge on a cell's face -/

namespace Roof

/-- The field: matter under the roof `y = -m |x|`. Cell C is `x ≤ 0`, cell D is `0 ≤ x`. -/
def f (m x y : ℝ) : ℝ := y + m * |x|

/-- Across cell C's plane (its surface's own: normal `(-m, 1)`, not made unit). -/
def wC (m x y : ℝ) : ℝ := y - m * x

/-- Across cell D's plane (normal `(m, 1)`). -/
def wD (m x y : ℝ) : ℝ := y + m * x

/-- Cell C's surface lies in a slab of width zero about its plane. -/
theorem flat_C {m x y : ℝ} (hx : x ≤ 0) (h : f m x y = 0) : wC m x y = 0 := by
  unfold f at h
  unfold wC
  rw [abs_of_nonpos hx] at h
  linarith

/-- Cell D's surface lies in a slab of width zero about its plane. -/
theorem flat_D {m x y : ℝ} (hx : 0 ≤ x) (h : f m x y = 0) : wD m x y = 0 := by
  unfold f at h
  unfold wD
  rw [abs_of_nonneg hx] at h
  linarith

/-- **Every cell certified at width zero, and the mesh `m * a` off the surface.** The vertices
`(-a, -m a)` in C and `(a, -m a)` in D lie on the surface; the middle of the edge between them,
`(0, -m a)`, lies in C; so does the surface's point `(0, 0)`; and across C's plane they are
`m * a` apart. -/
theorem certified_yet_apart (m a : ℝ) (ha : 0 ≤ a) :
    f m (-a) (-m * a) = 0 ∧ f m a (-m * a) = 0 ∧ f m 0 0 = 0 ∧
      wC m 0 0 - wC m 0 (-m * a) = m * a := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · unfold f
    rw [abs_neg, abs_of_nonneg ha]
    ring
  · unfold f
    rw [abs_of_nonneg ha]
    ring
  · simp [f]
  · unfold wC
    ring

end Roof

/-! ### A surface of constant bend -/

namespace Bowl

/-- The surface `y = -(κ / 2) x²`. Cell C is `0 ≤ x ≤ h`, cell D is `h ≤ x ≤ 2 h`. -/
def s (κ x : ℝ) : ℝ := -(κ / 2) * x ^ 2

/-- Cell C's chord, from `(0, s 0)` to `(h, s h)`. -/
def chordC (κ h x : ℝ) : ℝ := -(κ * h / 2) * x

/-- The mesh's edge from C's vertex `(0, s 0)` to D's vertex `(2 h, s (2 h))`. -/
def edge (κ h x : ℝ) : ℝ := -(κ * h) * x

theorem edge_ends (κ h : ℝ) : edge κ h 0 = s κ 0 ∧ edge κ h (2 * h) = s κ (2 * h) := by
  unfold edge s
  constructor <;> ring

/-- Cell C holds its surface in a slab `κ h² / 8` high, above its chord. -/
theorem cell_slab {κ h x : ℝ} (hκ : 0 ≤ κ) (hx0 : 0 ≤ x) (hxh : x ≤ h) :
    0 ≤ s κ x - chordC κ h x ∧ s κ x - chordC κ h x ≤ κ * h ^ 2 / 8 := by
  have e : s κ x - chordC κ h x = κ / 2 * (x * (h - x)) := by
    unfold s chordC
    ring
  rw [e]
  constructor
  · exact mul_nonneg (by linarith) (mul_nonneg hx0 (by linarith))
  · nlinarith [mul_nonneg hκ (sq_nonneg (x - h / 2))]

/-- **The mesh's edge stands four certificates off.** At the face between the cells the surface
is `κ h² / 2` above the edge: four times the height of the slab either cell is certified to. -/
theorem edge_gap (κ h : ℝ) : s κ h - edge κ h h = 4 * (κ * h ^ 2 / 8) := by
  unfold s edge
  ring

end Bowl

end

end Theory.Sheet
