/-
Sheet certificates, part 5: the slab across the enclosure's own gradient.

`certify_sheet` asks for a slab parallel to the plane of the cell's vertex. Nothing in parts 2
and 3 needs that direction: `Hull.gap_le_width` wants one slab that holds the surface and the
mesh's corners, and `Hole.no_hole` measures the side rays' reach across whatever normal the slab
has. This part takes the slab the enclosure gives of itself, across its own slopes `l`.

Results, in the box's coordinates, for a field `c + l·u + η` with `|η| ≤ e`:

* `zero_in_own_slab`: every zero of the field in the box has `|c + l·u| ≤ e`. Across `l` the
  surface lies in a slab of half width `e`: no tilt term, whatever the vertex's normal is.
* `own_slab_engine`: the same through `Slab.engine_slab`: with the plane's coordinate along
  `l / α` the residual is zero and the half width is `e / |α|`.
* `mesh_in_own_slab`: a convex combination of zeros of the field in the box lies in the same
  slab. So a triangle whose corners are on the surface and in the box is held with the surface,
  with no further check: this is the hypothesis `Hull.gap_le_width` wants, for the corners the
  box holds.
* `field_on_mesh`: and the field at any point of such a triangle is at most `2 e`.
-/
module

public import Theory.Sheet.Slab

@[expose] public section

namespace Theory.Sheet

open Finset

variable {ι : Type*} [Fintype ι]

/-- **The surface lies in the enclosure's own slab.** -/
theorem zero_in_own_slab (c e η : ℝ) (l u : ι → ℝ) (hη : |η| ≤ e)
    (hzero : c + dot l u + η = 0) : |c + dot l u| ≤ e := by
  have : c + dot l u = -η := by linarith
  rw [this, abs_neg]
  exact hη

/-- The same slab as `certify_sheet` would find it, were its plane's normal the enclosure's
gradient: the coordinate `(l / α)·u`, no residual, a half width of `e / |α|`. -/
theorem own_slab_engine (c e η : ℝ) (l u : ι → ℝ) (hu : ∀ i, |u i| ≤ 1) (hη : |η| ≤ e)
    (hzero : c + dot l u + η = 0) {α : ℝ} (hα : α ≠ 0) :
    |dot (fun i => l i / α) u - (-c / α)| ≤ e / |α| := by
  have h := engine_slab c e η l (fun i => l i / α) u hu hη hzero hα
  have hr : norm1 (residual l (fun i => l i / α) α) = 0 := by
    unfold norm1 residual
    refine Finset.sum_eq_zero fun i _ => ?_
    have : l i - α * (l i / α) = 0 := by
      field_simp
      ring
    rw [this, abs_zero]
  rw [hr, zero_add] at h
  exact h

/-- `l·` of a weighted sum of points is the weighted sum of `l·` of each. -/
theorem dot_combo {κ : Type*} [Fintype κ] (l : ι → ℝ) (t : κ → ℝ) (u : κ → ι → ℝ) :
    dot l (fun i => ∑ k, t k * u k i) = ∑ k, t k * dot l (u k) := by
  unfold dot
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun i _ => by ring

/-- **The mesh lies in it too.** A convex combination of points of the enclosure's own slab is
in that slab: so is every point of a triangle whose corners are zeros of the field in the box. -/
theorem mesh_in_own_slab {κ : Type*} [Fintype κ] (c e : ℝ) (l : ι → ℝ) (t : κ → ℝ)
    (u : κ → ι → ℝ) (ht : ∀ k, 0 ≤ t k) (hsum : ∑ k, t k = 1)
    (hcorner : ∀ k, |c + dot l (u k)| ≤ e) :
    |c + dot l (fun i => ∑ k, t k * u k i)| ≤ e := by
  have hc : c = ∑ k, t k * c := by rw [← Finset.sum_mul, hsum, one_mul]
  have h1 : c + dot l (fun i => ∑ k, t k * u k i) = ∑ k, t k * (c + dot l (u k)) := by
    rw [dot_combo]
    nth_rewrite 1 [hc]
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun k _ => by ring
  rw [h1]
  calc |∑ k, t k * (c + dot l (u k))| ≤ ∑ k, |t k * (c + dot l (u k))| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ k, t k * e := Finset.sum_le_sum fun k _ => by
        rw [abs_mul, abs_of_nonneg (ht k)]
        exact mul_le_mul_of_nonneg_left (hcorner k) (ht k)
    _ = e := by rw [← Finset.sum_mul, hsum, one_mul]

/-- **The field on the mesh.** At a point `m` of the box that is a convex combination of zeros of
the field in the box, the field is at most `2 e`. -/
theorem field_on_mesh {κ : Type*} [Fintype κ] (c e ηm : ℝ) (l : ι → ℝ) (t : κ → ℝ)
    (u : κ → ι → ℝ) (ht : ∀ k, 0 ≤ t k) (hsum : ∑ k, t k = 1)
    (hcorner : ∀ k, |c + dot l (u k)| ≤ e) (hηm : |ηm| ≤ e) :
    |c + dot l (fun i => ∑ k, t k * u k i) + ηm| ≤ 2 * e := by
  have h := mesh_in_own_slab c e l t u ht hsum hcorner
  calc |c + dot l (fun i => ∑ k, t k * u k i) + ηm|
      ≤ |c + dot l (fun i => ∑ k, t k * u k i)| + |ηm| := abs_add_le _ _
    _ ≤ 2 * e := by linarith

end Theory.Sheet
