/-
Sheet certificates, part 2: what a slab that holds the surface says of the mesh.

Engine: `lib/engine/matter/certify.wrela` ("a mesh whose vertices lie on the surface lies in any
slab that holds them, so mesh and surface stand no further apart, across the plane, than the slab
is wide"), `lib/engine/mesh/outline.wrela` (`certify_watched`).

Results:

* `convex_slab`, `hull_subset_slab`: a slab is convex, so it holds a triangle whose corners it
  holds.
* `gap_le_width`: if ONE slab holds the surface's points and the mesh's corners, a point of the
  mesh and a point of the surface are no further apart across it than it is wide.
* `gap_le_joint`: if the mesh's corners lie in a second slab about the same plane, the gap is at
  most the two slabs' joint extent. This is the hypothesis `certify_sheet` leaves open: it holds
  the surface in one cell, and no corner of the mesh (see `Theory.Sheet.Chord` for a mesh whose
  every cell is certified at a width of zero and which stands off its surface).
-/
module

public import Mathlib.Analysis.InnerProductSpace.Basic
public import Mathlib.Analysis.Convex.Hull
public import Mathlib.Tactic

@[expose] public section

namespace Theory.Sheet

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- A point's coordinate across the plane through `o` with normal `n`. -/
def acrossE (o n p : E) : ℝ := ⟪n, p - o⟫

/-- The slab about the plane through `o` with normal `n`, from `a` to `b` across it. -/
def Slab (o n : E) (a b : ℝ) : Set E := {p | a ≤ acrossE o n p ∧ acrossE o n p ≤ b}

theorem acrossE_combo (o n x y : E) {s t : ℝ} (hst : s + t = 1) :
    acrossE o n (s • x + t • y) = s * acrossE o n x + t * acrossE o n y := by
  unfold acrossE
  have h : s • x + t • y - o = s • (x - o) + t • (y - o) := by
    have ho : o = s • o + t • o := by rw [← add_smul, hst, one_smul]
    rw [smul_sub, smul_sub]
    nth_rewrite 1 [ho]
    abel
  rw [h, inner_add_right, inner_smul_right, inner_smul_right]

/-- The coordinate across a plane of a point a step from another. -/
theorem acrossE_add (o n p v : E) : acrossE o n (p + v) = acrossE o n p + ⟪n, v⟫ := by
  unfold acrossE
  rw [show p + v - o = (p - o) + v by abel, inner_add_right]

theorem convex_slab (o n : E) (a b : ℝ) : Convex ℝ (Slab o n a b) := by
  intro x hx y hy s t hs ht hst
  simp only [Slab, Set.mem_ofPred_eq] at hx hy ⊢
  rw [acrossE_combo o n x y hst]
  have ha : s * a + t * a = a := by rw [← add_mul, hst, one_mul]
  have hb : s * b + t * b = b := by rw [← add_mul, hst, one_mul]
  have h1 := mul_le_mul_of_nonneg_left hx.1 hs
  have h2 := mul_le_mul_of_nonneg_left hy.1 ht
  have h3 := mul_le_mul_of_nonneg_left hx.2 hs
  have h4 := mul_le_mul_of_nonneg_left hy.2 ht
  constructor <;> linarith

/-- A slab holds every triangle (every convex combination) of points it holds. -/
theorem hull_subset_slab {verts : Set E} {o n : E} {a b : ℝ} (h : verts ⊆ Slab o n a b) :
    convexHull ℝ verts ⊆ Slab o n a b :=
  convexHull_min h (convex_slab o n a b)

/-- **Mesh and surface in one slab.** If one slab holds the mesh's corners and a point of the
surface, then that point and any point of the mesh are no further apart across the slab than it is
wide. -/
theorem gap_le_width {verts : Set E} {o n : E} {a b : ℝ} (hv : verts ⊆ Slab o n a b)
    {m s : E} (hm : m ∈ convexHull ℝ verts) (hs : s ∈ Slab o n a b) :
    |acrossE o n m - acrossE o n s| ≤ b - a := by
  have hm' := hull_subset_slab hv hm
  simp only [Slab, Set.mem_ofPred_eq] at hm' hs
  rw [abs_le]
  constructor <;> linarith [hm'.1, hm'.2, hs.1, hs.2]

theorem slab_mono {o n : E} {a b a' b' : ℝ} (ha : a' ≤ a) (hb : b ≤ b') :
    Slab o n a b ⊆ Slab o n a' b' := by
  intro p hp
  simp only [Slab, Set.mem_ofPred_eq] at hp ⊢
  exact ⟨le_trans ha hp.1, le_trans hp.2 hb⟩

/-- **What the certificate needs beside it.** The surface in a slab `[a, b]` (the certificate) and
the mesh's corners in a slab `[a', b']` about the same plane (a few dot products: each corner's
coordinate across it): the gap is at most the two slabs' joint extent. -/
theorem gap_le_joint {verts : Set E} {o n : E} {a b a' b' : ℝ} (hv : verts ⊆ Slab o n a' b')
    {m s : E} (hm : m ∈ convexHull ℝ verts) (hs : s ∈ Slab o n a b) :
    |acrossE o n m - acrossE o n s| ≤ max b b' - min a a' :=
  gap_le_width (fun _ hq => slab_mono (min_le_right a a') (le_max_right b b') (hv hq)) hm
    (slab_mono (min_le_left a a') (le_max_left b b') hs)

end Theory.Sheet
