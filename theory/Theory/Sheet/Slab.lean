/-
Sheet certificates, part 1: from an affine enclosure of a field over a box to a slab that holds
the field's surface there.

Engine: `lib/engine/matter/certify.wrela`, `certify_sheet`.

The setting, in the box's own coordinates `u ∈ [-1, 1]^ι` (the engine's `ι` is three axes):

* the enclosure: `f = c + l·u + η` with `|η| ≤ e` (`af.c`, `af.l`, `af.e`),
* a coordinate across a plane: `w(p) - w_mid = wl·u` (`w_l = h_in * n`, see `across_box`),
* the surface: the points where `f = 0`.

Results:

* `slab_upper`, `slab_lower`: for EVERY multiplier `lam`, a bound on `wl·u` over the surface.
* `engine_slab`: the engine's slab is the instance `lam = 1 / α`, for ANY `α ≠ 0`. So the
  engine's choice `α = dot(l / h_in, n)` matters to how tight the slab is, never to soundness.
* `two_slope_width`: the upper and the lower bound may take different multipliers.
* `Tilted`: an enclosure where the engine's `α` gives a half width of 1.05 and the best
  multiplier gives 0.6, which the surface attains. The engine's `α` is the least-squares slope;
  the bound it enters is an L1 one, and the least-squares slope does not minimise it.
-/
module

public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Algebra.Order.Field.Basic
public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Fin.VecNotation
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Tactic

@[expose] public section

namespace Theory.Sheet

open Finset

variable {ι : Type*} [Fintype ι]

/-- A row of coefficients applied to a point of the box: `a·u`. -/
def dot (a u : ι → ℝ) : ℝ := ∑ i, a i * u i

/-- The most `a·u` can be over the box `[-1, 1]^ι`: the sum of `|a i|`. -/
def norm1 (a : ι → ℝ) : ℝ := ∑ i, |a i|

theorem norm1_nonneg (a : ι → ℝ) : 0 ≤ norm1 a :=
  Finset.sum_nonneg fun i _ => abs_nonneg (a i)

/-- Over the box, `|a·u| ≤ ‖a‖₁`. -/
theorem abs_dot_le (a u : ι → ℝ) (hu : ∀ i, |u i| ≤ 1) : |dot a u| ≤ norm1 a := by
  unfold dot norm1
  calc |∑ i, a i * u i| ≤ ∑ i, |a i * u i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, |a i| := Finset.sum_le_sum fun i _ => by
        rw [abs_mul]
        exact mul_le_of_le_one_right (abs_nonneg _) (hu i)

/-- `wl·u` split by a multiplier `lam` of the enclosure's linear part. -/
theorem dot_split (l wl u : ι → ℝ) (lam : ℝ) :
    dot wl u = lam * dot l u + dot (fun i => wl i - lam * l i) u := by
  unfold dot
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- **The slab, from above.** Where the field is zero in the box, `wl·u` is at most
`-lam c + |lam| e + ‖wl - lam l‖₁`, for every multiplier `lam`. (`lam = 0` is the box's own
span, `‖wl‖₁`.) -/
theorem slab_upper (c e η : ℝ) (l wl u : ι → ℝ) (hu : ∀ i, |u i| ≤ 1) (hη : |η| ≤ e)
    (hzero : c + dot l u + η = 0) (lam : ℝ) :
    dot wl u ≤ -lam * c + |lam| * e + norm1 (fun i => wl i - lam * l i) := by
  have h1 := abs_le.mp (abs_dot_le (fun i => wl i - lam * l i) u hu)
  have h2 : dot l u = -c - η := by linarith
  have h3 : |lam * η| ≤ |lam| * e := by
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left hη (abs_nonneg _)
  have h4 := abs_le.mp h3
  rw [dot_split l wl u lam, h2]
  nlinarith [h1.2, h4.1]

/-- **The slab, from below.** -/
theorem slab_lower (c e η : ℝ) (l wl u : ι → ℝ) (hu : ∀ i, |u i| ≤ 1) (hη : |η| ≤ e)
    (hzero : c + dot l u + η = 0) (lam : ℝ) :
    -lam * c - |lam| * e - norm1 (fun i => wl i - lam * l i) ≤ dot wl u := by
  have h1 := abs_le.mp (abs_dot_le (fun i => wl i - lam * l i) u hu)
  have h2 : dot l u = -c - η := by linarith
  have h3 : |lam * η| ≤ |lam| * e := by
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left hη (abs_nonneg _)
  have h4 := abs_le.mp h3
  rw [dot_split l wl u lam, h2]
  nlinarith [h1.1, h4.2]

/-- The residual of the engine's split: `r = l - α wl`. -/
def residual (l wl : ι → ℝ) (α : ℝ) : ι → ℝ := fun i => l i - α * wl i

theorem norm1_engine (l wl : ι → ℝ) {α : ℝ} (hα : α ≠ 0) :
    norm1 (fun i => wl i - (1 / α) * l i) = norm1 (residual l wl α) / |α| := by
  unfold norm1 residual
  rw [Finset.sum_div]
  refine Finset.sum_congr rfl fun i _ => ?_
  have : wl i - (1 / α) * l i = -(l i - α * wl i) / α := by
    field_simp
    ring
  show |wl i - (1 / α) * l i| = |l i - α * wl i| / |α|
  rw [this, abs_div, abs_neg]

/-- **The engine's slab is sound for any slope.** `certify_sheet` takes
`w_ctr = w_mid - c / α` and `half_w = (‖r‖₁ + e) / |α|` with `r = l - α w_l`. That holds the
surface for every `α ≠ 0`: nothing rests on `α` being `dot(l / h_in, n)`. -/
theorem engine_slab (c e η : ℝ) (l wl u : ι → ℝ) (hu : ∀ i, |u i| ≤ 1) (hη : |η| ≤ e)
    (hzero : c + dot l u + η = 0) {α : ℝ} (hα : α ≠ 0) :
    |dot wl u - (-c / α)| ≤ (norm1 (residual l wl α) + e) / |α| := by
  have hup := slab_upper c e η l wl u hu hη hzero (1 / α)
  have hlo := slab_lower c e η l wl u hu hη hzero (1 / α)
  rw [norm1_engine l wl hα] at hup hlo
  have habs : |1 / α| = 1 / |α| := by rw [abs_div, abs_one]
  rw [habs] at hup hlo
  have e1 : -(1 / α) * c = -c / α := by ring
  have e2 : (norm1 (residual l wl α) + e) / |α| =
      1 / |α| * e + norm1 (residual l wl α) / |α| := by ring
  rw [e1] at hup hlo
  rw [e2, abs_le]
  constructor <;> linarith

/-- **Two slopes.** The upper bound and the lower bound may each take its own multiplier, so the
surface's extent across the plane is at most the gap between the best of each. -/
theorem two_slope_width (c e η η' : ℝ) (l wl u u' : ι → ℝ)
    (hu : ∀ i, |u i| ≤ 1) (hu' : ∀ i, |u' i| ≤ 1) (hη : |η| ≤ e) (hη' : |η'| ≤ e)
    (hzero : c + dot l u + η = 0) (hzero' : c + dot l u' + η' = 0) (lam mu : ℝ) :
    dot wl u - dot wl u' ≤
      (-lam * c + |lam| * e + norm1 (fun i => wl i - lam * l i)) -
      (-mu * c - |mu| * e - norm1 (fun i => wl i - mu * l i)) := by
  have h1 := slab_upper c e η l wl u hu hη hzero lam
  have h2 := slab_lower c e η' l wl u' hu' hη' hzero' mu
  linarith

/-! ### The plane's coordinate over a box

`certify_sheet` measures `w(p) = dot(p - at, n)` and writes the box as `p = c_b + h_in * u`.
Then `w(p) = w_mid + (h_in * n)·u`: the `wl` above is `h_in * n`, axis by axis. -/

/-- A point's coordinate across the plane through `o` with normal `n`. -/
def across (o n p : ι → ℝ) : ℝ := ∑ i, (p i - o i) * n i

theorem across_box (o n cb h u : ι → ℝ) :
    across o n (fun i => cb i + h i * u i) = across o n cb + dot (fun i => h i * n i) u := by
  unfold across dot
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- **The certificate does not read where the plane stands.** Moving the plane's point from `o`
to `o'` moves every coordinate by one constant, so the width of a slab is the same: `certify_sheet`
proves the surface flat across `n`, and says nothing of how far the plane's own point (the mesh's
vertex) is from that slab. -/
theorem across_shift (o o' n p q : ι → ℝ) :
    across o n p - across o n q = across o' n p - across o' n q := by
  unfold across
  rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

/-! ### The engine's slope is not the tightest

A unit box, the field `f = u₀` (its surface the plane `u₀ = 0`), and a plane tilted from it:
`n = (0.8, 0.6, 0)`. -/

namespace Tilted

def l : Fin 3 → ℝ := ![1, 0, 0]
def wl : Fin 3 → ℝ := ![0.8, 0.6, 0]

/-- The engine's slope: `α = dot(l / h_in, n)` with `h_in = 1`. -/
theorem engine_alpha : dot l wl = 0.8 := by
  simp [dot, l, wl, Fin.sum_univ_three]

/-- The engine's half width, `(‖r‖₁ + e) / |α|` with `e = 0`: 1.05 (the box's span is 1.4). -/
theorem engine_half_width : (norm1 (residual l wl 0.8) + 0) / |(0.8 : ℝ)| = 1.05 := by
  have h1 : |(1 : ℝ) - 0.8 * 0.8| = 0.36 := by rw [abs_of_nonneg] <;> norm_num
  have h2 : |(0 : ℝ) - 0.8 * 0.6| = 0.48 := by
    rw [show (0 : ℝ) - 0.8 * 0.6 = -0.48 by norm_num, abs_neg, abs_of_nonneg]; norm_num
  have h3 : |(0.8 : ℝ)| = 0.8 := abs_of_nonneg (by norm_num)
  simp only [norm1, residual, l, wl, Fin.sum_univ_three, Matrix.cons_val_zero,
    Matrix.cons_val_one, Matrix.cons_val_two, Matrix.head_cons, Matrix.tail_cons]
  simp only [h3]
  norm_num [h1, h2]

/-- The best multiplier, `lam = 0.8` (a slope of 1.25), gives a half width of 0.6. -/
theorem best_half_width :
    -(0.8 : ℝ) * 0 + |(0.8 : ℝ)| * 0 + norm1 (fun i => wl i - 0.8 * l i) = 0.6 := by
  have h3 : |(0.6 : ℝ)| = 0.6 := abs_of_nonneg (by norm_num)
  simp [norm1, l, wl, Fin.sum_univ_three, h3]

/-- And the surface reaches it: `u = (0, 1, 0)` is in the box, on the surface, 0.6 across. -/
theorem attained :
    let u : Fin 3 → ℝ := ![0, 1, 0]
    (∀ i, |u i| ≤ 1) ∧ (0 : ℝ) + dot l u + 0 = 0 ∧ dot wl u = 0.6 := by
  refine ⟨?_, ?_, ?_⟩
  · intro i
    fin_cases i <;> simp
  · simp [dot, l, Fin.sum_univ_three]
  · simp [dot, wl, Fin.sum_univ_three]

end Tilted

end Theory.Sheet
