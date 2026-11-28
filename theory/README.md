# The theory workshop

A place to work out the engine's mathematics in Lean 4, before it is engineered in wrela.

This folder is beside the three layers, not one of them. Nothing in `toolchain/`, `lib/`, `game/`,
the build or `tools/gate` reads it. A theorem here gives the engine no guarantee: its statement is
over the real numbers, and a person or an agent carries it into wrela by hand. The engine's own
judges, tests and audits still hold what is built.

## How to use it

1. **State the engine's claim as it is written in the code.** Name the file and the function.
2. **Prove it, or find why it does not hold.** A proof that fails names a missing hypothesis or a
   wrong constant. Write the counterexample down as a theorem.
3. **Prove rule schemas, not fields of the game.** Assume what a rule needs ("`f` is within `e` of
   a plane over the box") and prove what follows. Do not model a kind of thing.
4. **Record the result below**: the theorem, the engine code it speaks of, and what to carry
   over. When a result is carried into wrela, name the commit, and build the wrela test that
   attacks it.

What Lean does not settle here: rounding, the lattice, the order of GPU work. Those stay with
`judge::bits`, the hole audit and the gate.

## How to build

```
cd theory
lake exe cache get    # once: fetches Mathlib's compiled files (7.3 GB in .lake/)
lake build
```

- Lean 4.36.0 and Mathlib `v4.36.0` (`lean-toolchain`, `lake-manifest.json`).
- Never let Mathlib compile from source on a 16 GB machine: if `lake build` begins to build
  `Mathlib.*` files, stop it and run `lake exe cache get`.
- Do not run `lake build` beside a cargo build.
- Import the Mathlib files a proof needs and `Mathlib.Tactic`, not all of `Mathlib`.
- A file is done when it builds with no warning and no `sorry`.
- `checks/` holds numeric checks of claims that are not yet proven. Each says so.

## Results

### 1. What a sheet certificate implies (`Theory/Sheet/`)

The claim, from `lib/engine/matter/certify.wrela` and `lib/engine/mesh/outline.wrela`: the
surface in a cell lies in a slab no wider than `width` parallel to the plane of the cell's
vertex, so the mesh and the surface stand no further apart than that, so with `width` at
`HOLE_PX` pixels the mesh misses no pixel that the hole audit would count.

| Theorem | It says | State |
| --- | --- | --- |
| `Slab.slab_upper`, `slab_lower` | Where `c + l·u + η = 0`, `\|η\| ≤ e`, in the box: `wl·u` is within `-λc ± (\|λ\|e + ‖wl - λl‖₁)`, for every multiplier `λ` | Proven |
| `Slab.engine_slab` | `certify_sheet`'s slab (`w_ctr`, `half_w`) is the case `λ = 1/α`, and holds the surface for any `α ≠ 0` | Proven |
| `Slab.two_slope_width` | The upper and the lower side may each take its own multiplier | Proven |
| `Slab.Tilted.*` | An enclosure where the engine's `α` gives a half width of 1.05 and the best multiplier 0.6, which the surface reaches | Proven |
| `Slab.across_shift` | A slab's width does not depend on the plane's point: the certificate never reads where the vertex is | Proven |
| `Hull.gap_le_width`, `gap_le_joint` | If one slab holds the surface and the mesh's corners, mesh and surface are no further apart across it than it is wide; with the corners in a second slab, no further than the two slabs' joint extent | Proven |
| `Chord.Roof.certified_yet_apart` | A ridge on a cell's face: every cell certified at width zero, every vertex on the surface, the mesh `m·a` off it | Proven |
| `Chord.Bowl.cell_slab`, `edge_gap` | A surface of constant bend: the mesh's edge across two cells stands four times as far off as either cell's slab is high | Proven |
| `Hole.no_hole`, `covered` | A slab narrower than the side rays' reach across it leaves no hole | Proven |
| `Hole.hole_fits` | That is tight: any slab as wide as the reach has a field and a mesh with a hole | Proven |
| `Hole.side_reach`, `side_reach_ge`, `side_reach_diagonal` | The audit's four side rays reach `max \|x\| \|y\|` of half a pixel across an outline whose normal in the picture is `(x, y)`: `√2/2` of it across a diagonal one | Proven |
| `checks/slab_lp.py` | The best multiplier is one of four (`0` and `wl[i] / l[i]`), and the slab it gives is the true extent | Numbers only: 20,000 enclosures, largest difference 4.7e-15 |

#### Findings

**A. The certificate's slope is sound and not the tightest.** `certify_sheet` projects the
enclosure's slopes onto the plane's normal (`α = dot(l / h_in, n)`), which is the least-squares
slope. The bound it then uses is a sum of absolute values, and the least-squares slope does not
minimise it. The best bound takes one multiplier for each side of the slab, each one of four
candidates. On synthetic cells (`checks/slab_lp.py`: a unit-slope field, the normal tilted from
the enclosure's gradient, no bisection) the engine's slab is 1.07 to 1.17 times the best at the
median and 1.26 to 1.61 times at the 90th percentile for tilts of 2° to 10°. At a tilt of 5° and `e = 0.02` of a
half cell, 1,681 of 4,000 cells fit a slab 0.15 of a cell wide by the engine's slope and 3,711 by
the best. Those are not the game's cells: the measure that counts is `certify_report` on the
kinds of the world.

**B. The certificate holds the surface, and nothing holds the mesh to it.** `certify_sheet`
shows that the surface in one cell is flat across a direction. The step "so the mesh is no
further from it than the slab is wide" needs every corner of the triangles that cross the cell
to lie in the same slab (`Hull.gap_le_width`). The code checks none of them: not the cell's own
vertex (the result does not depend on `at`, and a vertex may stand up to 0.85 of a cell outside
its cell), and not the vertices of the cells beside it. `Chord.Roof` and `Chord.Bowl` are meshes
whose every cell is certified and which stand off their surface. For a surface of constant bend
the mesh's error is up to four times the certified width. `Hull.gap_le_joint` is the repaired
statement: hold the corners' coordinates across the slab's plane too (a dot product a corner),
and the gap is at most the joint extent.

**C. Half a pixel is too wide for an outline that is not along an axis.** A hole needs the field
to cover the four rays half a pixel out along the picture's axes. Across an outline whose normal
in the picture is `(x, y)`, the best of those reaches `max |x| |y|` of half a pixel. `no_hole` and
`hole_fits` show the slab must be narrower than exactly that reach. For a diagonal outline it is
`√2 / 4` of a pixel (0.354). The engine holds a certificate while its width is at most `HOLE_PX`
(0.5) of a pixel, and makes one at 0.375 (`WATCH_SLACK`). The engine's history agrees with the
direction: at twice `HOLE_PX` the audit found a hole. Where the slab's normal leans toward the
eye, `(x, y)` is shorter than a unit vector and the reach shrinks with it.

The model's two hypotheses (`hcert`, `hmesh`) are one-sided: the field has no matter above the
slab, and all below the slab is the mesh's matter. They follow from "the surface and the mesh
lie in the slab" only with the mesh closed and facing out there. That step is not formalised.

#### To carry into wrela (none is carried yet)

1. **The best multipliers in `certify_sheet`**, in place of `α = dot(g, n)`: evaluate the upper
   bound and the lower bound at `λ ∈ {0, w_l.x / l.x, w_l.y / l.y, w_l.z / l.z}` and keep the
   least and the greatest. Measure with `certify_report` and the candlestick's census before and
   after. Prove first, or keep as numbers: that a breakpoint is optimal.
2. **The mesh's corners in the slab.** Where `outline_quads` puts a quad's cells up, hold each
   cell's slab to the quad's four vertices as well. First build the wrela test that attacks it: a
   sphere a few cells in radius and a ridge on a lattice plane, at many placements, counted by
   the hole audit.
3. **The need across the outline's own direction**: `HOLE_PX * px * max(|n·x̂|, |n·ŷ|)` with the
   picture's axes, in `cert_due` and `certify_watched`. First add diagonal outlines to the audit's
   eyes if it has none that graze so.

### Next

- The thin cells' cores (`watched_core`: a cell less a quarter pixel on every side) by the same
  model.
- Second-order enclosures: the form, how it composes through `smin` and the maps, the remainder.
- Motion by structure: when a displaced surface is a graph over its rest surface.
- Steepness rules for std's and the kit's operators.
- Ring balance: rings by distance alone, snapped to parent bricks, give a 2:1 cut.
