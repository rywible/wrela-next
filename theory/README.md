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
| `Hole.no_hole_centred`, `no_extra_centred` | A mesh that fills to the slab's middle may have a slab twice as wide | Proven |
| `Own.zero_in_own_slab`, `own_slab_engine` | Across the enclosure's own gradient the surface lies in a slab of half width `e / \|α\|`: no tilt term | Proven |
| `Own.mesh_in_own_slab`, `field_on_mesh` | A triangle whose corners are zeros of the field in the box lies in that slab, and the field on it is at most `2e` | Proven |
| `Sandwich.sandwich_zero` | The field positive on a patch pushed out by `δ` and negative on it pushed in: every point of the patch is within `δ` of the surface | Proven |
| `Sandwich.sandwich_no_hole`, `sandwich_no_extra` | A sandwich of `δ` less than the side rays' reach leaves no hole and no extra | Proven |
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

#### What the engine said

Measured on the M4 at `69c6f34` (the GPU shared with another session's runs, so counts and
audits are firm and frame times are not). Exact cells are a frame's, at 1080p, frame 89; the
audit is the full one (640x360, 24 eyes, 20 kinds); the census is `certify_report`'s cells
(4,560 over 19 kinds at 1, 2.7 and 8 cm).

| Tried | How | Result |
| --- | --- | --- |
| A. The best multipliers | In `certify_sheet`, CPU reports and exact cells | Not carried. Certified cells 5,619 to 5,715 of 9,520, but exact cells on SCENE 10, 13, 14 moved -1.3%, -0.1%, +0.6%: with the plane's normal fixed, the engine's slab was already near the surface's true extent |
| B. The mesh's corners in the slab | A quad's cells solved where its corners stand more than a quarter of the need off a cell's plane | Not carried. It clears the pillars' holes as C does, at +77% exact cells on SCENE 10 against C's +5%. No hole the audit finds needs it |
| C. The reach across a diagonal outline | `HOLE_REACH_PX` (`mesh::dc`) in place of `HOLE_PX` for a certificate's need | **Carried** (`828598d`). The arena's pillars: 16 holes at the outline to 0, in three runs of the old and two of the new. No kind gained a hole |
| The slab across the enclosure's own gradient (`Own.lean`) | `sheet_width` reads both ways from the same enclosures | **Carried**, with the next (`b79923e`). Census: 5,619 to 5,855 certified of 9,520; 46 counts rose, 4 fell by one or two cells |
| 1. A certificate that says a width | `matter::certify::sheet_width`, kept by vertex for every eye (`mesh::outline`) | **Carried** (`b79923e`); its own full audit had not finished when it was. SCENE 10 VIEW 4 (running), each pass alone: the certificates' pass 4.2 to 2.2 ms a frame (median; mean 6.6 to 2.5), none due once settled. Exact cells with C: SCENE 10 23,060 to 22,341, SCENE 13 16,355 to 16,256, SCENE 14 10,917 to 10,908 |
| 2. The mesh in the middle of its slab | Census, then a rule in the engine: how far the slab reaches from the vertex's plane to either side | Open. The slab's middle is not where a mesh of chords lies (it lies inside a surface that bulges, and the slab's middle is inside it too), so no vertex was moved. The reach rule certifies 6,328 of 9,520 on the census; its audit has not run |
| 3. Draw the slab | Census of slab widths | Not built. Of the 2,133 cells that fail at the need, 531 have a slab under a pixel wide (at 2.5 pixels a cell) and 1,051 have one wider than their cell: too few to pay for a second raster path before the enclosures are tighter |
| 4. Certify on the mesh (the sandwich, `Sandwich.lean`) | CPU probe on a patch of three surface points a cell apart; then in `outline_quads`, each frame, no cache | Open, and the strongest. Probe, kinds not of the kit: 2,543 of 3,060 cells held at the need against the slab's 2,186 (the orb at 8 cm: 60 of 60 against 1). The kit's kinds: 15 of 1,500 (their enclosures are their own, not the compiler's form of their distance). In the engine it took SCENE 10's exact cells a further 1.9% down and SCENE 13 and 14 nowhere, and with no cache its scan costs 86 ms a frame on SCENE 10 VIEW 4. It needs a kept verdict a quad and the parts live in a quad's bricks |
| 4, by a block of cells | The slab of a block two cells wide | Not viable: 1,443 certified of 4,560 against a cell's 2,427 |
| 5. The outline as a field | `interval` of `dot(gradient, q - eye)` over a cell | No speed in it. The compiler encloses it and proves no outline in 2,097 of 3,060 cells not of the kit, but those are cells the mesh's facing test already passes over. It could prove that test's premise |
| 6. The cell size | `CELL_PX` at 2.0, 3.0 and 3.5 | Not read yet |

The census overstates what a frame gains. Surface cells sampled evenly gained 4% to 13%; a
frame's exact cells moved 0% to 5%, because more than a quarter of SCENE 10's come from the
extraction itself (folds, seams, thin cells) and most of SCENE 13's and 14's are the kit's, whose
enclosures have no slope to certify with.

#### To carry into wrela next

1. **The sandwich with a kept verdict.** A proof a quad, kept as the least `δ` proven (as
   `vert_cert` keeps a width), the field asked with the parts live in the quad's bricks. First
   the kit: a stated enclosure over a patch's own coordinates, or the sandwich proves nothing for
   the made things.
2. **The reach rule's audit** (idea 2), and its theory: what holds the mesh to the vertex's
   plane across a cell.
3. **The exact cell's live parts kept.** With no certificate due, the certificates' pass still
   costs 2.2 ms a frame on SCENE 10 VIEW 4: each frame it asks the field for the parts live in
   every failed cell again.

### Next

- The thin cells' cores (`watched_core`: a cell less a quarter pixel on every side) by the same
  model: they still hold `HOLE_PX`, not `HOLE_REACH_PX`.
- Second-order enclosures: the form, how it composes through `smin` and the maps, the remainder.
- Motion by structure: when a displaced surface is a graph over its rest surface.
- Steepness rules for std's and the kit's operators.
- Ring balance: rings by distance alone, snapped to parent bricks, give a 2:1 cut.
