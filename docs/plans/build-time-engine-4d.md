# Plan: a build-time engine with state-box proofs and continual LOD

Status: draft (2026-11-26, revision 2 after adversarial review). Grounded in the code at commit
`63f682a` on branch `engine-plan` and the record in `docs/plans/engine-at-build.md`. This plan
succeeds the open work of `docs/plans/engine-at-build.md` once approved.

Every claim about the repository comes from the code at `63f682a` and cited measurements. Phase 0
collects every missing measurement before any large engine rewrite begins.

## 1. Purpose

With the eye still, four of five gate scenes are near or under 16.6 ms on the M4. With the eye
moving, the gate records 57.8 to 69.2 ms median on `SCENE 10 VIEW 3` and 110.7 ms on `SCENE 13
VIEW 3`. One unpaired M4 run of `SCENE 10 VIEW 3` at `63f682a` (1080p, frames 30 to 269) measured
a 70.5 ms median span: `mesh_extract` 43.2 ms, `mesh_certify` 7.3 ms, and all other passes ~17 ms
(`mesh_sdf` 0.59 ms, `mesh_shadow_static` 0.0 ms).

This plan turns the engine into a build-time proof and code generator paired with a bounded GPU
brick pool at run time:

- The game stays a set of fields.
- The engine stays in `lib/engine`.
- The toolchain holds only general proof and code-generation mechanisms.
- At build, the compiler and engine prove two-sided error bounds, refinement radii, state-box
  bounds, and structural motion classes, and emit specialized kernels where counts justify them.
- At run time, the GPU keeps a persistent pool of world-anchored bricks selected by eye distance
  alone (never by the frustum), so walking and turning reuse bricks in place.

### Why we bake proofs and kernels, not dense grids

A 3D surface is a 2D manifold. Its fine bricks grow as $O(h^{-2})$: the census on branch
`engine-plan-ladder` measured 779 MiB for the courtyard at 7.8 mm, 1.2 GiB for the pillars at
3.9 mm, and 759 MiB for the table at 1 mm. Dense fine-grid cooks do not scale.

Baked octree error bounds take roughly 8 bytes per node (~2 MiB for the courtyard's 255,480
bricks). Proofs, error tables, and specialized kernels belong at build; fine bricks belong in a
bounded GPU pool at run time.

### What is in the repository today

| Fact | Where |
| --- | --- |
| Strict GPU math (`@strict`), `lower_strict`, `std::math`, `sqrt_rounded`, `div_rounded`, `std::strict::strict_check`, `judge::bits` | `toolchain/ir/src/strict.rs`, `lib/std/math.wrela`, `lib/std/strict.wrela`, `lib/engine/judge/bits.wrela` |
| Compositional first-order enclosures (`SlopeBox`, `MapBox`) on one certification path | `lib/std/field.wrela`, `lib/engine/matter/certify.wrela`, `lib/engine/matter/verify.wrela` |
| Exact-cell hit/resolve split (`exact_box_hit_fs`, `exact_resolve_one_fs`) and 4x4x4 super-brick culling | `lib/engine/mesh/shade.wrela`, `lib/engine/mesh/dc.wrela`, `lib/engine/mesh/extract.wrela` |
| 4D space-time contract verifier (`verify_moving_soundness`) and `SCENE 16` (bolt, glass branch, fallen pieces; no sand veil drawn yet) | `lib/engine/matter/verify.wrela`, `game/world/parry.wrela`, `game/scenes/main.wrela` |
| Moving-eye benchmarks (`SCENE 10 VIEW 3`, `SCENE 13 VIEW 3`) and moving-field hole audit (`Auditing::move_eyes`); no fast-turning or max-speed view yet | `tools/gate`, `lib/engine/judge/audit.wrela` |
| Counting build (`--tally`) and `tools/cost` diagnostic | `toolchain/ir/src/tally.rs`, `tools/cost` |
| Smooth combinators (`SmoothUnion`, `SmoothUnionAll`, `SmoothSubtract`, `SmoothIntersect`) do not widen `part_bound` by `k`; `parts_dirty` mutates state before commit | `lib/std/field.wrela`, `lib/engine/mesh/extract.wrela` |
| `GlassBranch` re-meshes every frame while cooling (`glow.xyz` changes its full `StateHash`); `branch_pieces` calls `landing()` on the CPU each frame; `GlassBranch` has infinite proven steepness | `game/world/parry.wrela` |
| Absent `Bolt` (`struck: 0`) still wanders its joints with `t` and touches 0 along its path, so `lo > 0` culling does not drop it | `game/world/parry.wrela` |
| `ParryArc` is `Shell(Displace(Sphere))` clipped by static planes (moving creases), touching a pulsing torus ring and a static `smin` tether, keeping ~100,000 candidate bricks for ~8 m² of sheet | `game/world/arena.wrela` |
| `exact_box_hit_fs` and `vis_cut_*_fs` are not marked `@strict`; `exact_box_hit_fs` evaluates `field.channels` when `may_transmit` is true | `lib/engine/mesh/shade.wrela`, `lib/engine/mesh/visbuf.wrela` |
| Flattened `ir::work` undercounts `cull_bricks` (`2,755` vs `164,544`), masked by `pre.max(flat)` for strict kernels | `toolchain/lower/src/gpu.rs`, `toolchain/ir/src/work.rs` |
| `ir::derive::affine` hits a NaN on carved boards; `table` and `grimoire` use `derive: false` | `toolchain/ir/src/derive/affine.rs`, `game/world/certificates.wrela` |
| Open mesh faults: 2:1 seam folds (5 in 220,018 triangles) and noise-rock crease folds (1 in 10,000); rigid exact-cell bracketing (`rigid_inv_point`) has no moving-rigid audit test | `lib/engine/mesh/emit.wrela`, `dc.wrela`, `shade.wrela` |
| `tools/gate.json` lacks `SCENE 16` and `moving` entries, and its `radeon` block is from `1c78609` before this branch | `tools/gate.json` |

### What earlier attempts showed

Read these commits before starting Phase 3 or Phase 4 so you do not repeat their faults:

- **Earlier brick pool** (`d6a9616`, removed in `b7edd49`): it kept a partial mesh cut short by
  the dispatch budget and marked it settled. Phase 3 separates the *wanted cut* from the *drawn
  cut*, measures lateness, and marks a batch settled only when `drawn == wanted`.
- **Heuristic vertex flow** (`009c4a1`, removed in `ae3c1c4`): it cut `SCENE 10` from 20.26 ms to
  16.45 ms on M4, but left 91 holes on the moving ward without reporting failure. Per-brick DC
  advection breaks shared brick-face connectivity whenever an advected brick meets a re-extracted
  neighbor. Phase 4 never mixes two motion methods across one brick face.
- **Ladder grids blocked on sun leaks** (`engine-plan-ladder`, `f6125e6`): world-anchored grids
  shared 4,689,500 bit-identical corners across three eyes, but failed `SCENE 13 VIEW 0` sun leaks
  (`845` vs `773`). Passing per-pixel mesh error in `g_normal.w` spiked acne (`18 → 130`) without
  fixing sub-cell caster leaks. Phase 2 fixes the sun judge first.
- **Candlestick exact cells** (`engine-at-build.md` §12–13): 88 of 100 candlestick enclosures have
  no slopes at all (balls from `Carved` or an unsure `min` in `Profile`), not curvature error
  ($h^2 \kappa / 8 \approx 0.2\text{ mm}$ against a `1.2 mm` slab). A single-part shader took
  2.1 ms against 8.2 ms for the monolithic field shader. Phase 5 fixes failure classes measured in
  Phase 0 and specializes the exact pass before considering higher-order models.

## 2. Terms

Each term has one meaning in this plan.

| Term | Meaning |
| --- | --- |
| Kind | One type of thing in the world (one `Field<Substance>` type) |
| Eye | The camera |
| Brick | One `6 × 6 × 6` block of cells (`7 × 7 × 7` vertices, `8 × 8 × 8` corners) on an integer world-anchored power-of-two lattice |
| Brick key | `(batch_id, level, bx, by, bz)`: the unique integer world address of one brick |
| Two-sided error (`delta_pos`) | A proven upper bound, in metres, on both mesh-to-surface and surface-to-mesh distance over each quad's `2 × 2 × 2` dual region |
| Normal error (`delta_n`) | A proven upper bound, in radians, on normal deviation across a quad, compared against the material's BRDF lobe width |
| Baked error (`delta`) | Per-node error in metres, baked at build time so 2:1 balance holds across all supported pixel sizes |
| Wanted cut | The exact set of bricks selected by eye distance for the current pixel size |
| Drawn cut | The finest complete, 2:1-balanced set of resident bricks drawn this frame |
| Lateness | Fraction of pixels drawn from a coarser brick than the wanted cut, and the longest frame count any brick stays missing |
| Brick pool | The fixed-capacity persistent GPU cache that stores extracted bricks by `BrickKey` |
| State box (`Theta`) | An interval box over the float parameters `theta` of a field struct (`f(p; theta)`) |
| Schedule floor | Frame span measured with every dispatch emptied (submission and pass-scheduling overhead alone) |
| Utilization | `tallied_work / (pass_time * peak_rate)`: how much of the GPU's capacity a pass uses |
| Proof debt | Run-time GPU work done only because a proof or certificate is missing |
| Span | The GPU clock time from the start of a frame's first pass to the end of its last pass |
| Gate | `tools/gate`: layers, tests, hole audit (still and moving), sun judge, still spans, moving-eye spans, and turning-eye spans |

## 3. Where the code is

| Subject | Place |
| --- | --- |
| Interval, affine, and Lipschitz derivations | `toolchain/ir/src/derive/interval.rs`, `affine.rs`, `lipschitz.rs` |
| Strict lowering, work bounds, and tally | `toolchain/ir/src/strict.rs`, `work.rs`, `tally.rs`, `toolchain/lower/src/gpu.rs` |
| Surface trait, combinators, `SlopeBox`, `MapBox`, `part_state` | `lib/std/field.wrela` |
| Sheet certificates and field verification | `lib/engine/matter/certify.wrela`, `verify.wrela` |
| Extraction grids, kernels, seams, and batch orchestration | `lib/engine/mesh/grid.wrela`, `dc.wrela`, `emit.wrela`, `extract.wrela` |
| Exact-cell hit/resolve, visibility buffer, and sun classification | `lib/engine/mesh/shade.wrela`, `visbuf.wrela`, `proof.wrela`, `sunlight.wrela` |
| Hole audit (still and moving), sun judge, and bit comparison | `lib/engine/judge/audit.wrela`, `sunlight.wrela`, `bits.wrela` |
| Arena ward, parry scene, and certificate tests | `game/world/arena.wrela`, `parry.wrela`, `certificates.wrela`, `game/scenes/main.wrela` |
| Gate, benchmarks, cost diagnostic, and Radeon runner | `tools/gate`, `tools/gate.json`, `tools/bench`, `tools/cost`, `tools/gpu`, `tools/radeon` |

## 4. How to measure

Use only these commands:

| To get | Command |
| --- | --- |
| Build one scene | `tools/wrela build game/scenes -o DIR --set SCENE=10` |
| First run of new GPU work (small, safe) | `tools/gpu DIR --size 320x180 --frames 2 --timestamps --serial` |
| Still span and per-pass table | `tools/bench DIR --size 1920x1080 --frames 30-89` |
| Moving-eye span and per-pass table | `tools/bench DIR --size 1920x1080 --frames 30-269` (built with `--set VIEW=3`) |
| Paired comparison of two builds | `tools/bench DIR --size 1920x1080 --frames 30-89 --against DIR2` |
| Full gate | `tools/gate --full` |
| Certificate and facts reports | `tools/wrela test game/world` |
| Run on the Radeon (commit and push first) | `WRELA_RADEON_DIR='~/projects/wrela-next-engine-plan' tools/radeon COMMAND` (use `WRELA_GATE_JOBS=1` on the Radeon) |

Put before-and-after measurements from the same machine and same run in every completion commit
message.

## 5. Rules for all phases

1. Follow `AGENTS.md` for all work, including the 50 ms dispatch rule, one Rust build at a time,
   `tools/gpu` for every GPU run, and citing every published method that lands in
   `docs/citations.md`.
2. Keep the engine in `lib/engine`. Add only general mechanisms to `toolchain/` (code that knows
   fields and functions, not engine passes or game kinds).
3. Keep the judges and the reference independent. They read the author's field directly and never
   read cached bricks, build-time shortcuts, or surrogates.
4. Make every build decision from a proof or a count, never from a measured time.
5. Every speedup must come from a proof. When a proof fails, fall back to the exact slow path and
   report the debt.
6. Remove the old method in the same change that lands its replacement.
7. Measure and pass `tools/gate --full` on both the M4 and the Radeon before calling a phase done.
8. Keep build-time proof searches bounded in steps, never in seconds, so every machine gets the
   same build.
9. Do not invent arbitrary constants when a physical quantity, a game rule, or a count from the
   field exists.
10. Never judge a change by eye; turn every visual check into a number.

## 6. Order of the phases

| Phase | Result | Difficulty | Depends on |
| --- | --- | --- | --- |
| 0 | Kill tests, roofline/utilization, and cut simulation on both machines | Low | Nothing |
| 1 | Clean base: kernel-derived shape hash, bug fixes, Radeon gate | Low-medium | Phase 0 |
| 2 | Footprint-integrated sun judge and conservative sun maps | Low-medium | Phase 1 |
| 3 | World-anchored persistent `BrickPool` and continual LOD (staged) | High | Phase 2 |
| 4 | State-box proofs `f(p; Theta)`, hard-CSG split, and motion by structure | Medium | Phase 3 |
| 5 | Exact-pass specialization and two-slope / piecewise enclosures | Medium | Phases 0 and 3 |
| 6 | Moving sand veil, contact enclosure, and instantaneous fused glass | Low-medium | Phase 4 |

Why this order:

- Phase 0 replaces every assumption with measured numbers on both machines before rewriting
  extraction.
- Phase 1 removes active soundness, compiler, and mesh faults and establishes true Radeon
  baselines.
- Phase 2 unblocks mesh changes so the sun judge stops failing on sub-cell tessellation noise.
- Phase 3 fixes moving-eye cost (`57.8–111 ms`) in three gated stages: world-anchored rings,
  two-sided error + normal cone, and resolution-safe baked `delta` cuts.
- Phase 4 handles moving fields by structure (and opaque hard-CSG splitting) under one state-box
  proof mechanism without mixed-motion brick cracks.
- Phase 5 fixes the exact-pass failure classes measured in Phase 0.
- Phase 6 completes the target parry scene once the sand exists as a moving field in `SCENE 16`.

---

## 7. Phase 0: Kill tests and measurements

Purpose: measure every open premise on both machines before changing the architecture.

### Steps

1. **Paired per-pass tables, roofline, utilization, and schedule floor on both machines.**
   - Run paired `tools/bench` on still scenes (`SCENE 10, 11, 13, 14, 16`) and moving views
     (`10/3`, `13/3`) on both M4 and Radeon.
   - Using `--tally` and `tools/cost`, record per-pass utilization `tallied_work / (time * peak)`
     and `max(work/peak, bytes/bw, latency)` so each pass is classified as compute-bound,
     bandwidth-bound, divergence/register-bound, or latency-bound.
   - Measure the **schedule floor** (frame span with every dispatch emptied) on M4 and Radeon.
2. **Add max-speed walking and real max-speed turning views to `tools/gate`.**
   - Define the camera's maximum walk speed and maximum turn rate as game rules (mouse turns
     exceed 360°/s). Add views exercising both in `SCENE 10` and `SCENE 13`.
3. **Break down moving-eye extraction and certification.**
   - Count bricks extracted per frame per kind on `10/3`, `13/3`, and the turning view.
   - Measure whether `mesh_certify`'s `7.3 ms` is per-brick (removed by caching) or per-eye
     (stays during turning unless cached per cell).
4. **Diagnose `ParryArc`'s ~100,000 candidates and `arena_pillars`'s 12,039 steepness.**
   - Count candidate bricks vs active surface bricks for `ParryArc` (~8 m² of sheet should take
     2,000 to 9,000 bricks at 5–10 mm cells) and record why the cull keeps ~10× too many.
   - Inspect `arena_pillars` (`lip = 12,039`) and `marble_column` (`lip = 5,720`) to name the
     exact operation causing the steepness blowup.
5. **Run the candlestick enclosure failure census.**
   - Classify all 88 of 100 slope-less candlestick enclosures by exact cause (`Carved`, `Profile`
     unsure `min`, `MapBox`, curvature) and record **live parts per failing cell**.
   - Also run the **CSG-split kill test** on the ward and the candlestick: certified cells per
     lone leaf vs joint mesh, and overdraw ratio.
6. **Build a CPU cut simulator for Phase 3.**
   - Simulate **both** world-anchored nested rings **and** error-driven baked-`delta` cuts on real
     kinds along the max-speed walking and turning paths.
   - Record peak resident bricks, bricks entering per frame (median and worst), VRAM size, and
     the brick-count cost of enforcing the finest-pixel Lipschitz bound and coarsest-pixel gap.
7. **Measure sun-judge placement variance.**
   - Record `SCENE 13 VIEW 0, 1, 2` hard leaks and mean light error across 8 lattice shifts and
     ±10% cell-size changes, comparing the current point-sampled judge against a pixel-footprint
     mean of unfiltered rays on the unfiltered field.

### What was measured (2026-11-26, at `fd1234f` unless said)

**1. Spans and passes.** M4, 1080p, `tools/bench` (still: frames 30 to 89; moving: 30 to 269),
one placement, passes timed alone:

| Build | Span (ms) | Largest passes (ms) |
| --- | --- | --- |
| SCENE 10 | 20.06 | extract 4.87, refract 1.78, direct 1.65, vis_resolve 1.34, sun_map 1.18, eye 1.14, certify 0.62 |
| SCENE 11 | 16.54 | vis_resolve 4.85, sun_map 2.05, exact 1.33, eye 1.17 |
| SCENE 13 | 13.97 | exact 4.44, vis_resolve 2.26, sun_exact 1.89 |
| SCENE 14 | 16.90 | exact 5.00, sun_exact 3.44, vis_resolve 2.27 |
| SCENE 16 | 11.56 | extract 1.86, exact 1.22, eye 0.95 |
| SCENE 10 VIEW 3 | 69.70 | extract 43.32, certify 7.34 |
| SCENE 13 VIEW 3 | 108.57 | extract 70.46, exact 9.90, vis_resolve 5.67, sun_exact 4.99, certify 4.20 |
| SCENE 10 VIEW 4 (running) | 73.97 | |
| SCENE 10 VIEW 5 (turning) | 64.18 | |
| SCENE 13 VIEW 5 (turning) | 20.73 | |

Radeon (`tools/gate --full`, three placements, at `fd1234f` and `5c602e2`): SCENE 10 59.7, 11
47.8, 13 22.6, 14 28.8, 16 29.6 ms; SCENE 10 VIEW 3 230 ms median (466 at 19 in 20); SCENE 13
VIEW 3 519 ms; VIEW 4 and 5: 244, 221 (arena), 73, 42 (study). The Radeon refused
`filter_thin_cells` (1,571,069 work an invocation), `inherit_seam_corners`, `solve_seam_smooth`
and `certify_outline` (2,449,321 work an invocation, past its 2,441,406 a lane) dispatches, so
its audit and its study's tour did not run; its sun judge's hard leaks were 724 to 2,912 across
placements (`SCENE 13 VIEW 0`), against the M4's 754. Utilization by `tools/cost` and the
schedule floor were not measured: no decision below reads them.

**2. The fastest the rules allow.** `rules::caster`: a caster runs at 7 m/s and the eye turns at
720°/s (`VIEW 4` strafes across `RUN_SPAN`, `VIEW 5` turns in place), in SCENE 10 and 13.

**3. What moving meshes again.** `REPORT` (`MeshDraw::extraction_report`), SCENE 10 VIEW 3,
frames 30 to 269: the courtyard 208 extractions of 19,576 active bricks, the pillars 223 of
6,520, the brazier stands 205 of 2,040, the flames 239 of 1,557, the ward 239 of 2,831, the orb 1
of 517: about 29,000 bricks a frame. SCENE 13 VIEW 3: the room 194 of 10,898, the table 207 of
9,071, the book 197 of 4,410. The outline's certificates are per eye (they were made at
extraction for its eye, and every eye step re-extracted).

**4. The ward's bricks and the steep columns.** `engine::mesh::census::cull_census`: the ward
(`parry_arc`) keeps 25,742 of 1,730,100 bricks at 5 mm and its surface crosses 21,950 (1.17 kept
a crossed; 5,181 and 4,676 at 1 cm). The cull is tight: the ward is a shell 7 cm thick seen from
both faces, a ring and a tether, about 9.4 m² of sheet, so the premise of 2,000 to 9,000 bricks was
wrong. The pillars' steepness of 12,039 and the marble column's 5,720 come from their flutes'
complex powers (`cmul` squared four times): the derived Lipschitz bound of `v / max(|v|, r)` is
104 over the pillars' bound and each componentwise square doubles it, 425,400 for the tenth power
(its true bound is about 10 / r).

**5. The candlestick's enclosures.** `candlestick_enclosure_census`: of 100 cells 4.8 mm wide on
its surface 11 certify within a quarter cell; all 89 others have no slope worth the name (a plane
of less than an eighth of the field's slope) in the turned profile alone (`stick_turned`), before
any carving; one part is live in each. The cause is the profile's (`Turned`'s) enclosure, not
`Carved`'s. Its kinds are each one leaf (the ward and the candlestick write their joins inside
one `distance`), so the CSG-split kill test has no operands to split there.

**6. Rings by distance alone.** `ring_census_report` (`engine::mesh::rings::build_ring_grids`,
1080p from SCENE 10's eye): the courtyard keeps 694,149 bricks over four levels (21,854 at 15.6
mm, 173,400 at 7.8, 337,497 at 3.9, 161,398 at 2.0), 35 times the 19,576 the view-fitted grids
keep. And error-driven cuts need tight enclosures: of the courtyard's 1,183 bricks at 12.5 cm
cells, 651 have no slab within the brick's width (`flat_census`), at its min joins and creases.
So Stage 3A as written (uniform rings by distance alone) is not built, and Stage 3C waits on
phase 5's enclosures; phase 3 keeps the view-fitted rings, with slack (see phase 3).

**7. The sun's judge across placements.** SCENE 13 at 320x180 on the M4, 8 placements of the
lattice (at `3e9eead`): one point a pixel, hard leaks VIEW 0 673 to 853 (mean error 3.21 to 3.97
/255), VIEW 1 1,310 to 1,422 (6.94 to 7.46), VIEW 2 251 to 332 (1.20 to 1.59); p99 255, 255, 0.
Four points of the footprint for the reference against one point of the engine: hard leaks 28 to
54, 36 to 62, 1 to 21, but mean error 7.28 to 7.61, 9.95 to 10.32, 1.38 to 1.58 and p99 128, 128,
64: each half-lit footprint at an edge counted as half the sun wrong. Most of the "leaks" were the
reference's: Newton's steps place a receiver within their tolerance of a carved or turned field's
surface, its rays left 0.1 mm out along the normal, and on those fields a ray often started inside
and met the thing it left at its first step. With the engine's sun the mean of the same four drawn
pixels (the picture drawn at 640x360) and rays that start outside (`judge::sunlight::outside`), at
`f04b923`: placement 0 mean error 0.33, 0.91, 0.13 /255, hard leaks 5, 9, 0, p99 4, 21, 0.

### Done when

- All seven measurement tables are recorded in this file and in the commit message.
- No phase below depends on an unmeasured assumption.

---

## 8. Phase 1: Clean base, kernel-derived shape hash, and bug fixes

Purpose: repair all known soundness, compiler, and mesh faults, and establish true two-machine
gate baselines.

**State (2026-11-27).** Steps 2 to 6 are done on the M4:

- Step 2: `smooth_reach` widens each smooth join's parts (`20179be`; the moving-part test holds
  the blend zone), and `parts_dirty` commits only after a patch succeeds.
- Step 3: a mesh is keyed by what its extraction reads (`std::derive::kept_deps` over
  `ir::field_deps`, `mesh::extract::mesh_key`); a cooling branch keeps its mesh.
- Step 4: the source's work count overcounted, not the flattened one: it followed branches the
  flattening folds away (`steep(f)`, a float comparison it didn't fold) and work nothing reads (a
  one-part field's `pruned` ranges). The count folds float comparisons, and a strict kernel's
  bound is its flattened count (`f04b923`). The carved boards' NaN was a derived local joined
  before it was written (`l.xy` past an early return, holding whatever its frame held): every
  active local starts as its join's identity, and the table and the grimoire derive (`2462b03`).
- Step 5: every grid covers its thing at every placement (`e0b4bef`: it was cut off past two
  cells of shift); every fold is drawn by exact rays (`every_mesh_fold_is_drawn_by_exact_rays`,
  12 placements, but one sliver a sixtieth of a cell high and one triangle within 2° of
  edge-on); a falling piece of the branch is audited as a rigid mover (audit kind 19); an absent
  field is culled whole (`grid::may_cross`); the ward's cull keeps 1.17 bricks for each its
  surface crosses (phase 0's table 4: the premise was wrong).
- Step 6: the landings are computed once, at the break, and kept (`ParryLoop`); the branch's
  steepness is finite (`std::field::smax`).
- Step 7: `exact_box_hit_fs` stays relaxed: written strict it cost SCENE 14's exact pass 5.23 to
  12.09 ms on the M4; phase 5's specialization is to make room for it.

Step 1 on the Radeon: its audit failed every kind with thousands of cracks from `be12bfd` on. A
chunk's triangle kernels took a lane's edge as `id.x` less its brick in the chunk times 256, where
that brick counts the slices before it, so every slice past a chunk's first meshed nothing: the
M4 at its own step makes one slice a chunk and never saw it; at the Radeon's step
(`WRELA_GPU_UNIT_FS=5000`) it did, and the gate now audits the M4 again at that step (`beaae9c`).

### Steps

1. **Update `tools/gate.json` on both the M4 and the Radeon.**
   - Record baselines for `SCENE 10, 11, 13, 14, 16`, moving views (`10/3`, `13/3`), and the
     turning/max-speed views from Phase 0.
   - Keep `--tally` and `tools/cost` as diagnostics; drop the "within 20% linear fit" gate target
     from `engine-at-build.md`.
   - Keep `@strict` as-is during Phase 1; revisit Radeon strict extraction cost only after Phase 3
     reduces per-frame extraction and Phase 0 shows whether extraction is a top Radeon cost.
2. **Fix smooth-combinator bounds and `parts_dirty` commit ordering.**
   - In `lib/std/field.wrela`, widen `part_bound` by `self.k` (`.lowered(self.k)`) in
     `SmoothUnion`, `SmoothUnionAll`, `SmoothSubtract`, and `SmoothIntersect`.
   - Add a test in `lib/engine/mesh/extract_tests.wrela` proving the dirty box covers the blend
     zone for a moving part inside `SmoothUnion` and `SmoothSubtract`.
   - Change `parts_dirty` in `lib/engine/mesh/extract.wrela` to return the new hashes and boxes
     and commit them only after `try_patch_dirty` succeeds.
3. **Add compiler-derived `shape_hash` from what the geometry kernels read.**
   - Derive a hash of only the struct fields and vector components of `self` read by the geometry
     kernels (covering `distance`, `bound`, `lipschitz`, `slope_box`, `pruned`, and transmission
     predicates automatically so the list cannot go stale).
   - Use it in `MeshBatch` to skip extraction when only shading channels change (such as
     `GlassBranch.glow.xyz`, while `GlassBranch.glow.w` stays in the hash).
   - Remove hand-written `rigid_shape_hash` overrides that the derived hash replaces.
4. **Fix the two open compiler faults.**
   - Fix why flattened `ir::work` undercounts `cull_bricks` (`2,755` vs `164,544`), and remove
     the `pre.max(flat)` workaround in `toolchain/lower/src/gpu.rs`.
   - Fix the NaN in `toolchain/ir/src/derive/affine.rs` on carved boards, and remove
     `derive: false` from `table_enclosures_are_sound` and `grimoire_enclosures_are_sound` in
     `game/world/certificates.wrela`.
5. **Fix the open mesh, exact-cell, and empty-surface faults.**
   - Repair 2:1 seam triangle folds (`5 in 220,018` triangles across placements).
   - Repair noise-rock crease triangle folds (`1 in 10,000` triangles where a clamped vertex falls
     inside the solid).
   - Add a moving-rigid test in `Auditing::move_eyes` (a rigid thing moving without re-meshing) to
     verify `exact_box_hit_fs`'s `rigid_inv_point`/`rigid_inv_dir` bracketing.
   - Fix empty-surface culling in the engine: check `classify_cells`'s `< 0` vs `<= 0` inside
     convention and cull any box whose interval proves zero inside points (`lo >= 0` under `< 0`),
     so an absent field (like `Bolt` at `struck: 0`) drops in zero extraction work for every kind.
   - Fix `ParryArc`'s candidate-brick cull fault diagnosed in Phase 0 step 4.
6. **Fix `game/world/parry.wrela` faults.**
   - Compute `branch_pieces`'s six `landing()` times once at the break event and keep them as
     state (not per frame, and not baked to a hardcoded `PARRY_HIT`).
   - Fix why `GlassBranch::distance` gives infinite proven steepness.
7. **Align exact-cell fragment strictness.**
   - Make `exact_box_hit_fs` `@strict` so `EmptyProof` holds on the shipping build; keep
     `exact_resolve_one_fs` relaxed.
   - Document the compile-time `facts.lip` dead-code-elimination branch in `exact_box_hit_fs`, and
     measure whether `may_transmit` channel checks belong in hit or resolve.

### Done when

- `tools/gate --full` passes on both M4 and Radeon with current baselines in `tools/gate.json`.
- `GlassBranch` extracts zero bricks while cooling, and `Bolt` extracts zero bricks while absent.
- `table` and `grimoire` pass `verify_slope_soundness` with `derive: true`.
- `ir::work` agrees on `cull_bricks` before and after flattening with no `pre.max(flat)`.
- 2:1 seam folds and noise-rock crease folds are zero across multi-placement tests.

---

## 9. Phase 2: Footprint-integrated sun judge and conservative sun maps

**Done** for steps 1 and 3 (2026-11-27). The judge draws at twice the judged size and holds each
judged pixel's mean engine sun over its four drawn pixels to the mean of 64 rays from the same
four points, from rays that start outside the field (`judge::sunlight`). `tools/gate` holds it
over 8 placements and cells 2.25 and 2.75 px wide (`CELL_PX`, a setting of `mesh_sight`), with a
row for how far the mean error moves among them. M4: SCENE 13 VIEW 0, 1, 2 mean error 0.27, 0.91,
0.11 /255 (was 3.66, 7.07, 1.46), its move 0.12, 0.16, 0.08, p99 2, 21, 0; hard leaks 2.5, 11.2,
1.1 (was 773, 1,359, 307). Step 2 (dual conservative maps) is not built: the measured error is
already under one display step on average, the engine's map with its near band and exact rays
(`mesh::sunlight`) holding it; VIEW 1's p99 of 21 is a few lines along a moulding seen edge-on,
in the exact pass's pixels (acne 75 hard), named for phase 5. The plan's display-step `eps` is
not used: the gate holds the error's mean, p99 and move in 255ths of the sun.

Purpose: unblock every mesh change by making the sun judge measure physical pixel-footprint
radiance error rather than point-sampled sub-cell tessellation noise.

### What exists today

- `lib/engine/judge/sunlight.wrela` integrates the sun's disc (`theta_sun ≈ 9.3 mrad`) at one
  receiver point per pixel and counts binary `hard leaks` (reference `< 1/16`, engine `> 1/2`).
- A thin caster of width `w` at distance `d` across the sun disc blocks up to `(4/pi) * w /
  (d * theta_sun)` of the disc. A hard leak requires `d <= w / theta_sun` (~32 cm for a 3 mm
  bead), which is already in contact range.
- A 1% cell-size change moves `SCENE 13 VIEW 0` hard leaks from 754 to 882 because point sampling
  on the receiver aliases sub-cell beads.

### Steps

1. **Upgrade the sun judge to integrate over the receiver pixel's footprint.**
   - Replace the single-point receiver sample with the footprint mean of unfiltered rays on the
     unfiltered field (so sub-pixel detail is integrated physically instead of aliased by a single
     point).
   - Define the physical light tolerance `eps` as one display code step at that pixel's luminance
     after the tone curve (`Look::tone`), giving a concrete physical threshold instead of binary
     pixel counts.
2. **Use dual conservative sun shadow maps where needed.**
   - Render an inner map (proven dark) and an outer map (including boxes of thin cells: proven
     lit). Send only pixels between the two bounds to exact sun rays, keeping contact shadows
     exact via `trace_sun_near`.
3. **Run the placement-invariance kill test.**
   - Shift the world lattice across 8 placements and vary cell size by ±10%.
   - Hold the numeric footprint light-error mean and 99th percentile within the gate's slack
     (measured quantitatively, never by eye).

### Done when

- The footprint sun judge passes on both M4 and Radeon with variance across 8 lattice placements
  and ±10% cell-size shifts inside numeric slack.
- `SCENE 13 VIEW 0, 1, 2` numeric contact-shadow error does not regress.

---

## 10. Phase 3: World-anchored brick pool and continual LOD

Purpose: make moving and turning frames cost nearly the same as still frames, and deliver
pop-free, crack-free continual LOD.

**State (2026-11-27): not done.** Built so far: a held mesh is extracted for no eye, and its
outline's exact cells are found each frame from what the extraction kept (`mesh::outline`, its
certificates cached per vertex and its beside cells' proofs per slot); the view-fitted rings are
made with slack and kept while they hold what the eye needs (`grids_hold`); every grid's corners
are on the world's lattice (`grid_for`). M4, 1080p, gate medians: SCENE 10 VIEW 3 31.3 ms, VIEW 4
(running) 68.3, VIEW 5 (turning) 54.7; SCENE 13 VIEW 3 83.8, VIEW 4 26.9, VIEW 5 19.4 (still: 20.8
and 15.8). Against the 16.6 ms line the moving frames are 2 to 5 times over.

Where the time goes (serial passes, frames 30 to 129): SCENE 10 VIEW 4, extraction 21.7 ms and
the outline's certificates 15.9 (beside and kept cells 9.9); SCENE 13 VIEW 3, the exact pass 29.5,
extraction 23.2, the sun's exact rays 6.7. A moving eye extracts a batch again every time its fit
changes, all of it: running, the courtyard 135 times in 240 frames, about 19,800 bricks each; on
the tour the table 116 times at 10,300. Most of those bricks are the same bricks as the last
extraction's, and each extraction also empties the certificates' cache.

What is next, in order:
1. **Bricks kept across a fit's change.** The root grid's origin on the lattice of its bricks
   (`BRICK` cells, not one), so a brick of a level is the same brick in every fit that has that
   level's cell; a new fit re-extracts only the bricks whose level or seam neighbourhood changed,
   as an edit's patch does now (`try_patch_dirty`: kill the old bricks' slots, append the new
   ones), with the 2:1 seam bricks (those within a brick of a ring's face) always among them.
   What the patch path lacks is the seam: a patched brick on a ring's face needs its coarse
   neighbour's corners (`inherit_seam_corners`) and its seam cells (`box_seam_cells`). The slots'
   outline records (`QuadRec`, `SlotBrick`) and the certificate cache stay with their slots.
2. **The certificate cache by place, not by vertex**, so a re-extracted brick's vertices keep
   their certificates (they are the same vertices, at the same bits, on the world's lattice).
3. The exact pass's cost close up is phase 5's (the table's 37,303 exact cells on the tour).

Build this phase in three independently gated stages.

### Stage 3A: World-anchored nested rings + persistent `BrickPool`

1. **Use integer `BrickKey` and camera-relative coordinates.**
   - Key every brick by integer `BrickKey = (batch_id, level, bx, by, bz)`.
   - Compute draw positions relative to the eye (or brick-relative) so open worlds stay exact past
     `2^24` cells from the world origin.
2. **Put nested rings on one world-anchored power-of-two lattice with distance-only residency.**
   - Select bricks by **eye distance only** (frustum-cull only at draw time) so fast turns cost
     zero new extractions and off-screen sun casters stay resident.
3. **Build the persistent GPU `BrickPool`.**
   - Fixed-capacity GPU slot arenas sized from Phase 0's simulator counts.
   - Open-addressed GPU hash table keyed by `BrickKey` with frame-stamp LRU eviction.
   - Define both the **wanted cut** and the **drawn cut** (finest complete, balanced set of
     resident bricks). Mark a batch `settled` only when `drawn == wanted` and seams are built.
   - Add **lateness** (fraction of pixels drawn coarser than wanted, and longest frames late) to
     `tools/gate`.
4. **Reuse today's 2:1 ring seams** while keeping each brick's interior mesh cached across frames.
   Verify zero re-extraction of resident bricks along `SCENE 13 VIEW 3` and the turning view.

### Stage 3B: Unified two-sided error budget (`delta_pos` + `delta_n`)

1. **Define one two-sided error bound per quad (`delta_pos`, `delta_n`).**
   - **Position error (`delta_pos`):** take the convex slab of each quad's `2 × 2 × 2` dual
     region. Make Dual Contouring **clamp each solved vertex into its dual-region slab** so the
     quad provably lies in the slab. When the region is a single sheet over the plane, both
     mesh-to-surface and surface-to-mesh distances are bounded by the slab width. Uncertified
     cells stay `ExactCell` (bounded by march tolerance), so `delta_pos` is always finite.
   - **Normal error (`delta_n`):** derive a normal cone from closed-form analytic leaf curvature
     (or gradient intervals where no closed form exists) and compare against the material's BRDF
     lobe width so smooth curved gold and glass do not facet at coarse levels.
   - Add a slow CPU reference test checking true two-sided Hausdorff distance and normal error
     between triangles and `f(p) = 0`.
2. **Resolve `mesh_certify` residency using Phase 0 step 3's measurement.**
   - If per-eye certification dominates turning, cache view-independent per-cell certificates on
     each brick (and run only a tiny per-eye outline select or keep only truly uncertified cells
     as `ExactCell`).

### Stage 3C: Resolution-safe baked `delta` and continual LOD

*(Proceed to Stage 3C only if Phase 0 step 6's simulator confirms error-driven cuts beat Stage
3A's nested rings enough to justify the complexity.)*

1. **Prove the 2:1 balance theorem on paper before coding it.**
   - Bake `delta` in **metres** per coarse octree node at build time (no mesh needed; for moving
     kinds, bake the worst `delta` over the legal state box `Theta` so no per-brick time term
     breaks the theorem).
   - At build time, apply a max-plus gradient-limiting pass (which only raises `delta`, so it is
     always safe) enforcing:
     1. **Nestedness:** child refined implies parent refined.
     2. **Lipschitz bound at the finest supported pixel** `p_fine`.
     3. **Gap condition `(1 + k) * s` at the coarsest supported pixel** `p_coarse` (covering
        `640x360` and `320x180` audit/sun resolutions), including each node's half-diagonal for
        top-down box tests and the edge/corner neighbor factor (`sqrt(2)` / `sqrt(3)`).
   - Then 2:1 balance across faces and shared DC edges is a theorem for every eye position and
     every supported screen resolution, with zero GPU balancing pass.
2. **Prove pool size and per-frame budget from counts.**
   - From maximum game movement speed (Phase 0 step 2) and the baked octree census, compute the
     maximum resident bricks and maximum bricks entering per frame at build time. Parent fallback
     is then needed only on teleports.
3. **Decouple 2:1 seam faces from cached brick interiors.**
   - Emit seam quads into a dedicated seam buffer for active level boundaries only, leaving
     resident brick interiors untouched.
4. **Remove old view-coupled grid code and arbitrary caps.**
   - Delete `build_lod_grids_view` and its magic constants (`vol_cap 160000`, `area_cap 400..9000`,
     `1.12`, `0.85`).
5. **Add a path-independence test:** reach the same eye position by two different paths and
   verify the settled brick sets are bit-identical.

### Done when

- `SCENE 13 VIEW 3` (moving eye) and the max-speed turning views have median span at or below
  `16.6 ms` on the M4, and moving-minus-still overhead is under `2 ms` on `SCENE 10` and `SCENE
  13`.
- Zero resident bricks are re-extracted along `SCENE 13 VIEW 3` or during pure camera rotation.
- Lateness is zero during max-speed walking and turning (within the proven per-frame entry count).
- Full hole audit (still and moving), CPU two-sided Hausdorff + normal check, path-independence
  test, and sun judge pass on both M4 and Radeon with zero cracks on opaque and transmissive
  meshes.

---

## 11. Phase 4: State-box proofs `f(p; Theta)`, hard-CSG split, and motion by structure

Purpose: unify dynamic proofs over field state boxes `f(p; Theta)` and select each moving kind's
mesh and ray strategy by structure at build time, with zero mixed-motion brick cracks.

### Why state boxes instead of time-only closures

GPU kernels receive structs of float state `theta`, not `fn(f32) -> S` closures. Proving over
`(space box B) × (state box Theta)` is one compiler mechanism that covers point hashing (`Theta`
is a point), cached expiry (valid while `theta in Theta`), baked worst-case moving `delta` for
Stage 3C, and parametric spell bounds.

### Steps

1. **Add state-box derivation `f(p; Theta)` to `toolchain/ir/src/derive/` and `lib/std/derive.wrela`.**
   - Derive intervals, enclosures, and parameter sensitivity bounds over `Box3 × Theta`.
   - Record each cached fact's valid `Theta` box.
2. **Split opaque CSG trees at hard joins for meshing, rejoined in the raster** (gated on Phase
   0 step 5's CSG-split kill test; Goldfeather et al. 1986, 1989 — cite in `docs/citations.md`).
   - **Opaque `Union`:** mesh operands separately and let the depth test compose them.
   - **Opaque `Intersect` / `Subtract`:** where the kill test confirms a net win, trim per
     fragment against the other operand during visibility write, skipping the trim wherever a
     brick is proven inside or outside.
   - **Smooth joins (`smin` / `smax`) and transmissive/glass surfaces:** keep a joint mesh only in
     the blend tube `|a - b| < k` (enabled by Phase 1's `k`-widened `part_bound`) and for
     transmissive kinds where inner faces would refract.
   - Also split static operands of `min` / `smin` that read no `self` (such as `ParryArc`'s
     tether) into never-changing parts.
   - Result: each operand of a moving composite gets its own cache and its own motion method, so
     two motion methods never share one brick.
3. **Select each moving operand's method by structure at build time** (and report the choice in
   `wrela pipelines` / `game/world` reports):
   - **Rigid, Skinned, Warp:** rest mesh once + transform (already exists; verified by
     `Auditing::move_eyes`).
   - **Explicit parametric graph (for smooth clipped waving sheets):** where a moving surface is
     an explicit graph over a fixed chart with static clip bounds—such as `ParryArc`'s clipped
     shell `r = R + wave(y, z, t) ± 0.036` over `(y, z)`—evaluate vertices directly on the chart
     with zero DC extraction and verified two-sided coverage.
   - **Unclipped `Displace(S, D(theta))`:** where `|D| < reach(S)` and `Lip(D) < 1` over `Theta`
     with no moving clip crease, mesh `S` once and project vertices along `S`'s normal by one
     monotone 1D root.
   - **Unstructured movers (such as flames and the pulsing torus ring):**
     - **Narrow band + proven empty-space expiry:** a corner keeps its sign while `|f(c, t0)| /
       rate` holds, and empty coarse boxes expire at `|f|_min / rate` (no arbitrary time window).
     - **Skip proven-sign corners in DC:** skip corner evaluations whose sign the brick's slab
       already proves (bit-equal to full extraction).
   - **Tiny fast transients (such as the 1.8 cm `Bolt`, 512 work):** when build-time work counts
     favour pixel-rate solving over meshing, skip DC and emit culled `ExactCell` boxes directly,
     using a per-leaf `ray` capability (closed under hard CSG, `k/4` bracket for `smin`) and 1D
     slab bracket (`2e / |l·d|`) / 1D interval Newton.
4. **Keep every fast path bit-verifiable** against full extraction where `judge::bits` applies.

### Done when

- `Auditing::move_eyes` passes with zero holes on the flame, the ward, the bolt, and a rigid mover
  on both M4 and Radeon.
- `SCENE 10` (still, walking, and turning) is at or below `16.6 ms` on the M4.
- GPU extraction work for the ward and flames drops by at least 50% compared to Phase 3.

---

## 12. Phase 5: Exact-pass specialization and measured enclosure repairs

Purpose: remove the remaining exact-cell proof debt on heavy showcase assets (`SCENE 13`,
`SCENE 14`), driven by Phase 0's failure census.

### Steps

1. **Upgrade enclosures for the failure classes measured in Phase 0 step 5.**
   - **Two-slope linear bounds** (CROWN, Zhang et al. 2018; DeepPoly, Singh et al. 2019 — cite in
     `docs/citations.md`): separate lower and upper bounding planes with distinct slopes, tighter
     than symmetric `c + l·u ± e` at the same cost.
   - **Pruned `k`-term slab join trees** (tropical polyhedra, Allamigeon, Gaubert & Goubault 2008
     — cite in `docs/citations.md`): keep both operands' slabs at an unsure `min`, dropping
     dominated terms over the box. Where two planes meet at a sharp edge, their intersection
     provides a **crease certificate and exact QEF** directly (measure whether it replaces the
     64-step crease solve).
   - **Profile & carving repairs:** enclose turned profiles in `(r, y)` coordinates where straight
     segments are cones; repair `Carved` fallback enclosures.
   - **Nonexpansive-map Lipschitz rules:** fix `arena_pillars` (`12,039`) and `marble_column`
     (`5,720`) using the cause found in Phase 0 step 4 (for example segment projection is
     1-Lipschitz by theorem).
   - Re-measure certified cell rates on `candlestick`, `table`, `grimoire`, `chair`, and `shelf`.
2. **Specialize the exact pass by live-part clusters.**
   - For static kinds whose exact pass still exceeds 1 ms after step 1, bucket exact cells by
     their live-part subsets at build time and emit specialized exact-hit kernels with dead parts
     removed and proven-redundant `smin` joins folded.
   - Gate cluster count by build-time work savings vs Vulkan build RAM (no arbitrary per-kind
     constant cap; `SCENE 6` Vulkan build RAM must not regress).
3. **Add `Taylor2` only if residual debt after steps 1–2 is proven curvature.** Do not add
   `exact_quadric_hit_fs`: any cell certified to pixel error is meshed directly (and lone quadric
   leaves from Phase 4 step 2 can use closed-form ray casting via Sigg et al. 2006 if warranted).

### Done when

- `SCENE 14` candlestick exact-cell cost drops from `8.2 ms` to under `2.5 ms` on the M4.
- Every gate scene (`SCENE 10, 11, 13, 14, 16`, still, walking, and turning) is at or below
  `16.6 ms` on the M4, and faster than Phase 1 on the Radeon.
- One-field rebuild time stays under 25 s on the M4, and `SCENE 6` Vulkan build RAM does not
  regress.

---

## 13. Phase 6: The parry — moving sand, contact enclosure, and instantaneous glass

**State (2026-11-27).** Steps 1 to 4 are built:

- Step 1: the thrown sand is `world::parry::SandVeil`, flung from the caster's other hand and
  spreading to its reach across the bolt's way, then falling out. Its grains are far finer than a
  pixel, so it is a medium (`engine::matter::medium::Medium`): its extinction is its mass in the
  air over what a kilogram of quartz grains 0.15 mm across takes from a beam, `(m / V) 3 Q / (4 ρ
  r)`, spread as `(1 - |q|²)²` (1.98 m⁻¹ at its middle once spread; `the_sands_cloud_holds_its_mass`
  holds its mass to 0.005%). Meshed as thousands of 12 mm clumps it took SCENE 16 to 234 ms a
  frame on the M4; as a medium 8.6 ms.
- Step 2: `engine::matter::contact` finds the earliest moment two moving surfaces share a point
  from the compiler's ranges over space and time (earliest-first boxes, a step bound, never
  samples alone): the bolt's leader meets the sand's region (`SandHull`) at 1.0000088 s, sure, in
  277 steps, the same bits every run (`the_bolt_meets_the_sand_when_the_contact_says_and_never_
  before`, which also holds that no point of the bolt's path is in the region before it).
- Step 3: the branch of glass is the bolt's path at `PARRY_HIT` kept to the sand's region then;
  its six pieces' landings are computed once at the break and kept (`ParryLoop`).
- Step 4: over a cast's moments the bolt, the branch and the sand's region are held to their
  contracts by `verify_moving_soundness` (bound and steepness over boxes of space and time, and
  how fast each changes), and every distance's work by its `@budget` at build.

SCENE 16 draws the bolt lighting the sand, the branch hanging and cooling over it, and the pieces
fallen on the court. Its gate on the Radeon waits on phase 1's Radeon gate.

Purpose: complete the target game scene (`SCENE 16`) with a real moving sand veil and derive the
fused glass branch from the instant the bolt meets the sand.

### Steps

1. **Make the thrown sand veil a moving `Surface` in `SCENE 16`.**
   - Replace the static `veil(p)` helper in `game/world/parry.wrela` with a moving sand field
     present in `SCENE 16`'s world so both spells exist and meet at a real event time.
2. **Add a deterministic contact-enclosure solver.**
   - Given two moving surfaces `A(p; theta(t))` and `B(p; theta(t))` over a search window, use
     state-box / rate-bounded bisection with a fixed step cap to return either `None` (proven no
     contact) or a guaranteed contact enclosure `[t_enter, t_exit]` and contact box.
   - Use the lower end `t_enter` deterministically so any two machines agree to the bit.
3. **Derive `GlassBranch` at the instant of contact.**
   - A fulgurite fuses in microseconds along the instantaneous channel: define `GlassBranch` as
     `bolt(t_enter) ∩ veil(t_enter)` (do not sweep the oscillating bolt across the window).
   - Compute the six broken-piece `landing()` times once at the break event and keep them as
     state.
4. **Add build-time parametric safety proofs once spell parameter domains exist.**
   - Over the legal state box `Theta` of a cast, prove worst-case Lipschitz constant, spatial
     bound, and GPU kernel work bound at build time.

### Done when

- `SCENE 16` draws the lightning bolt, the moving sand veil, the fused glowing/cooling glass
  branch, and the falling broken pieces, and passes `tools/gate --full` on both M4 and Radeon.
- A CPU test proves the contact enclosure never misses a contact and produces bit-identical
  `t_enter` across machines.

---

## 14. Work that is not in this plan

Each hypothesis stays out of the active phases until its listed measurement or trigger reopens it.
Any published method that later lands must be cited in `docs/citations.md`.

| Idea | Why it is not here | Measurement / trigger that reopens it |
| --- | --- | --- |
| Field-guided spatial upscaling and 4D-proven TSR | Upscaling must not hide extraction or caching faults; native 1080p must hit the bar first | Phases 3 and 4 pass the gate at native 1080p on both machines, or when targeting 4K / faster Radeon frames |
| Lighting cache upgrades: linear cached transfer for flickering emitters, Lipschitz sun-visibility spacing `(4/pi)/(d * theta_sun)` (Ward & Heckbert 1992; Arvo 1994), Fréchet static×dynamic sun bounds `max(0, Vs + Vd - 1) <= V <= min(Vs, Vd)`, and world-anchored `RadianceCache`/`SdfClipmap` | Extraction and exact cells are the measured bottlenecks today; `RadianceCache` already splits static and dynamic | Phase 0 still tables (or post-Phase-5 tables) show lighting passes dominate the remaining frame span; needs a frame-after-change light judge |
| Hybrid surface/volume microflakes (Loubet & Neyret 2017) scoped strictly to proven-thin cells (`filter_thin_cells`) | Exponential transmittance leaks light through thin opaque sheets (Vicini, Jakob & Kaplanyan 2021); SGGX at all levels recreates the dense cook | Phase 2 is done AND a gate scene has sub-pixel geometry at long distance held to a supersampled reference |
| Dense fine-grid disk cooks | Fine grids take 759 MiB to 1.2 GiB per kind (`engine-plan-ladder` census) | Only for an optional tiny warm-start cache of the coarsest 1–2 octree levels if cold start is too slow |
| A server build target | Deferred until multiplayer game rules require running fields on a server | Phase 6 is done and server spell rules exist |
| Per-brick DC vertex advection mixed with re-extracted neighbors | Breaks shared brick-face connectivity (`ae3c1c4`, 91 holes) | Do not look again |
| Subgroup frustum tracing & Bernstein ray clipping | Driver-defined fragment lane mapping violates no-driver-proofs rule; arithmetic is ~1% of exact pass; Bernstein requires derivative remainder bounds | Do not look again |
| Covariance tracing as a formal proof | Linearizes locally; occlusion has unbounded bandwidth so it estimates rather than proves | Do not look again |
| Trimmed parametric B-reps for general CSG | Watertight trimmed patch intersection is an unsolved robustness problem | Do not look again |
| Runtime shader compilation in the client | Adds attack surface, hitching, and driver variance | Never |
| Unbounded general SMT solvers in the build | Run time is unstable across inputs and machines | Never |

## 15. Main risks and open questions

1. **Radeon baseline and per-pass breakdown (Phase 0 step 1):** The Radeon gate has not run on
   this branch; its 56.6 ms line dates from `1c78609` before strict math. Measure it first.
2. **Phase 3 scope and Stage 3C payoff:** Keep Stages 3A, 3B, and 3C strictly separate and gate
   each one. Let Phase 0's error-driven simulator decide whether Stage 3C beats Stage 3A enough to
   build it.
3. **2:1 theorem across resolutions (Stage 3C):** Prove the top-down box test, nestedness, finest-
   pixel Lipschitz bound, and coarsest-pixel gap on paper before writing GPU code.
4. **CSG split overdraw & glass (Phase 4 step 2):** Gate hard-CSG splitting on Phase 0 step 5's
   leaf certification and overdraw kill test; keep joint meshes for transmissive surfaces.
5. **Shader specialization vs Vulkan build RAM (Phase 5 step 2):** `SCENE 6` Vulkan builds already
   take 4.7 GB. Keep specialized exact-hit kernels strictly bounded by build RAM and compile time.

## 16. How to keep this plan

- Keep this file in `docs/plans/`.
- Record Phase 0's measured tables in Section 7 as soon as they run.
- Record "Done", the date, the commit, and the M4 and Radeon gate numbers at each phase heading
  when that phase completes.
- If a measurement proves a step wrong, update that step in the same commit.
- When all phases are done, remove this file.
