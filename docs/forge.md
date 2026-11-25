# The forge: from a brief to a gold asset

Agents write the whole game, and the forge is how they write assets that hold up close straight
from fields, without breaking the rule that nothing is prebaked (AGENTS.md):

```text
 brief ──▶ concept views ──▶ reference ──▶ outline ──▶ fit ──▶ judge ──▶ gold asset
 (text)    (2D images)      (3D, Colab)   (an agent    (wrela   (numbers)  (.wrela in game/,
                                           writes the  solve)              + provenance)
                                           program)
```

Only code ships. Concept images, reference meshes and samples are working material in `.forge/`
(git-ignored); nothing in `game/` or `lib/` reads them, at build time or run time. What lands is a
`.wrela` file an agent can read, name the parts of and edit by hand, and its provenance record
beside it. Change this file in the same change that makes it wrong.

## Where things are

| What | Where |
|---|---|
| The pipeline's commands | `tools/forge` (`tools/forge --help`) |
| Its checks (no GPU, ~40 s) | `tools/forge-test` |
| Its test fixture | `tools/forge-fixtures/leg/` (a leg written well and written badly) |
| Concept views and references on Colab | `tools/forge-notebooks/` |
| An asset's working material | `.forge/<asset>/`: `brief.md`, `forge.json` (its source, field, size, fit stages and chain), `prompts.json`, `reference/` (mesh, `samples.bin`), `work/` (the fitter's copy), `judge/`, `fits/`, `scores.json` |
| The vocabulary | `lib/kit/shapes/` (`outline`, `making`, `carve`, `ornament`, `wear`, `made`, `craft`) and `lib/kit/substances/` (`wood`, `leather`, `brass`, `paper`) |
| The judge's and fitter's wrela | `lib/engine/judge/samples.wrela` |
| The first gold assets | `game/world/table.wrela`, `game/world/grimoire.wrela`; SCENE 13 shows them |

## The vocabulary comes first

Today's primitives can't express carving, however good a model is, so the vocabulary is built
first and a model comes last and has to earn its place. The ladder, and what of it is built:

1. **Primitives and blends** (std::field): distances, unions, smooth unions, offsets.
2. **Making operations** (`kit::shapes::making`), each a few numbers a joiner would name, exact or
   a bound, Lipschitz 1: `turned` (a lathe's profile), `moulded` (a slab whose edge is run with a
   moulding), `banded` (a section run round a plan: an apron, a frame), `extruded` (a rail),
   `board` (eased edges), `joined` (two parts with their seam showing), `folded` (one part at the
   corners of a symmetric whole), `placed`, `grown` (leather over a board), `sheathed` (a sheet of
   brass over a corner), `sweep` (`kit::shapes::sweep`: a section carried along a cubic Bézier,
   its frame turning as little as the path lets it: an arched strap, a tendril, a curved handle;
   24 mitered prisms of the section, their union exact as a bound and of steepness 1, held to the
   curve by a test, a strap of a 60 cm arc within half a millimetre). Profiles and sections are **outlines** (`kit::shapes::outline`): points
   joined by segments, each straight or bowed into an arc by its bulge, so a bead is one number,
   a cove one, an ogee two, and a turned leg is fourteen points where a polygon would need a
   hundred.
3. **Processes**: the board's grain comes from where it lay in its log (`kit::substances::wood`:
   `Log`, `grain`); a cut by a gouge has the gouge's cross-section and shows the grain inside at the
   angle it cuts (`kit::shapes::carve`: `groove`); wear comes from the shape's own story
   (`kit::shapes::made::Story`: a convex edge is exposed and rubbed, a cove or a carving's ground
   is recessed and dirty; brass is polished where exposed and tarnished in hollows). The shape and
   its look come from the same numbers, so they agree. Use and the hand leave their own marks
   (`kit::shapes::wear`): `dinged` (shallow dents, recessed in the story so dirt and wax sit in
   them), `cupped` (a board moved with the seasons); a carving's gouge leaves facets on what stands
   (`carved`'s `tooling`) and a carver's hand makes each of a row a little different (`row`'s
   `wander`); oak's rings differ year to year and its surface carries scratches, grime and a worn
   finish, leather crazes, scuffs and wears through at its edges, paper stains, brass smudges.
   Every mark keeps a part's bound near 1 (a test holds each gold asset to 1.05, the engine's
   steepness): a mark that would step a surface (a book's uneven gatherings, set back in the
   shape, made the grimoire's bound 8.8) belongs to its material.
4. **Ornament rules** (`kit::shapes::ornament`): motifs built from blades (lens-shaped lobes on a
   bowed spine) and grooves (U-gouge cuts) — `acanthus`, `scroll`, `rosette`, `bead_and_reel`,
   `egg_and_dart`, `flute`, `lobe` — and the rules they repeat by: `mirrored`, `row` (along a path,
   or round a turned part without end), `ring` (about a centre), `set`, `pair`. A carving lowers a
   ground over a region of a face (flat, round or draped) and leaves the motif standing.
5. **Style**: gold examples and rules, not a function. Not built: two gold assets so far.

Every new field kind has a CPU reference it is held to (`lib/kit/shapes/*_test.wrela`): outlines
against their 2000-points-a-segment polylines, turned and moulded forms against clouds of up to
1,049,220 points of their own surfaces (worst 0.07 and 0.14 mm), carvings against a 160,000-point
grid of their carved heights (never over the brute force: a bound), each motif's stated steepness
against its gradient measured by finite differences, and std's derived-gradient Lipschitz check.

Every number in an asset belongs to a named part and means something physical. No neural
correction terms: a `.wrela` file may hold hundreds to a few thousand numbers if the detail needs
them, and none is a weight.

### What the engine had to learn

- **Culling from the bound.** Made shapes answer the mesher's culling (`Surface::pruned`,
  `interval_near`) from their distance at a brick's centre and their Lipschitz bound
  (`made::stepped`), not by deriving intervals through every blade: the derived form made the
  carved table's `cull_bricks` shader 351 KiB, past Metal's 288 KiB.
- **Repeats read one cell.** A row or ring reads the motif in the cell a point is in, alone (motifs
  keep within their cells); reading the next cell too made the table's `refine_creases` 377 KiB.
- **Crease slices sized by work.** `refine_creases` was dispatched in fixed slices of 65,536 cells;
  the carved table's 319,294 work a cell made a slice 52 ms on the M4, over the guard's 50, and the
  host refused it. The slice is now the compiler's count of a cell's work for the field extracted
  (`std::derive::work`, `extract.wrela`'s `crease_slice_of`) against the GPU's step: the same for
  every field that fitted before, smaller for a heavy one, the same picture.
- **A gradient on the surface.** An outline's distance was `sqrt(d²)`, which has no gradient on
  the outline itself: points the judge projected onto a surface, and vertices the mesher snaps onto
  it, read a normal of (0, 1, 0). The brute-force judge found it. The distance is now the nearest
  segment's signed offset where the nearest point is inside a segment.

- **The audit marches in passes.** The hole audit's own reference march (`judge::audit`,
  `marching`) took its 512 steps in one invocation: 5.8M work for the table, 120 ms, which the
  host refused before it ran. It now marches in passes of 512, 128, 32 or 8 steps (a pipeline
  constant, so the compiler bounds each), chosen from the compiler's count of a step's work, each
  ray's march kept between passes. The same holes for every kind the audit held before.
- **No fixed cell floor.** A thing's cells were never finer than 5 mm, beside the batch's own
  budget floor (`size_floor`: its bound's radius over 180). A book or a candlestick at 60 cm got
  5 mm cells (facets 15 to 30 px wide); now they get what the budget allows, about a millimetre.
- **A pixel asks only its part, and only what changes from pixel to pixel.** Every pixel of the
  table asked the whole table's shape (to tell its story: exposed, recessed, which piece) and
  every part's material, about four fifths of what a pixel cost. Now a field says which part
  decides a point and what slow-changing numbers it keeps there (`Field::kept_at`: for a made
  shape, its story), the mesher works them out once a vertex (`keep_verts`) and the vertex carries
  them, and a pixel asks only that part's material with them (`Field::channels_kept`). A union
  routes by part, so a pixel never asks a material that isn't its own. A material may use what is
  kept smoothly; a sharp edge in it alone draws the mesh's triangles, so a sharp change (leather
  worn through) is broken by noise at the pixel.
- **No vertex kernel asks a material.** Making a vertex asked its field's whole material only to
  learn whether it lets light through (which triangles it goes with), so every kernel that made
  vertices carried the oak's grain, scratches and grime: the worn table's `refine_creases` came
  to 348 KiB. Now `keep_verts`, run on each chunk's vertices before its triangles are made, is
  the one kernel that asks a material, and the vertex's kept word carries the answer
  (`KEPT_TRANS`); the skinned path does the same (`keep_rest_verts`).
- **The finer cull asks what the coarse one does.** A steep thing's bricks are checked again in
  octants, and that check derived its range by interval arithmetic through the whole field (the
  dented table's `cull_bricks` 381 KiB); it now asks the same range the brick's own check does
  (`Surface::pruned`), which a made shape answers from its bound and a union part by part.

- **Physical material bump & pixel-rate carving detail (`Substance::bump`, `Shaped::detail_at`).**
  Features finer than a mesh cell (oak's irregular earlywood pores, longitudinal cellulose fibres,
  hand-planed plank seams, drawbore oak dowel pegs and gouge facets; leather's supple hide ripples,
  pebbles, crazing and blind-tooled grooves; paper's stacked deckled leaf edges; brass hammering
  and corrosion pits) return a physical height offset in metres (`Substance::bump`), tilted into
  the shading normal at pixel rate via Mikkelsen (2010) screen-space surface gradients in
  `lib/engine/mesh/draw.wrela::apply_micro_relief` with footprint attenuation and view-hemisphere
  safety. `Shaped::detail_at` routes by `part` so carved faces (`Carved::detail_at`) evaluate their
  exact motif at pixel resolution inside `region` — sharpening crevice `Story` darkening, cut
  depth, and fine groove/vein normals without raising the global mesh budget.
- **Sculpted C¹ carving, circular lunettes & seamless round repeats.** `Blade` uses a C¹
  `(1 - x²)^(3/2)` cushion profile (cutting its slope bound 2.5× so carvings mesh cleanly at
  `r / 180` without faceted spikes), `Scroll` is a self-contained semicircular Jacobean fluted
  lunette-and-palmette arcade (true circular arch moulding, springpoint bosses, upper spandrel
  jewels, calyx heart and radiating 3-petal palmette fan), `Rosette` pairs veined petals with a
  circular domed `Button`, `Row`'s hand `wander` tapers to zero at cell edges (`cos(π s / period)`)
  and wraps cell IDs modulo `wrap` on `RoundFace` so turned carvings never tear across `atan2 = ±π`,
  and `resolve_normal_from` relaxes its smooth-normal threshold (`0.85` to `0.65`) so curved relief
  stays smooth while crisp creases stay sharp.
- **Natural matte-satin antique materials.** `Oak` and `Leather` drop mirror clearcoat (`coat = 0`)
  and drive authentic beeswax and hand-rubbed calfskin sheen strictly through GGX roughness, paired
  with turbulent multi-octave oak grain, non-lattice pores, floor-proximity dust, antique cordovan hide
  with restrained gold-leaf traces in blind-tooled recesses, warm cast bronze-brass, and aged
  thumb-worn rag paper.
- **Fluent, unit-suffixed asset vocabulary (`kit::shapes::craft`).** Gold assets compose parts
  left-to-right with `.at(...)`, `.carve_round(...)`, `.joined_to(...)`, `.named_piece(...)` and
  `.in_material(...)`, pair profile points with their bows (`pt(vec2(...), bow: ...)`), write
  physical measures with Wrela SI suffixes (`cm`, `mm`, `deg`), and return opaque `Craft` /
  `Asset` trait types with zero nested type-alias boilerplate.
- **Spatial judge diagnostics.** `tools/forge judge` reports `by_part` and `by_height` error
  breakdowns (mean and 95th-percentile mm per named part and across 5 vertical bands) alongside
  overall scores, and handles SI unit literals in literal counts and scrambling.

Gaps filed (an author needed this, for that):

- **A thing still has one uniform cell size.** `size_floor = r / 180` keeps all existing scenes
  within the 8,192-brick batch budget; view-adaptive per-brick LOD (bricks nearer the eye finer,
  with seams kept closed) remains the general engine fix for even finer hero-distance carving.
- ~~A scene's kinds of thing share one shader~~: closed 2026-11-24 by taking away the shader
  size budget (Metal's 288 KiB of MSL): a pipeline is held to its weight and kept bytes alone, so
  the SDF clipmap's one splat may hold as many kinds of thing as a world has.
- **Work a segment needs once is asked at every pixel.** An outline's arcs (their centres and
  radii) depend only on the asset's numbers, yet the shading pass computed them at every pixel:
  the compiler refused the candlestick's 22 of them (E0712, 304 work). Outlines now work each
  segment out where they are built (`outline::arc_of`), which the fitter still differentiates.
- ~~The engine's micro relief (`Substance::micro`) has one fixed world scale~~: closed for
  material-authored scale via `Substance::bump` (above).
- ~~Channels are evaluated at a pixel's centre with no footprint~~: closed, the engine filters a
  thing's channels over the pixel's footprint (`mesh::proof::filter_channels`, `e86f5c0`), so
  `lib/kit` still need not ask for one (tools/layers rule 3).
- ~~Union's channels ask every part's material~~: closed, a pixel asks only its part's
  (`Field::channels_kept`, above). The assets still put one oak over all a table's pieces
  (`Story::piece` picks each piece's log), as a joiner works one stock.
- **A thing with one dent is steep everywhere.** The engine knows one Lipschitz bound a thing, so
  a mark's slope anywhere sets how finely the whole thing is meshed and culled: dents had to be
  made gentle (1 in 40) to keep the gold set's bounds under 1.05. Bounds by region, as cells by
  region (above), would let a carving be steep where it is carved alone.

## The judge

Turns "does this look like the concept" into numbers (`tools/forge judge`, its wrela in
`lib/engine/judge/samples.wrela`):

- **Shape**, against the reference's samples: from each reference surface sample to the field's
  surface (projected along the gradient), and from the field's surface (a grid projected) to the
  reference's (to the plane of the nearest sample, as far as half the samples' spacing reaches),
  each as a mean and a 95th percentile in mm; the normals' agreement (mean angle, share within
  10°); the share of volume samples on their own side; and outlines from the concept views'
  cameras (front, side, top, three-quarter: the field traced, the reference splatted, a pixel's
  disagreement forgiven).
- **Look** (`tools/forge look`): a render against a concept image of the same view, the render's
  linear light scaled to the concept's over the object, then the mean colour difference (ΔE, CIE76)
  and SSIM of lightness in 8-pixel windows over the object.
- **Readability**: the asset's float literals, its named parts (`parts()`, `part_name`), and the
  literals no fit ever moved (`wrela solve`'s `changes`: dead structure).
- **Cost**: the asset's frame time in SCENE 13, on the M4 and the Radeon (`tools/gpu --timestamps`).

Checked against brute force (`tools/forge-test`, 4): the fast path finds the field's surface by
projection and the nearest sample through a grid; the brute force finds it by sign changes on a
grid edge (24 halvings, no gradient), compares every pair, and takes normals by central
differences. On the true leg against its own 27,793 samples: to the field 0.000 / 0.000 mm (fast)
and 0.030 / 0.184 mm (brute, its grid's own limit); to the reference 0.012 / 0.024 and
0.009 / 0.022 mm; every volume sample on its side, both ways; every outline 1.0.

Ranks a degraded asset below a good one on every count it has without a GPU (`tools/forge-test`,
6): the leg written with straight pieces, twice the numbers (58 to 36), nothing named and a number
nothing reads is worse on both distances, normals, outlines, literals, named parts and literals
never moved. The look and cost counts need a GPU and concept images; the look's mechanics are
checked on pictures (itself: ΔE 0, SSIM 1; darkened and matched: ΔE 0.00; blurred and shifted:
ΔE 11.2, SSIM 0.10).

## The fitter

`tools/forge fit` runs `wrela solve --fit fit::residuals` (Levenberg–Marquardt on
`std::lift::jacobian`, language.md §23.9) over the work copy's named literals, stage by stage
(`forge.json`'s `stages`: proportions, then parts, then ornament), then `wrela edit` writes the
numbers back in the form they were written. The misses (`samples::misses`) are the field's value
at the stage's surface samples, and for volume samples that fill the reference's box, how far the
field puts each on the wrong side and, near the surface, how far its value is from the
reference's distance: without the box's samples, a fit satisfied the surface samples by growing
surface the reference doesn't have (measured: the leg's top went to a self-crossing profile, 30 mm
mean from the reference). `--starts K` tries K starting values for a stage's first literal (a
motif's phase) and keeps the best. Everything runs on the CPU: the 50 ms rule is the GPU's.

Done when a known program with its numbers scrambled is recovered to within the judge's tolerance
(95th percentile to the field ≤ 0.1 mm, to the reference ≤ 0.25 mm, normals ≤ 2° mean). The leg's
42 numbers moved by up to 4 mm each way (three seeds) come back in 7 to 13 steps and 1.3 to 2.4 s:
to the field 0.002 to 0.025 mm mean, normals 0.15° to 0.87°, outlines 1.0. Moved by up to 10% of
themselves instead, they don't: 10% of a height near 0.6 m is 6 cm, which swaps the collar's
points, so the scrambled profile crosses itself before the fit starts; a structure that wrong is
the outline's author's to fix, not the fitter's. That is a kill line's worth of evidence that
discrete structure stays the agent's and the numbers the fitter's.

What the fitter doesn't yet do, found by the authors in the trial below: numbers the samples
barely see drift (a bulge on a segment hidden under the seat went to 10 mm; a corner's radius went
below 0), because a least-squares step moves what the misses don't hold; and a whole profile freed
at once can cross itself. Both authors worked round it by naming one constant per physical measure
and freeing a few at a time. The fix belongs in `Minimize::fit`: a pull toward each literal's
starting value, light enough that what the samples see overrides it, and a check that an outline
keeps its order.

A trap to know: a number inside a constant built from other constants (`[vec2(FOOT_R, 0.0), ...]`)
is computed when the program is built, so it isn't lifted (language.md §23.1) and the fitter can't
move it; `fit` reports it as never moved. Write the array's numbers as literals, or build it in a
function.

## The authoring trial (phase 5's kill line, run once)

Four fresh authors (Claude Sonnet 5.5, the `wrela-author-sonnet` setup: a sealed workspace, a
brief, a budget of 120 tool calls) each wrote a joint stool against the same hidden program's
samples (118,051 surface, 70,970 volume): two with `docs/forge.md` and the kit alone, two with the
four gold assets beside them as examples. Every author reached the brief's bar (both 95th
percentiles under 1 mm). The 95th percentiles, both ways, mm; the normals' mean angle; the
literals and those never moved; the tool calls and minutes:

| Author | Reference to field | Field to reference | Normals | Literals (never moved) | Calls, minutes |
|---|---|---|---|---|---|
| docs 1 | 0.877 | 0.810 | 4.27° | 74 (2) | 32, 5.1 |
| docs 2 | 0.666 | 0.690 | 4.77° | 94 (88) | 84, 20.1 |
| gold 1 | 0.187 | 0.291 | 2.95° | 138 (3) | 24, 4.2 |
| gold 2 | 0.294 | 0.410 | 1.86° | 121 (4) | 46, 7.6 |

The gold examples did not fail the kill line: with them both authors were closer on every shape
count (about a third of the docs-only miss) and quicker. Two runs an arm show a direction, not a
measurement. What the authors asked for, each its next piece of work: the judge says where the
misses are (by part, by height), not one number; the fitter holds numbers the samples barely see
(above); and the brief's prose was rougher than the reference (both gold authors read the samples
to find that the stretchers were boards, not turnings).

## Provenance

Every step the forge runs is kept in the asset's chain (`forge.json`) with its command line, and
`tools/forge record` writes the record: the brief, each prompt, each concept image's model,
version and seed, the reference's model, version, inputs, alignment and hashes, each fit's stages,
losses and literals never moved, and the judge's scores. A gold asset's record is beside its source
(`<file>.provenance.json`); a trial's is in its folder. `tools/forge replay` runs the chain again
in a scratch asset and compares each step's hashes and scores: the leg's chain (synthetic
reference, scramble, fit, judge) replays with every step the same in 4.6 s.

## The showcase: a carved oak table and a grimoire

`game/world/table.wrela`: a library table 1.2 by 0.75 m, 76 cm high: a planked breadboard-end top
with drawbore oak dowel pegs and an ovolo-over-cove edge, a frieze carved with an 8-petal rosette
between semicircular Jacobean lunette-and-palmette arches set back 3 mm from the legs' die-blocks
(each carrying a carved corner rosette medallion), four turned baluster legs (gadrooned bulbs,
acanthus round their necks, fluted columns) joined to the frieze with shoulder lines showing, and
an H-stretcher of turned rails. One oak over four pieces, each from its own log (the top and frieze
plain-sawn, the legs and rails quarter-sawn). Four named parts: top, frieze, legs, stretcher.

`game/world/grimoire.wrela`: a book 26 by 19 by 7 cm: dark cordovan leather over wooden boards,
blind- and gold-tooled with a border, a central lozenge and French-groove hinge crease, a rounded
spine with four tucked raised cord bands, a hollowed fore-edge with thumb-worn deckled gatherings,
riveted L-bracket brass corner mounts, two hinged brass fore-edge clasps, and a turned brass boss
carved with an 8-petal rosette. Six named parts: covers, spine, pages, corners, clasps, boss.

SCENE 13 shows them on the stage (`VIEW` 0 the room, 1 the carving at 60 cm, 2 the book at 60 cm).
Neither has a reference yet: the concept views and the TRELLIS.2 reference are made on Colab
(`tools/forge-notebooks/`), which needs a person to run. Until they are, the assets are outlines
written from their briefs, judged for readability and cost, not for likeness.

## Risks and licensing

Not legal advice: the points to know before shipping.

1. **Image ownership.** The concept images never ship. Each asset records its prompt, model,
   version and seed.
2. **Reference meshes.** Whether a model's output inherits its training data's terms is unsettled.
   The reference never ships, and the program is written and fitted on its own. TRELLIS.2's code
   and weights are MIT, but its README names dependencies (nvdiffrast, nvdiffrec) under NVIDIA's
   Source Code License, which is for research or evaluation: running it to make a reference that
   never ships is the use to check against that licence before a commercial release. The clean
   chain removes the question.
3. **Hunyuan3D is out**: its licence excludes the EU, the UK and South Korea.
4. **Prompts** never name another game, film, book, character or living artist, and never say "in
   the style of": `tools/forge lint` refuses them, and `tools/forge prompts` won't write one that
   fails.
5. **Storefront disclosure**: Steam asks for player-facing content made with AI help; these assets
   count.
6. **Copyright in the shipped code**: the same question as the whole repository's.
7. **Training stays clean** if it uses our own synthetic grammar data and an Apache- or
   MIT-licensed base model, and never trains on assets with non-commercial terms.

**The clean chain**: drop the reference and fit straight to the concept views through the judge's
outlines and look. Slower and less certain, but every step is ours.

## Phases, and where each stands

| Phase | Done when | Where it stands |
|---|---|---|
| 0. Workspace and provenance | One asset's chain replays from its record | Done: the leg's chain replays, every step the same |
| 1. Vocabulary | Kill if a carved motif can't hold the judge's error at the hero distance with fewer numbers than plain primitives | Built and checked against CPU references; the kill line needs a reference of real carving to measure, not run yet |
| 2. Judge | Ranks a degraded asset below on every count; shape agrees with brute force | Shape and readability done (`tools/forge-test`); look checked on pictures; look and cost against a degraded asset need GPU renders of both |
| 3. Fitter | A scrambled known program recovered to tolerance | Done for a 42-number profile; the whole table's 246 numbers in stages not yet |
| 4. Showcase | Agent + fitter reach the judge's bar against a reference | Assets written and shown; their reference is the Colab step |
| 5. Gold set, in-context authoring | Kill if gold examples don't raise a fresh author's scores | Gold set written (table, grimoire, candlestick, shelf, chair; SCENE 14); the trial ran once, two authors an arm: gold examples raised every shape score (not killed) |
| 6. Synthetic data, trained model | Beats agent + fitter on held-out assets | Not started; gated on 4 and 5 |
