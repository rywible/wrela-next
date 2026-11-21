# Design: matter, a look, an eye

This is what the engine is for and how it is cut. It was agreed with Ryan on 2026-11-18, after a
spike (branch `spike/volume`; its log is `wrela-notes/volume/NOTES.md`) measured which ideas hold.
Where a claim here was measured, it says so and gives the number. Where it was not, it says that
too. Change this file in the same change that makes it wrong.

## What it is for

Agents author the whole game. A game is content and rules; the engine makes it good. An author
writes what is in the world and never how it is drawn, and what they write is realised
beautifully, cohesively and within the frame's budget with no renderer code of their own.

"Unified" is asked of three places, and only these:

- **What an author writes.** One obvious thing to write, and never a choice of renderer.
- **The light.** Everything is lit by, and lights, everything else through one store.
- **The picture.** One place where light becomes an image.

It is not asked of the engine's inner loops. Inside, the engine may use as many strategies as it
likes, provided no author ever picks one or tunes one. (Measured: one inner loop for everything
is false. A stone stack drawn by asking its own function at every pixel cost 19.2 ms a frame at
1080p on the M4; the same path answered from cells cost 2.56 ms.)

## What an author writes

Everything in the world is **matter at a point**. The author says what is there and what it is
made of. The engine decides how to see it.

| The author gives | Which says |
| --- | --- |
| **Matter** | where it is (how far from here, or how dense here) and what it is made of |
| **A look** | how light becomes the picture: tone, ink, grade |
| **An eye** | where the picture is taken from |

- A solid is matter with a hard edge. Fog is matter with a density. A glow is matter that emits.
  Whether light stops at it or passes through it is a fact about what it is made of, said once,
  there.
- **A world is things, and a thing is what moves as one.** The author hands the engine a world of
  things (a stage, a body, a stack). What counts as one thing is a fact about the world, as a
  substance is: it is not a way to tune cost, and splitting or joining things must not show in
  the picture or the frame's time.
- **The sky** is matter with a direction and no place. It is the one named exception, because
  hiding it would obscure more than it would simplify.
- **The look owns tone, alone.** A substance says how matter answers light, physically. Only the
  look says how light becomes a tone on the screen. Style enters nowhere else.
- **The sky owns all light from where there is no matter**, below the horizon as above it. **The
  eye's field of view is the author's.**
- **Matter never sees how it is viewed.** It may depend on a point, on time and on the game's
  state, and on nothing else: not the eye, not the screen, not the frame. Only the look sees
  light, and nothing in the game sees a pixel. Almost every workaround needs to know how a thing
  is being looked at, so this is what makes workarounds hard to write.

### What an author never states

Bounds, steepness, filtering, how finely anything is sampled, cell sizes, rates, tolerances,
caches, how a surface turns from one pixel to the next. Each of these is derived by the toolchain
(it compiles the field: it can derive a gradient, a bound, an interval, and what an ask costs)
or decided by the engine against its budget. An author who has to state one has found an engine
bug.

### Cost is the engine's, not the author's

An author writes what they mean, in the plainest way, and it comes out right and fast, as code
does from a good compiler. How a thing is phrased should not show in the picture or in the
frame's time, except in the few cases no engine could hide.

**The safe default.** Every field is realised from cells the engine keeps (a cache of it). An
author cannot write a slow thing. Where the toolchain's cost for a field is low, the engine asks
the field itself. The engine chooses; no author does.

**The choice may never show.** Two rules follow, both paid for in the first slice.

- **Either answer draws the same picture**, to the judges' lines. Cells are a cache of the field,
  not a second version of it: where cells answer, they are as fine as the eye asks or as the
  field needs, whichever is coarser. A brick is split only where a pixel reads it finer and its
  cells are off their field by more than a tenth of that pixel. (Measured: that holds a stone
  stack from 100 m to 3 m in 10 to 16 bytes a screen pixel; cells a pixel wide everywhere draw
  the same picture in 250 to 310.) (Measured: cells three pixels wide lost 72 to 79 of a creature's 85 thin pixels, a 1 cm
  wand and the tips of its ears, and opened holes.) Until that holds, the choice changes what is
  drawn, and nothing may choose it.
- **The choice is the same on every run and every machine.** It is made from what the toolchain
  counts or measures an ask of the field to cost, fixed when the game is built (or once for a
  machine, and kept), never from timing while playing.
  (Measured: an engine that chose by timing its own frames chose differently for the same build
  with the GPU in a different state, and so drew a different picture.)

**Three things make the difference between the two invisible.** The first is now a requirement
(above); the other two are not measured yet. Each is a line in the author test below.

- **The engine sizes its cells from what the eye sees.** Cells about a pixel wide near the eye
  lose only detail that could not be seen. (The spike's rounded edges came from cells five pixels
  wide, a size fixed by hand.)
- **The compiler finds the structure.** Which of a thing's parts matter near a place is derived
  from a plain union, by the interval arithmetic the toolchain already has; an author does not
  write it. (By hand in the spike, it took a body's deposit from 4.5 ms to 0.73 ms.)
- **The two answers are mixed by budget, not by how a thing was written.** Cells bring a ray
  near; the field itself is asked for as many pixels as the frame can afford, nearest first.
  Quality then varies with load, never with phrasing.

**What stays visible is the egregious case**, as with any compiler: a thing that is dear, moving
and close all at once; a thing thinner than a cell, as fur is. For those the build says so
plainly, with the number and what would cure it.

### When the engine cannot do what an author needs

The author stops and files the gap: "I needed this, for that." They do not work round it. A gap
filed is the engine's next piece of work; a workaround is a special case nobody can see. This has
to be the easy path, so the author's tools must make filing a gap one step.

## What it must run on

The game ships on Steam, so the bar is what most Steam players have. In October 2026 that was an
RTX 3060 class card, 16 GB of memory and a 1080p screen (citations.md). The aim is to win on
quality and at least match comparable games on speed.

| Tier | Hardware | What it must hold |
| --- | --- | --- |
| **The bar** | RTX 3060 class | 1080p at 60 frames a second, full quality |
| **Stretch** | the same card | 1440p at 60, upscaled from a lower rate |
| **Must run** | Steam Deck or a laptop chip | 30 to 40 frames a second, upscaled from a lower rate |

A tier is a budget the engine fits itself to (how finely it samples, how much light it asks for
a frame, what rate it draws at before upscaling). It is never something an author tunes.

**How it is measured, with the machines there are.** Nobody here owns the bar's card yet.

- **The M4 Air is where absolute numbers are held.** It is weaker than the bar's card (by two to
  three times, an estimate, not a measurement), so the working line is the bar's own target on
  the weaker machine: **1080p at 60 frames a second on the M4 Air**. What holds there holds on
  the bar's card with room, if the estimate is right. It has no fan: a number is taken after the
  machine has warmed, with the load beside the frame that keeps its clock up, and a long run
  that slows is reported as such.
- **The Radeon laptop chip is for Vulkan, and for deltas.** It proves the Vulkan path draws the
  same picture, and a change is held to its own number before the change there, not to a line.
  Its absolute times are not a bar. (The two machines do not move together: one frame that cost
  twice another on the M4 tied it on the Radeon. So a delta there is evidence about that change
  on that machine, and no more.)
- **Not measured by either:** the bar's card itself, and whether the must-run tier holds on a
  Steam Deck. The M4 is stronger than a Deck, so it does not show that. A Windows machine with
  the bar's card is wanted, and every number in this document is provisional until it exists.

## The frame's budget, and two kinds of line

A frame at the bar is 16.6 ms: 1080p at 60 frames a second on the M4 Air, for a whole scene of
the game, not a test scene. It is shared out once, here, so that a slice's targets come from a
share and not from a round number somebody liked.

| Share | ms | What it pays for |
| --- | --- | --- |
| The eye's rays | 3.0 | what every pixel meets first, for the whole world |
| Light | 3.0 | keeping stored light, and reading it where a pixel is shaded |
| Matter light passes through | 2.0 | fog, fire, smoke |
| Edges | 1.5 | a pixel as the mean of what the look shows over it |
| Things that move | 1.5 | whatever is found or deposited again each frame |
| Cells filled as the eye moves | 1.0 | |
| The look | 1.0 | tone, bloom, the screen |
| Held back | 3.6 | what is not built (the sea, fur, text), and a frame's own variation |

The shares are a first proposal from what is measured and will move; they move here, said
plainly, never inside a slice. A test scene of two or three things should take a small part of
each share, and a slice says what part before it measures. Until many things are measured
together, nobody knows how a share fills as a world grows: that probe sets the first real
targets.

**Two kinds of line, and they are not the same.**

- **A kill line ends an idea.** It asks whether a way of doing something pays at all, it is
  relative (halve this; save a third of that), it is written before measuring, and when it fires
  the idea leaves in the same change. Nobody goes round one.
- **A target is a share of the budget.** Missing it does not end anything and needs nobody's
  decision: it stays open, it is reported again each time, and the report says how far over its
  share the engine is and which spans hold the difference.

A report keeps the two apart, and leads with correctness, which is neither: a picture that is
wrong is a fault, whatever it costs.

**When a kill line asked only for an amount.** A change is kept, its line recorded as missed and
left as an open target, when all of these hold: every judged measure and the audit are the same
as before within their run-to-run spread; it is cheaper on every scene and count measured, on
both machines; and it repeats to the bit. (Three such on one night: shadows swept from the sun,
the sun's map for moving things, the sky's maps: each several times cheaper with the same
picture, each short of the figure its own line named.) In every other case the line stands as
written: anything judged that got worse, anything dearer somewhere, or an idea whose machinery
only its amount would justify, which is said when the line is written.

## The frame costs by the pixel (2026-11-19: the engine's direction; measured in parts, not built)

**Why.** Every pass of the frame as built is "for each thing, draw its part of the screen with
its own program": the eye's, the outlines', the edge rays', the shadows', the sun's map, a map of
sky a moving body. So every pass grows with things. Measured on the M4 at 1080p: a stage, 256
boxes and one creature 12.1 ms; with 64 creatures 50; with 256, 177 (the eye 110, edge work 50,
light and the rest 17). Where a pass already costs by the pixel the constants are fine: the whole
screen from sampled cells is 0.9 ms, and light for the whole screen 1.6. Edges are over half of
every frame, the still world's included (3.3 of 6.4 ms).

**The rules.** They are the owner's, and every stage below is held to them.

1. **Every pass costs by the pixel**, with a constant the toolchain can count. No pass draws a
   thing again after the eye has found it.
2. **No mechanism is chosen by what kind of thing is there.** Still or moving, a creature or
   rock, kept in cells or written in closed form: every point of a surface asks the same
   questions the same way, and every piece of matter answers them the same way. This is "the
   engine knows properties, never things" applied to the frame itself.
3. **One way to see, one way to light.** A cache is allowed only as a cache of the one answer,
   filled from all matter alike, and it is judged against that answer.
4. **Cohesion is measured.** The same field drawn once as still matter and once as a thing that
   moves but does not happen to move gives the same picture. A moving thing's contact with still
   matter is judged against brute force. Both are judges with lines, like any other.
5. **The frame is summed before it is drawn.** What a pass will cost is its pixels times counted
   work times the device's measured time a unit of work. Over the budget, the engine lowers a
   rate that has a physical meaning and that every scene gets alike, never the frame's time.

**The pieces.**

- **Stand-ins.** Between the game's fields and every pass sits one store. For each thing it holds
  the cheapest form of it that is proved good enough for the size the thing has on the screen
  now, and a form is one of a closed list of families: a round cone (a sphere is its end case),
  a box, or cells. So a thing is its closed-form parts where they are of those families; fewer
  such parts, where a bound on the distance between the two surfaces is under what the eye is
  held to; or cells kept once a kind in the kind's own rigid frame, at the fineness the nearest
  of that kind needs, shared by every thing of the kind (a joint is the smooth join of two such
  reads, which is within the cells' own error of the true join because the join is monotone and
  steepness one in each part). A shape outside the list (an ellipsoid, a displaced thing) is
  kept in cells or bounded by the list's shapes with a proved error; the list does not grow by
  a thing, or the kernel below becomes the shader that holds every answer. With a stand-in come two
  hulls, one inside the surface and one outside, both proved, its bound, and its counted work.
  Matter that changes makes again only what its change's bound touches, to a fixed amount a
  frame, the coarser level standing in meanwhile. Passes and the CPU's questions ask the store,
  never the field.
- **One way to see: one buffer of what was met, and two fillers of it.** Closed-form stand-ins
  are listed by screen tile, nearest first. One kernel walks a pixel's list: a stand-in's hulls
  are met in closed form, so the surface is bracketed between them and found by a guarded
  root-finder (the inner hull's entry is asked first: away from a join it is the hit). A ray
  that passes within a pixel of a stand-in without meeting it leaves how near it came and which
  way the outline runs. The kernel writes what was met and how far, and that edge record;
  nothing else. Cells keep the rasteriser. That is the one exception to "one way", it was
  measured (below), and it holds only under these conditions, each of them checked: both
  fillers write the same ray's distance; the edge record is shared across the two; and nothing
  after the eye's pass can tell which filler found a pixel. The cohesion judges are what prove
  it.
- **What a pixel shows** (facing, substance) is found once a pixel after the eye's pass, by a
  program a kind run over the pixels where that kind was met. A thing's substance stays its own
  code, and a pixel runs only what its thing can use, as this document already requires.
- **One light.** The question is "what light arrives at this point through this cone", answered
  by one march through one index of every stand-in and everything that glows, the level read
  following the cone's width. The sky, bounce, a spell's light, contact and a mirror's reflection
  are that question with different cones. Two caches of it: the sun's cone, too thin to march at
  every pixel, in one map filled from all matter alike and read with a blur that widens with the
  blocker's distance; and the sky's and bounce's cones stored on surfaces, one store for still
  and moving surfaces (a still surface is a moving one whose frame does not move), filled by the
  march to a fixed amount a frame.
- **Edges: today's two samples, found without a march through a draw a thing.** A pixel an
  outline crosses is covered in closed form from its record (a line, an arc's correction, a clip
  where two cross), as today. Its colours are NOT read from neighbouring pixels: that was
  measured and it loses (below). They are today's two samples at today's places, each found by a
  short solve started from a neighbouring pixel's hit, in a pass that costs by the pixel, and
  lit by the one light. A crease between two faces of an exact thing is a line from the two
  facings, with no ray. A part thinner than a pixel is covered in closed form. No outline pass,
  and no pass that draws a thing again to find an edge's sample or its shadow.

**What it is predicted to cost** (M4, 1080p, ms; from the engine's measured unit costs and
estimates; nothing here is measured as a frame), against the shares above:

| Share | Share | Predicted, 256 creatures and 256 props | Of it |
| --- | --- | --- | --- |
| The eye's rays | 3.0 | 6.0 | the kernel 3.98 (measured alone, with a rim that is sound), cells behind it 0.9 (measured), what a pixel shows 1.1 |
| Light | 3.0 | 3.55 | the sun's map 0.55; the sun read, stored light, cones a pixel 3.0 |
| Edges | 1.5 | 1.5 | an estimate: two short solves and two shadings an edge pixel |
| Things that move, and cells filled | 2.5 | 0.85 | stand-ins, lists, light's index (the lists alone took 1.1 ms of CPU in the kernel's measurement) |
| Matter light passes through; the look | 3.0 | not designed here | |
| Sum of what is designed | | 11.9 | of which only the kernel and the cells are measured |

The shares stay as they are until a stage is measured against them. With the two shares not
designed here the frame at 256 creatures is about 14.9 of 16.6 ms, and that is with the kernel
as it measures alone on a scene laid out kindly: on the engine's own crowd it is dearer still
(below). The eye's share is over by 3, and the design does not hold at 256 until the kernel's
searches at joins and its lists are cut.

**What was measured the night it was written** (2026-11-19; M4 at 1080p unless said).

- *The kernel, alone, outside the engine* (`wrela-notes/next/scratch/tilekernel/`): 256 creatures
  of 18 round cones 1.49 ms; 64, 0.99; the eye three times nearer 1.58. No holes against a
  brute-force march on either machine. The Radeon: 6.60 at 256. Its lines were 2.2 for the
  budget to be real, 3.0 for doubt, 4.5 for death. As first written it took 2.98 to 3.03: the
  cut that mattered was not mathematics but the fold's loop indexing a small vector by its
  variable, which sent it to memory on both GPUs (unrolled: 2.99 to 1.72 on the M4, 16.2 to 7.9
  on the Radeon); then one ask for a near miss over a pixel away, and the root found to a
  twentieth of a pixel. It asks 2.74 times a hit pixel. It is a fragment pass over the screen:
  the guard refuses a compute walk whose length depends on data. THAT 1.49 WAS NOT SOUND: where
  a ray grazes a join it fitted a near miss's least by a parabola, and on the engine's real
  poses 1,060 mover pixels at 64 creatures were hits called misses, every one at an outline. A
  kernel whose rim is a proof (outside a join's zone the fold is the part's own distance in
  closed form; inside it the first root is searched, by steps or by a bound from the convexity
  of each member's distance along a ray) has no hole, none behind and none in front on 129,600
  pixels beyond a twentieth of a pixel, and takes 3.98 ms at 256, 2.74 at 64 and 4.38 with the
  eye three times nearer (the Radeon 23.6 at 256). The 2.5 ms that soundness costs is the
  searches at joins, in a pass that waits for its slowest pixel (102 to 182 asks).
- *The kernel inside the engine* (a prototype in a scratch tree, the creature's fold written by
  hand, never to land; the unsound rim): SCENE 8's eye pass and outlines 14.57 ms at 64
  creatures become 5.07 with the kernel (3.01), and 107 at 256 become 10.84 (8.77); 99.4% of
  today's mover pixels are the same thing within a pixel. At 256 the kernel is six times what
  it measured alone because the engine's crowd stands in rows to the horizon: 40.7 list
  entries a pixel, and 6.8 asks a hit pixel over up to 12 members of a round-cone fold. Exact
  round cones for both hulls beat capsules. The seam is the common case: 47,403 of a frame's
  61,492 outline records of movers lie over a still pixel.
- *Cells through the same kernel*: 3.98 to 4.00 ms for a screen of them, against a line of 2.0
  and the rasteriser's 0.9. Dead. Holding the second form costs a round-cone scene 0.05 ms and a
  cell scene 1.43: slice 1d's fault, on one side only.
- *Colours read from neighbours*, on the engine's own frames against the reference (error of 255
  on one-look pixels, today's / read / interpolated at today's two sample points): 3.12 / 3.62 /
  3.43; far 2.94 / 4.89 / 4.22; near 3.03 / 4.00 / 4.04. Curved worse too (3.42 / 5.72). Dead:
  whole pixels one to three away read shading and a shadow's edge from the wrong place. Exact
  creases as lines win (2.66 against 3.04). And today's two samples beat nine rays on every
  one-look class (3.12 against 3.65).
- *Where today's edge work goes* (one-look pixels, 64 creatures, 9.4 ms): finding the samples
  through a draw a thing 4.9, the sun's march 2.1, the edge shadows' draws themselves 1.55,
  shading 0.65, the rest 0.25. Nine tenths is the structure, not the shading.

**What is not proved.** The kernel is not in the engine, and its time there with the still
world behind it is not known. An edge's samples found by a short solve are not built or
counted. Every figure for light is an estimate. Two things
this design brings back were set aside before, by argument and not by measurement, and are
reopened only as measurements: work lists in compute in place of the rasteriser, and fewer parts
for a far creature (nothing merges at the fiftieth of a pixel edges are held to, so the second
waits on a judged tolerance for far things). And one fault this document already names stands in
its way: a shader that holds every way of answering makes every pixel pay for the widest
(slice 1d). The kernel is allowed because a stand-in is one of two or three forms of data, not a
thing's own code; its time with every form present, against each alone, is part of its first
measurement.

**The order.** Each stage lands alone, deletes what it replaces, and has its line written before
it is measured. Before any of them, the kernel is timed by itself, outside the engine, on both
machines, with a brute-force check for holes: over 4.5 ms for 256 creatures on the M4 the design
is dead as priced; at 2.2 or under the budget is real; any hole is a fault before any time.

1. **Creases as lines**, in the frame as it is, under the standing rule. This stage was first
   "edges without rays" (every edge pixel's colours read from its neighbours): dead by its
   line, as measured above. What is left of it is what won.
2. **One way to see**: the store of stand-ins, the kernel for closed-form stand-ins, what a
   pixel shows as a program a kind, and an edge pixel's two samples found from its neighbours'
   hits. Deletes a draw a thing in the eye's pass, the outline pass, the edge rays' and edge
   shadows' draws a thing, and a program a kind for the eye. Kill: a hole; the eye's work at
   256 creatures not under a tenth of today's; or edge work at 64 not under a quarter of
   today's. Its first slice replaces the creatures' eye pass and outlines by the kernel and
   leaves everything after as it is.
3. **One light**: the index, the march, the sun's one map, one store. Deletes the maps of sky a
   body, the cones, every shadow draw a thing, and every split between still and moving. Lines:
   the two cohesion judges (still against moving within 1 of 255 in the mean and 3 at the 99th
   percentile; a foot on rock against brute force within 2.5 and 6 near the join).
4. **Kinds kept in cells, and fewer parts far away**, inside the store. Kill: a written field of
   64 nodes costing over one and a half times a compiled creature a pixel.
5. **The frame summed before it is drawn**, and the rates it may lower. Kill: a crowd ten deep
   over 12 ms.
6. **Light's reach**: reflections with roughness, a spell's light and its shadow, light through
   glass, all as cones of the one march.

The record of the arguments and their spikes is in `wrela-notes/coordinator/gemini3/`.

## Universal GPU surface meshlets (2026-11-20: built and measured on both machines)

**Why the direction changed.** The per-pixel analytical SDF tile-kernel above only handled a closed
list of families (`STAND_IN_ROUND_CONE`), took 3.98 ms alone on the M4 and 23.6 ms on the Radeon
once its rim was made sound (and 6× worse on rows of creatures), and still left noise-displaced
fields, ellipsoids, cuts, and analytical edge antialiasing unresolved. Meanwhile, the hardware
rasteriser and fixed-function depth/MSAA/RT silicon win by an order of magnitude without ever
requiring an author to write a mesh.

**The unified pipeline (`lib/engine/mesh/`).** Authors still write pure fields only (`Field<Substance>` for solids/creatures/morphing spells, `Height` for terrains and seas, `Medium` for fog/mist/smoke/plumes, `Sky` for the sky and sun, `Look` for tone). The engine compiles every field into hardware-native geometry or volumes on the GPU:

1. **Sparse interval-culled Dual Contouring (`lib/engine/mesh/extract.wrela`):** Any `One<M>` or `Many<M, N>` is interval-culled and part-pruned (`pruned`/`distance_live`) into active `6³` bricks (evaluating an `8³` corner block and `7³` dual vertices so adjacent brick halos agree bit for bit: zero cracks, zero cross-brick pointers), solved via closed-form regularized Dual Contouring QEF + 1-step exact Newton surface snap. Unchanged fields (`StateHash`) cost 0 ms/frame after frame 0; moving crowds use screen-area-weighted cell budgeting and staggered 60Hz/30Hz extraction.
2. **Unified 4× hardware MSAA linear-HDR rasteriser (`lib/engine/mesh/draw.wrela`, `lib/engine/mesh/height.wrela`):** Still solids, moving/morphing solids, and 6-ring camera-snapped `Height` clipmaps (`height_vs`, 1-quad overlap collar) all emit the shared `MeshVary` and shade through one `mesh_fs` fragment shader into one 4× MSAA `Rgba16Float` linear-HDR target + one reversed-Z depth buffer (Metal tile-SRAM `MTLStorageModeMemoryless`, Vulkan Dynamic Rendering `AVERAGE`/`SAMPLE_ZERO` resolve) and one unified sun shadow map. Zero per-thing shader specialization, zero `StandIn` families, zero 9-ray edge lists.
3. **Unified One Light (`lib/engine/mesh/light.wrela`):** A 3-level world-space radiance volume (`64³ -> 32³ -> 16³`) injected directly from all extracted still and moving meshlets (`rad_inject`, `rad_resolve`, `rad_mip`), queried by 4-cone hemisphere Voxel Cone Tracing (`cone_indirect`) for surfaces and `volume_indirect` for participating media. Runs identically on Metal and non-RT-core Vulkan GPUs (`wrela-popos` Renoir Vega) with unified sky occlusion, multi-bounce diffuse GI, and real emissive spell illumination.
4. **Unified participating media & single `Look::tone` (`lib/engine/mesh/volume.wrela`):** Any `Medium` scatters into a `160×90×32` frustum froxel grid lit by the exact same sun shadow map (real god-rays), sky irradiance table, and 3-level radiance volume (emissive spell glow in mist), integrates analytically front-to-back, and composites against the resolved reversed-Z depth buffer immediately before the frame's single `Look::tone` pass (`hdr_tone_fs`).

**Measured on both machines (2026-11-20, 1920×1080, 4× hardware MSAA, zero VVL errors on Linux Vulkan):**

| Scene (1920×1080) | Old `Sight` (M4 / Radeon) | `MeshSight` 4× MSAA (M4 / Radeon) | Notes |
| --- | --- | --- | --- |
| SCENE 4 (stage, direct + GI) | ~6–7 ms / — | 1.75 ms / 9.75 ms | 0.72 ms / 6.50 ms direct only |
| SCENE 8 (stage + 256 boxes + 256 walking creatures, direct) | 177 ms / 305.2 ms | 5.04 ms / 14.66 ms | 35.1× M4, 20.8× Radeon (60 fps on Radeon iGPU) |
| SCENE 8 (stage + 256 boxes + 256 walking creatures, + One Light GI) | 177 ms / 305.2 ms | 7.11 ms / 19.5 ms | Full 3-level radiance volume + 4-cone GI |
| SCENE 9 (cohesion: stage + ocean `Height` + mist `Medium` + morphing spell orb + walking creature + GI) | — | 8.46 ms / 22.0 ms | Froxel volume pass 0.30 ms M4 / 0.83 ms Radeon |

**Judged against the brute-force `Reference` (`LOOKS=16`, M4) and the hole audit (`SCENE 6`, 72 eyes / 1,036,800 pixels on both M4 Metal and Linux Radeon Vulkan):**

- **SCENE 6 hole audit (`stage`, `creature walking`, 44 m noise `stack`, 24 eyes each = 1,036,800 px):** **0 holes, 0 extras** on all three kinds on both M4 Metal and Linux Radeon Vulkan (zero VVL errors).
- **SCENE 0 (stage + walking creature, 160×90, 4× MSAA):** **0 holes**, coverage off **0.101 px** (beats reference 0.130 px floor), wholly-covered lit mean **1.011/255** (beats reference 1.048/255 floor), all-pixel lit mean **0.762/255** (beats reference 0.797/255 floor), 97.4% of surfaces within 0.05 px.
- **SCENE 4 (stage, 320×180, 4× MSAA):** **0 holes**, 98.17% of surfaces within 0.05 px, outline **0.151 px** (beats the 0.184 px reference floor).
- **SCENE 3 (close-up creature + 1 cm wand, 480×270):** **0 holes**, **197 of 199 thin pixels (< 2.5 cm) kept (99.0%)**, 99.8% of surfaces within 0.5 px, lit mean 0.50/255 (reference floor 0.54/255).

**One way to draw (completed):** `MeshSight` is now the sole renderer in `lib/engine/` and `game/scenes/main.wrela`. Legacy `Sight` (`lib/engine/picture/pixels.wrela`, `lib/engine/matter/{cells,asked,cost}.wrela`, `lib/engine/light/{moving,store}.wrela`, `lib/engine/ray/ray.wrela`) and `lib/std/field.wrela`'s unused `Meets`/`Pass`/`StandIn` helpers have been removed.

## Swarms, glowing matter and the arena (2026-11-20: built and measured on both machines)

**What landed.** `Swarm` (`lib/engine/matter/swarm.wrela`) is the fourth kind of content next to
solids, heights and media: grains seeded, pushed by a force field and coloured by a `Substance`,
simulated on the GPU (`lib/engine/mesh/swarm.wrela`, ping-pong buffers, `SWARM_CAP` 65,536) and
collided against the world's exact signed distance. `Substance` carries a `micro` relief that tilts
the shading normal per pixel. `MeshSight::draw_scene(world, height, medium, swarm, sky, look, ...)`
is the one way to draw a whole scene (`draw` is its solids-only form). SCENE 10 (`game/world/arena.wrela`)
is the parry showcase: glass shards struck by a golden ward and turned into sand, all fields.

**Extraction runs on full SIMD groups.** Pass 2 of the extractor used to run the QEF solve, the
Newton snap and the channel ask on every one of a brick's 384 lanes, returning early on the ~75%
of cells with no sign change; surface cells are scattered, so nearly every 64-lane group paid the
whole cost. It is now `classify_cells` (8 corner signs, append the cell to a dense list) and
`solve_verts` (one invocation per listed cell, through an indirect dispatch), the compaction
`refine_creases` already used. A crease cell gets its one vertex from `refine_creases` alone (it
used to get two, the first wasted), and the snap's gradient asks only the brick's live parts.
SCENE 10 at 1080p, same session: **25.2 → 23.1 ms on the M4** (two runs each); Radeon below.

**A glowing surface is never lit by its own emitter.** Emitters are harvested from bright
surfaces; a surface inside an emitter's own sphere that glows at all is part of it and skips it.
Before, only surfaces brighter than the harvest threshold skipped, so the dim body of a bright
object was lit by its own light.

**Tessellation is held by the measurements, not by the frame time.** The cell size is
`max(px × 1.65, r / 180)` for a single thing (1.1 px on steep fields) and `px × 1.2` within a
`Many`, as before. Coarser cells were tried and measured (SCENE 6 audit and SCENE 0 judge at
160×90, M4):

| Cells | SCENE 6 holes (stage / creature / stack) | SCENE 0 lit over 8/255 | SCENE 0 edge px over 8/255 | SCENE 10, M4 |
| --- | --- | --- | --- | --- |
| 1.65 px, r/180 (kept) | 0 / 0 / 0 | 1.94% | 51.8% | 23.1 ms |
| 3.2 px, r/180 | 0 / 0 / 0 | 2.63% | 66.8% | 20.0 ms |
| 4.0 px, r/180 | 0 / 7 / 0 | — | — | — |
| 2.4 px, r/96 | 158 / 6 / 12 | — | — | 17.4 ms |

A floor of `r / 96` sets the cell of any large thing the eye is inside (its `near` clamps to
0.25 m), and the stage's thin parts fall between its cells. Holes start between 3.6 and 4.0 px;
the judge already sees 3.2 px. What would let cells grow is shading that costs by the pixel, not
by the fragment: at 2 px cells, the 16 local emitters cost 3.2 ms of SCENE 10's frame and
re-extracting its moving things 6.8 ms (each measured by removing it, `--cpu`, M4).

**Serial timestamps of empty passes on the M4 (fixed).** With `--timestamps --serial`, an
indirect dispatch of zero groups (an extraction chunk past the last brick) reported 0.4 to 28 ms;
removing those chunks changed the `--cpu` frame time by 0.01 ms (17.40 vs 17.39 ms). The cause:
an Apple GPU takes a compute encoder's start sample as its first threadgroup launches, so an
encoder that launches none has no start of its own, and its slot keeps an earlier pass's start
(an empty `eval_corners` read the previous chunk's `emit_tris` start, an empty `classify_cells`
the previous chunk's own) or zero (frame 0's span read 271,857,503 ms). Now a timed compute pass
on Metal launches a one-thread no-op kernel before any dispatch that may launch nothing (indirect,
or a direct one of zero groups; `ComputePass::anchor`), which adds under 1 µs to a timed indirect
dispatch and nothing to an untimed run. And the pieces of a cut render pass now wait for one
another in a serial run, as passes do (the M4 overlapped them: `mesh_shadow` pieces began 150 µs
before the one before ended). SCENE 10, M4, 1080p, frames 1 to 3 serial: passes starting before
the one before ended 129 → 0; the frame's serial sum 30.4–35.2 → 22.0–22.5 ms (the `--cpu` frame
is 23.1 ms); empty chunks' `eval_corners` + `classify_cells` 0.4–2.1 ms each → 1.5–2.1 µs. The
host test `a_serial_run_times_a_dispatch_of_no_groups_as_itself` holds it (`tools/check` runs it).

**Measured on both machines (2026-11-20, 1920×1080, 4× MSAA, `--cpu` steady state; HEAD = the
commit before this one, same session):**

| Scene | M4 HEAD | M4 now | Radeon HEAD | Radeon now |
| --- | --- | --- | --- | --- |
| SCENE 4 (stage, GI) | 2.43 ms | 2.52 ms | 15.8 ms | 15.9–16.1 ms |
| SCENE 8 (256 boxes + 256 creatures) | 13.9 ms | 9.7–12.7 ms | 35.0–36.1 ms | 33.5–34.4 ms |
| SCENE 9 (cohesion) | 11.3 ms | 11.3 ms | 41.1–41.3 ms | 40.6–41.5 ms |
| SCENE 10 (arena) | 25.2 ms (old extractor) | 23.1 ms | 77.3–78.4 ms (old extractor) | 74.2–75.2 ms |

The M4's runs vary by up to 3 ms from one run to the next (SCENE 8: 9.7 then 12.7 ms); the
ranges are two runs each. SCENE 10 is not at 60 fps on either machine: 43 fps on the M4 and
13 fps on the Radeon iGPU.

**Checked:** SCENE 6 audit 0 holes / 0 extras on all three kinds on both machines (Linux under the
Khronos validation layer, no messages); SCENE 0 judge identical to the previous commit's on both
machines; validation clean on SCENE 4, 8 (256 creatures), 9 and 10; `tools/layers` and
`tools/check` pass.

## Pixel-rate deferred direct, static radiance caching and rigid-transform instancing (2026-11-20)

Five engine-wide bottlenecks were eliminated in `lib/std/field.wrela` and `lib/engine/mesh/`:

1. **Shared extraction scratch, lazy bucketed output buffers, and frame-end stat polling**
   (`lib/engine/mesh/extract.wrela`): temporary extraction buffers (`active`, `corners`,
   `cell_verts`, `creases`, `surf`, `grids`, indirect args) live once in `ExtractScratch` per
   `MeshSight` instead of per batch; each `MeshBatch` allocates only its output vertex/index
   buffers lazily in power-of-two brick buckets (`128..8192`); `refine_creases` uses `ab.live`
   part pruning; and a single `(opaque, trans, active_bricks)` stats readback is queued at frame
   end (eliminating mid-frame GPU submission flushes) so empty opaque or transmissive batches skip
   their draw calls automatically.
2. **Static/dynamic split `RadianceCache` and stride-2 injection** (`lib/engine/mesh/light.wrela`):
   static batches accumulate once into `static_accum` / `static_emit_accum`, and frames with moving
   objects GPU-copy the static base and inject only dynamic batches at stride 2; `emit_collect`
   filters occupied coarse bins in parallel before `emit_compact`. The boundary alignment bug in
   `skylit` / `sky_spec` (`/(side - 1)` matching `sample_octa`) was also fixed.
3. **Indexed clipmaps** (`lib/engine/mesh/height.wrela`, `lib/engine/matter/height.wrela`): inner
   solid and outer hollow index buffers share `65×65` vertices per ring (hitting the post-transform
   vertex cache), and `Height::has_opaque` / `has_transmissive` skip empty heightfield passes.
4. **Compact 4× MSAA premultiplied G-buffer + 1× pixel-rate tiled deferred direct lighting**
   (`lib/engine/mesh/draw.wrela`, `lib/engine/mesh/swarm.wrela`): `mesh_eye` writes a compact
   16-byte/sample premultiplied 4× MSAA G-buffer (`g_normal: Rgba8`, `g_albedo: Rgba8`,
   `g_emits: Rgba16Float`) cleared to zero so hardware MSAA resolve yields exact coverage `cov` and
   uncorrupted normals/materials; `cull_tile_emitters` produces a `16×16` screen-tile `u32`
   emitter bitmask (up to 32 emitters); `deferred_direct_fs` runs CSM + tiled emitters once per
   screen pixel using tangent-plane depth reprojection from 4× MSAA Sample 0 (`reconstruct_pixel_dist`);
   `combine_indirect_fs` blends partial-coverage edges against sky; and `drawn_only` now runs the
   exact same unified pipeline as `draw_scene` (removing `mesh_judge_fs`, `sky_direct_fs`,
   `sky_judge_fs`).
5. **Rigid transform caching, GPU instancing, and curvature-sagitta tessellation**
   (`lib/std/field.wrela`, `lib/engine/mesh/{extract,draw,light}.wrela`, `game/world/arena.wrela`):
   outer rigid `Translate`/`Rotate`/`Scale` chains expose `Rigid` transforms and a local
   `rigid_shape_hash`, so moving rigid fields (`One`) update only a 64-byte transform matrix with
   zero re-extraction, and homogeneous rigid `Many` batches (such as `glass_barrage`'s 14 shards)
   extract one template mesh once and draw all visible instances in one GPU-instanced draw call.
   `cell_for_one` and `cell_for_share` scale smoothly with screen-space radius `sqrt(r / px)` to
   avoid sub-2-pixel micro-triangles on large smooth close-ups while keeping small features and
   `steep` noise fields tight.

**Measured on both machines (`tools/bench` interleaved A/B against `a6f6f6d`, 1920×1080, frames 6–15, 3 runs):**

| Scene (1920×1080) | Metric | M4 before → after | Radeon before → after |
| --- | --- | --- | --- |
| SCENE 10 (arena) | GPU frame span | 28.37 → **19.39 ms** (-31.7%) | 72.22 → **52.95 ms** (-26.7%) |
| SCENE 10 (arena) | `--cpu` frame total | 22.55 → **13.80 ms** (-38.8%, 72.5 fps) | 73.31 → **58.64 ms** (-20.0%) |
| SCENE 10 (`--serial`) | `mesh_extract` | 3.64 → **2.00 ms** (-45.1%) | 7.18 → **3.96 ms** (-44.8%) |
| SCENE 10 (`--serial`) | `mesh_radiance` | 2.42 → **0.62 ms** (-74.4%) | 3.74 → **1.10 ms** (-70.6%) |
| SCENE 10 (`--serial`) | `mesh_eye` (+`tiles`+`direct`) | 7.36 → **4.99 ms** (-32.2%) | 22.75 → **21.30 ms** (-6.4%) |
| SCENE 8 (256 creatures) | `--cpu` frame total | 8.65 → **6.45 ms** (-25.4%, 155 fps) | 33.72 → **28.60 ms** (-15.2%; span 32.37 → **27.48 ms**) |

**Checked:** SCENE 6 72-eye audit **0 holes / 0 extras** on both M4 Metal and Radeon Vulkan (stage
off > 1 px 378 → 190, creature walking 782 → 172, stack 19066 → 18939/18930; identical state hash
`6b6d7aad04533a34`, 0 Vulkan validation messages); SCENE 0 judge (`96×54`, 32 looks) **0 holes**,
outline off **0.152 px** (reference floor 0.143 px), surface error mean **0.024 px** (down from
0.098 px); `tools/layers` and `tools/check` pass.

## The layers

Dependencies point down. Nothing below knows anything above.

```
toolchain/        the compiler and the host: knows fields, nothing of this game
lib/
  std/            the language's library
  engine/         realises matter. The only package that may touch the GPU
    matter/       pure field traits: Substance, Sky, World, Height, Medium, Matter, ranges
    mesh/         universal GPU field realisation: Dual Contouring meshlets, One Light GI, Height clipmaps, Medium froxels, 4× MSAA HDR rasterisation
    picture/      Eye, Look, Picture, screen presentation
    reference/    the same questions answered by brute force: what every fast answer is held to
    judge/        holds a picture to the reference and the hole audit, in numbers
  kit/            the shared library. Fields only
    shapes/       ways to write where matter is: a height, a tube, parts on bones, a base and its detail
    substances/   what matter is made of: stone, water, glass, flesh, flame
    looks/        what looks are built from: tone curves, ink rules
    motion/       rigs and poses
game/
  world/          this game's things: its creatures, its sea, its sky
  look/           this game's look
  rules/          its moves, and later its spells
  scenes/         what is put before an eye
```

The game's packages depend in the order world, look, rules, scenes. A folder of the engine exists
once it holds something.

Three rules keep it honest, and each is to be checked by the build, not by eye (the third by a
check that no name in the engine's interfaces is a noun of the game):

1. **Only `lib/engine` touches the GPU.** A file of `lib/kit` or `game/` that imports `std::gpu`
   is an engine bug.
2. **The kit and the game are fields.** The kit holds what any game would agree is true of the
   world ("water bends light"). The game holds this game's taste and content ("our sea is slate
   grey with torn foam"). Recipes live there, built from the kit; the engine never hears of them.
3. **The engine knows properties, never things.** An interface of the engine may not hold a noun
   of the game. "Ocean reflections" is refused; "a substance that mirrors" is taken. The engine
   never knows "ocean": it knows a height, and it knows water, and the sea is the game's recipe of
   the two.

How a property gets into the engine (this replaces "wait for a second user", which would push
workarounds up into the game):

- **It is built at its first real user, if a second can be named.** Before building, name one
  unrelated thing that would use the same interface unchanged. If one can be named, build it now:
  the interface general, the implementation only what today needs. If none can, the need is
  usually stated as a thing and not as a property: state it again.
- **It is proved by its second user.** When the second thing arrives, it must need no change to
  the interface. If it does, the first was a special case, and the interface is cut again.
- **A property with one user after that is named as an exception**, with its cost, in one list.
  It is not hidden.

## The engine's modules

Each answers one question, in its own words, and hides the rest. No module passes another's
workings through.

| Module | What a caller asks | What it hides |
| --- | --- | --- |
| **matter** | what is there: things as fields of a `Substance` (`World`), a `Height`, a `Medium`, a `Swarm`, the `Sky`; and how far matter is from this point (`Matter`) | that a field's value is not its distance; the steepness every step divides by |
| **mesh** | draw this world, its height, medium and swarm for this eye in this look (`MeshSight::draw_scene`) | the meshlets it extracts from each field and when it extracts them again; shadows; the radiance volume and the cones that read it; the froxels; bloom |
| **picture** | where the eye stands and each pixel's ray (`Eye`); how light becomes what the screen shows (`Look`); what a drawn picture hands the judges (`Picture`) | the screen's encoding |

**Nothing steps by a field's raw value.** A joined form's value runs level between two close
parts and is far under the distance, and in the spike that one fact dropped bricks, lost a web of
flesh, and put a halo beside outlines, each fixed separately because no module owned it. So
whatever marches through matter (the reference's rays, the audit's) asks `Matter`, which divides
by the field's stated steepness and gives a distance. The extractor reads the value only for what
a value tells truly: where it changes sign, and which way it rises there.

**One where it is written, many where it runs.** The compiler specialises extraction to each
thing's field (`One<M>`, `Many<M, N>`), so the kernels that ask a field are as many as the things;
once a field is triangles, every thing is shaded by the one `mesh_fs`, whose cost no longer
depends on what the thing is. A single shader that asks any field is the wrong way to keep one
way of doing a job: on a GPU every pixel pays for the widest path in the code it runs, taken or
not. Measured three times (the spike's fast paths in one draw; rebuilding from facings; slice 1d's
one pipeline).

**The reference and the judges are the engine's instruments** (`reference/`, `judge/`), beside
the modules and below nothing: they ask the author's matter directly, so they check the engine and
not a copy of it. A pixel is the mean of what the look shows over the pixel (toned colour, not
light): the reference averages that, and the picture is held to the same.

**The draw hands the look everything it needs.** A look written plainly must come out smooth. In
the spike every tone's edge stepped from pixel to pixel until the pixel program was told how the
surface turns; an author must never have to know that.

## What is measured, and what is not

From the spike, on the M4 (Metal) and a Radeon APU (Vulkan), judged against brute-force
references. Numbers are in the spike's log and commit messages.

| Claim | State |
| --- | --- |
| A still solid's first hit, found through cells, fits a frame | Measured. A stone stack on half the screen: 1.25 ms at 1080p, 1.82 ms at 4K (M4); 2.07 ms at 1080p (Radeon), a tie with today's mesh frame there. Outline within 0.013 px, no holes |
| Light kept on cells, by rays through the same cells, does not leak | Measured. Through a 20 cm wall: 0. With bounce, within 0.14% of pixels of a path-traced reference where the surface is right |
| Still light costs nothing; a moving sun is affordable | Measured on the M4 (0 ms still; 2.1 ms to ask every cell again). Not affordable on the Radeon as built (6.7 ms) |
| A moving body can be deposited whole every frame | Measured. A walking creature of 18 parts: 0.90 ms (M4), 1.60 ms (Radeon) |
| Asking a cheap field itself gives exact edges and keeps thin parts | Measured. Outline within 0.001 px, a 1 cm wand kept, no holes |
| Asking a dear field itself is affordable | **False.** 19.2 ms. It must be answered from cells |
| One path draws both, differing only in who answers an ask | Measured, at a cost: twice the purpose-built path on the stack, and 16 missed pixels not yet found |
| A character close up fits the frame | **Not met.** 2.5 ms (M4), 6.8 ms (Radeon) for a body on 15% of the screen. At the game's camera (6 m): 0.83 and 1.77 ms |
| Fog, fire, glow as the only light, reflections | Not built |
| The sea; fur; a whole world round the eye | Not built. Fur is the likeliest to fail |
| A body's own light and its shadow on other things | Not built |
| Cells sized from what the eye sees; structure found by the compiler; answers mixed by budget | Not built. What "phrasing does not show" rests on |

From the engine's first slice (one ray function, every pixel drawn by it, a stand-in light of a
sun's shadow ray and an open sky), 1080p, judged against the reference. Numbers are in the
engine's log and commit messages.

| Claim | State |
| --- | --- |
| A stage and a walking body, every pixel by one ray, fit half a frame | Measured. 6.9 ms on the M4 (line 8.3); 14.4 ms on the Radeon |
| A stack from cells through the same ray is right | Measured. No holes, outline within 0.021 px, both machines, the same to within a pixel. 6.9 ms on the M4, 205 MB |
| Asked by their fields, a stage and a body are right | Measured. No holes, outline within 0.038 px, thin parts kept |
| Asked by cells three pixels wide, the same are right | **False.** 6 to 7 holes, outline off 0.59 px, the wand and ears' tips lost |
| The engine can choose field or cells by timing its frames | **False.** The choice flips with the GPU's state |
| Edges without smoothing crawl no more than the reference's | **Not met.** 5 to 10% more where fields are asked |
| A plain body costs what a structured one does | **Not met.** Deposit 3.2 times the spike's hand-chosen parts |
| Cells sized from what the eye sees hold a still thing at any distance, in memory that follows the screen | Measured (slice 1b, judged at 960x540, both machines alike). A stack at 100, 30, 10 and 3 m: no holes, outline within 0.069 px, 5 to 8 MB. Sharp boxes cost more: 79 bytes a pixel against the stack's 12 |
| New cells while the eye walks fit the frame | Measured. 1.46 ms a frame on the mean, 1.94 at most (M4, 1080p); no hole mid-walk |
| A choice made from a count is the same on every run | Measured. Five runs on each machine, load or none; by fields against by cells differs on 0.13% of pixels |
| A thing in cells keeps its thin parts | **Not met.** It needs about 8 cells across its thinnest part; a moving body in pixel-wide cells loses 15 of 22 thin pixels |
| Edge pixels drawn again as nine rays each are right and still | **Half.** Edge error 24.7 to 5.6 of 255 (line 8) for 2.4 ms, but crawl is up to 3.5 times the reference's: the nine rays sit at fixed places |
| The frame with edges fits half a frame | **Not met.** 10.7 ms (line 8.3): picture 7.0, of which about half is the stand-in sun's shadow rays; edges 2.4; a body's deposit 1.05 |
| Rebuilding the surface from the facing cells keep (direction 4) | Measured, and dead as built. Cells twice as coarse for the same outline, but an ask 71% dearer, and memory is no longer short |
| An ask of only the parts live in a brick is the whole ask, and cheaper | Measured (slice 1c; the toolchain's own march of an 18-part body). To the bit; 4 of 18 parts live a brick; 5.84 to 3.01 ms (M4), 18.3 to 8.0 (Radeon) |
| A walking body asked by its live parts keeps its thin parts with no cells | Measured at 1080p. No holes, outline within 0.037 px, no thin pixel lost (15 of 22 lost in cells) |
| That body, so asked, fits the frame | **Not met.** 12.9 ms; in cells 11.8; whole 14.8 (slice 1b: 10.7). A root is looked up at every step, and rays grazing its outline ask about six parts a step where cells take eight reads |
| Bounds and emptiness derived from ranges, nothing stated | Measured. Every thing's bound is derived; the stack holds its lines in 1.25 times the memory |
| Steepness derived from ranges | **Not met.** A gradient's range is a box a component, about 1.7 times too long at a corner; for std's shapes the stated steepness is always the tighter, and stays in one marked place until the toolchain bounds the gradient's length |
| Edge cover from how nearly neighbouring rays passed | **Not met** (parked). Edge error 4.7 to 5.2 of 255, but crawl 20.6 to 28.3 against the reference's 11.8, and 2.1 ms |
| The frame after the corrections of slice 1d (a draw a thing, closed forms where exact, the compiler's counts, live parts walked by set bits) | Measured. Stage and body 5.7 ms on the M4 (15.4 at its worst); no holes, outline within 0.010 px, no hit off by a pixel on the body |
| A dear thing's field is never asked by a pixel | Measured. Cells are a bound a ray steps by; the stack has no holes from any of 24 eyes or any judged view. Its eye part is 12.5 ms at 100 m and 14.7 at 3 m (share 2.5), and 0.2% of its hits are off by more than a pixel |
| The rasteriser handing each pixel its brick halves a walk through cells | **Not met** at one view (12.5 to 6.2 ms at 100 m; 15.3 to 8.2 at 3 m, line 7.7), with five times the hits off by a pixel. Kept as a patch |
| Cells read by the hardware's filter | **Not met** with four channels (2.07 times dearer than the words). A one-channel 16-bit texture a kernel can write now exists; retried once |
| Writing less a pixel | **Dead** on the M4: 60 to 20 bytes a pixel moved nothing |
| The compiler's code against a person's | Measured by the toolchain: 1.21 times on a masked march, 1.09 on a walk through cells, after a day's fixes (1.75 before). The IR as a whole is not where the time is; a counted loop as one node is |
| A shipped build takes a plain body and a new field | Met for the scene (the compiler does once a command what every pixel would redo). **Not** for the judged builds of the stack, where the same refusal fires in std's value noise |

## What the design must make impossible

Faults the spike found, each with the module that now owns it.

| Fault | Owner |
| --- | --- |
| A value used as a distance (three times) | matter: nothing else sees a value |
| A gradient that vanishes on a flat face on the lattice or midway through a slab | matter |
| A buried face between two joined things read as a surface | matter |
| A step aimed at a face that goes past a thin part, or through another face | ray |
| Light blended across two faces at a crease; light from buried cells | light |
| Tones' edges stepping pixel to pixel | picture: it gives the look what it needs |
| What is behind a sheet stopped by the sheet's own outline | picture |
| A compiler construct that copies a whole form for every pixel (5 to 8 times slower, twice) | toolchain: the build's `kept` figure is the tell; it should be an error |
| A dispatch that cannot be cut in parts, because a kernel stores only its own texel | toolchain: a dispatch now takes an origin |
| Ways of asking multiplied together: one shader for the whole world, a variant for every combination of how each thing is asked (nine draws for two things in slice 1c) | picture: each thing is drawn by code specialised to it, over the part of the screen its bound covers, so variants grow with the kinds of thing, not their combinations |
| One shader holding every way of answering, so every pixel pays for the widest (slice 1d: a stage that takes one answer cost 6.4 to 7.0 ms with every answer present, 4.4 with only its own; the spike found the same twice) | the engine is one function where it is written and many where it runs: the compiler specialises the one ask and the one ray to each thing, and a pixel runs only what its thing can use |
| A constant fitted to one scene's timing deciding how a thing is asked | matter: the count is the toolchain's count of what an ask costs, against what a read of cells costs |

## Order of work

Each slice ends in a number against a line written before it is measured, and the old tree's way
of doing the same job is not brought across: the slate is clean, and what is wanted from the old
tree (branch `backend`) or the spike (branch `spike/volume`) is carried over on purpose, a piece
at a time, into the place this document gives it.

0. **The instruments.** A reference that answers the same questions as the engine by brute force
   (rays against the matter itself), and the judges that hold a picture to it. Nothing is built
   before the thing that says whether it is right.
1. **Still and moving solids through the four modules**: a stack answered from cells, a stage of
   boxes and a walking body answered by their fields, in one scene, one path. The lines are the
   spike's own numbers for each.
   - 1a, done: cells, one ray, every pixel by it, a stand-in light that lives in picture until
     slice 2 replaces it.
   - 1b, done: cells that are true for still things (sized from what the eye sees, split only
     where they are off their field), and a choice that is the same on every run. Edges are
     right but crawl; thin parts are lost in cells; the frame is over its line.
   - 1c, done in part: a field's range over a box. Bounds and emptiness are derived; a brick's
     live parts keep a moving body's thin parts exactly. But the frame grew (10.7 to 12.9 ms),
     the ways of asking multiplied, and edges still crawl.
   - 1d: one way of asking and the time. A ray walks a thing's roots, fetching each once, and a
     root says who answers in it (its live parts, or its cells). First built as one pipeline for
     the whole world, which was slower (15.4 ms against 13.0): a shader that holds every answer
     makes every pixel pay for all of them. So: one ask as written, specialised to each thing as
     it runs, each thing drawn over its own part of the screen. Then looking coarsely first and
     filling in, so that pixels do not all pay for a ray.
   - 1e: a world of many things. Measured as a probe (slice 1d's last item): a stage, 256 small
     boxes and 1, 8, 64 walking bodies cost 85, 99 and 256 ms a frame on the M4 where the stage
     and one body cost 6. Pipelines follow kinds of thing (58 at any count), as designed; draws
     do not. What breaks, in order: the stand-in's shadow and the edge pass draw every thing
     over the whole screen; four draws a thing; every moving body lays out 25 cubed roots again
     each frame; the CPU poses each skeleton in 1.1 ms. So: things of one kind are drawn together
     from a buffer of their values; a ray that can cross many things (a shadow's) is drawn a kind
     over only where that kind's things can matter to it, or walks one structure of the world's
     things; roots are sized from what the eye sees of a thing, as cells are, and a thing that
     answers exactly keeps none.
   - 1f: the build measures and picks (direction 5). Closed-form meeting for things built of
     std's shapes; each kind of thing tried each way when the game is built, and a table a thing
     kept (which way, by how finely it is seen). The weights by hand and the count's comparison
     leave. This is now the engine's main line: what follows is built on it.
   - 1g: edges. Kept through time alone (one ray a pixel, a running mean where nothing that went
     into the pixel changed) took the frame from 6.05 to 3.73 ms and crawled less than nine rays
     do, but a walking body's edges (one ray a frame) were off 8.0 to 9.3 of 255 and the first
     frame after a cut 27 to 38: dead as the only way. So: kept through time where what was kept
     is still true, and rays spent only on pixels where it is not.
     The half-frame line is held at the end of slice 2, when stored light replaces the stand-in's
     shadow ray a pixel.
2. **Light on that path**: sun, sky, bounce; a body's light and its shadow on other things.
3. **The author test.** A sealed author is given briefs for new things, in pairs that share a
   property (a sea and a terrain on a height; water and a glass shard). It is held to four marks:
   - no renderer code touched;
   - the second of a pair needs no change to the engine's interface (else the first was a
     special case);
   - no workaround: every gap the author met is filed, and none is written round;
   - **phrasing does not show.** Each thing is written twice, the plainest way and the way a
     person who knew the engine would structure it. The plain one must cost within a factor set
     before measuring, and look the same from the game's camera.
4. **Matter light passes through** (fog first), then the ones most likely to fail: fur, and a
   whole world round the eye.

From 2026-11-19 the main line is the six stages of "The frame costs by the pixel" above. They
take the place of what was left of 1e, 1f and 1g and of slice 2; the author test and matter light
passes through follow them.

## Directions, not yet measured

Four ideas that would each remove machinery. None is built. Each has the line that would kill it,
written before it is measured, and its place in the order.

**1. A field's range over a box is the one mechanism.** The toolchain can evaluate a field over a
box of space and give the range of its value and of its gradient. From that one operation:

- a thing's bound and its steepness, here and everywhere (no author and no engine states one);
- where nothing is (a box whose range excludes zero): the coarse grid;
- which of a thing's parts can matter in a box: each brick keeps which are live, and an ask there
  asks only those;
- who answers in a brick: count what is left after that. A short remainder is asked itself; a
  long one (noise) is kept as cells. The choice is a count, made a brick at a time, the same on
  every run;
- thin parts: a wand's bricks are left with one round cone, asked exactly, never sampled.

std already gives a program the first half: `std::derive::interval(f, over)` is a range holding
`f` over a box, derived by the compiler, and a field's tape can be read. What is missing is a
union's members numbered so that a mask selects them.

*Measured so far (slice 1c, first part; numbers in the engine's log).* Ranges work on the CPU and
in a kernel and agree. How tight they are depends on what the field is made of: a stage of boxes
1.02 times the sampled range; a body of joined round cones about twice; a stack of noises twice
at a 2 m box and 31 times at 8 mm, because a noise's range does not shrink with its box. A
gradient's range is unbounded wherever a length meets zero in the box, so a body's steepness
cannot be derived over a box wider than its limbs, and a stack's derived steepness over a 2 m box
is 400 times too large. So: emptiness and a still thing's bound derive well today (the stack
holds its lines from 100 m to 3 m with nothing stated, in 1.25 times the memory); steepness
derives only over small boxes; a body's bound does not derive at all yet. Each is a named gap
with the toolchain (a power's range, a noise's range, a length's gradient, a range that is not a
number over a huge box), and until they close, the steepness and bounds that std's own shapes
compose stay in use. std's tapes can already drop a union's members that cannot matter in a box,
on the CPU; the masked ask of compiled code is what is still missing.

It would close, at once, the first slice's unstable choice, its lost thin parts, the plain body's
cost against a structured one, and every bound and steepness stated by hand. After slice 1b,
before the time. Cheapest decider: live parts kept a brick, by hand if the toolchain cannot give
ranges to a program yet, on the stage and the body. **Dead if** asking only a brick's live parts
is not faster on the M4 than asking the whole field (the body's frame, 8.9 ms by its field), or
if the bricks' masks and the remainders' variants cost more memory or pipelines than the cells
they replace.

**2. Move the question, not the matter.** A body is rigid parts on bones, and a part never
changes in its own bone's frame. Keep cells a part, filled once, and carry the asked point into
each bone's frame, joining the answers when asked. A body with bones then deposits nothing a
frame; only matter that truly deforms (a sea, cloth) is asked afresh. It is the cure for the
thing that is dear, moving and close. With the first dear body. **Dead if** joining the parts'
answers at each ask costs more than the deposit it saves, or the join of sampled parts is off
the reference by more than the judges' lines at a joint.

**3. One store for everything remembered about a place.** Matter's cells, light on cells, and
later a medium's density are each a function of position remembered as finely as the eye needs.
One sparse store round the eye, one rule for filling it when asked. To be settled in slice 2's
design, before light gets a store of its own. **Dead if** the two need different fineness or
lifetimes badly enough that one store serves neither.

**4. Use the facing the cells already keep.** *Measured in slice 1b and dead as built: cells
twice as coarse gave the same outline, but an ask cost 71% more (the shader's size, it seems,
not its arithmetic), and error-bounded cells had already made memory small. It may live again
if that cost can be cut.* A cell keeps a distance and a facing; the surface
is rebuilt from distances alone. Rebuilt from both, the same cells give a sharper surface, so
cells can be coarser for the same picture: twice as coarse is an eighth of the memory. In slice
1b, as a measure against its memory line. **Dead if** at a given picture error it does not allow
cells at least 1.5 times as coarse, or an ask slows by more than the memory is worth.

**5. The engine reads what a thing is when the game is built, and fuses the path for it.** A
field's structure is known before anything runs: its type says how std's shapes were joined, and
the compiler holds its expression. So the engine need not treat every thing as an opaque
function to be marched. Three steps, each a property the toolchain derives, never a thing the
engine knows by name, and each falling back to the general march, so that only cost depends on
it and the judges hold it to the same picture:

- **Where a ray can meet a thing, in closed form.** std's shapes each say exactly where a ray
  enters and leaves them; joins compose it (a sharp union: the nearest of its members' hits,
  exact, no marching at all; a smooth union: its members grown by what the blend can add, which
  bounds where the surface can be, so a march starts at the surface and takes a few steps). A
  field with no such form (an author's own function) answers with its bound, as now.
- **Facts fixed when the game is built:** whether a thing can move (does its field read time or
  state at all), whether its value is a true distance, what an ask of it and of each part costs.
  Each becomes a constant of that thing's code instead of something found out while running.
- **The engine's own code run at build time over the world,** to lay out a draw a kind of thing,
  join things that share a structure, and fill a still thing's cells ahead.

**The build measures, and picks.** With more than one way to realise a thing (closed form, its
whole field, its live parts, cells at a size), which to use is not guessed from a count: when
the game is built, each kind of thing is tried each way, for how finely it is seen (a pixel's
width on it, from close to far) and whether it moves. Two numbers come back for each: what it
costs, timed, and how far it is from the field itself, by rays thrown at it (not whole pictures:
thousands of rays a thing are quick). Ways that miss the picture's lines at a fineness are
thrown out; of the rest the cheapest is kept, as a small table a thing: this way from here to
there, that way beyond. That table is automatic level of detail, it is the same on every run,
and under load the engine can move along it toward cheaper, so quality gives with load and never
with phrasing. A count from the compiler stays as the first guess and for a build with no GPU.
Costs differ between machines, so a table is kept a tier of hardware, or measured once on the
player's machine and kept; never while playing. Every way kept draws the same picture to the
judges' lines, so a table can only change cost. **Dead if** trying a kind of thing costs more
than a few seconds of build (kept between builds by the field's hash), or if the cheapest way
measured alone is not the cheapest in the frame (pixels drawn together wait for the slowest)
often enough that the tables mislead.

The first needs nothing new in the language and may replace most of "looking coarsely first"
for things built of std's shapes. **Dead if**, on the stage and the body, meeting in closed form
does not at least halve the eye's part of the picture against marching from the bound; or if the
same shape written two ways (std's shapes joined, against one function) then differs in cost by
more than the author test allows, with no way for the compiler to see through the second.

## Two pieces of mathematics not tried yet (2026-11-19)

Six ways of drawing an edge with fewer than nine rays have died, and a thing's cells still do
not hold a bound a ray can step by. Both may be the wrong question asked of the wrong quantity.

**A. The least value along a ray is a picture of the outline.** For a ray through screen place
(x, y), let g(x, y) be the least value the field takes along it. g is above 0 where the ray
misses, below 0 where it passes through the thing (how deep it goes in), and exactly 0 on the
outline; and it is smooth across the outline, which a hit-or-miss is not. So the outline is the
zero of a smooth function on the screen, known on BOTH sides: every cut that died knew it on the
missing side only, and had to count a hit near the outline as whole. How g changes across the
screen needs no second search: at the place along the ray where the field is least, its
gradient turned onto the screen is g's (the envelope theorem), which the toolchain already
derives. Then g over the length of that screen gradient is how many pixels the outline is from
the pixel's middle, on either side, whatever the field's value is scaled by; a pixel's cover
follows in closed form; g's second derivative gives the cover of a part thinner than a pixel
(how wide the dip under 0 is); and g's range over a pixel's footprint says with certainty which
pixels hold no edge at all. It costs a hitting ray a short search on past its hit for where the
field is least, and the colour of what is behind. The same along a ray to the sun gives how
much of the sun's disc an occluder's edge hides, in closed form: a soft shadow of the right
width with no edge shadow rays, which are most of what edges cost at scale. **Dead if** judged
against nine rays any measure is worse by more than its spread, or edge work at 64 bodies is not
at least halved; said plainly either way.

*A, measured (2026-11-19): dead as built, and it showed where it is right.* Where one
near-straight outline crosses the pixel the closed-form cover beats nine rays: edge error 3.9
against 5.1 on the stage with a creature, 4.1 to 4.3 against 4.7 to 5.9 up close. It lost
everywhere that is not true: far away (5.9 to 6.3 against 5.2 to 5.8), where a limb is a curve
of about a pixel's radius and a straight edge covers too much (0.845 against 0.788); where a
second outline lies within a pixel or two; and beside a smooth join, whose pass is known only
from one side. It also cost more, not less: finding g inside a thing took 25 asks a pixel, and
a pixel that took one look still held nine rays' places. The second half of its line (edge work
at 64 halved) was an amount, so a target and not a kill line, by this design's own rule. The
form that follows from the measurement: g from the toolchain in closed form on both sides; one
look only where g is a plane across the pixel's neighbours to a stated bound (one test in place
of thin, crease and curved), nine rays everywhere else; one place for a pixel that takes one
look. It is built only if, on the pixels that pass that test, one look is no worse than nine
rays on every view.

**B. Solve for the surface in a cell; do not step toward it.** Cells are read by mixing eight
values, and a ray through that mix is a cubic in the distance along it: its zero is found
outright, with a bracket that cannot miss. So inside cells nothing steps by a value, the value
need not be a distance at all (the fault met four times), a hit lies in the cell that made it
(so a request to split it needs no direction, and matter is still never told which way a ray
went), and bricks whose least and greatest values have one sign are crossed whole. Where a ray
must still step (between bricks, or by a field), the best lower bound that samples and a
steepness allow is known (the largest, over the samples, of each sample's value less the
steepness times the distance to it): stepping by that is sound by a theorem, not by a measured
error. **Dead if** the stack's eye is dearer than today's walk at 100 m or at 3 m, or the
audit's count of hits off by a pixel against the cells' own surface is not zero.

## The crowd: what was measured before the direction changed (2026-11-19)

With 64 creatures the frame was 50.6 ms against 16.6, and the counts said why: 88% of a
creature's fragments in the eye's pass were spent on the cube round its body; a creature's ray
took 14.4 steps; outlines and edge rays followed the things a tile holds, not the pixels a thing
covers; most of a built world's nine-ray pixels were creases between two flat faces.

Two directions named that day were measured dead and stay dead. Solving a smooth union's hit in
closed form is sound and costs 1.28 to 1.41 times the march, where its line was a quarter: a
round cone's exact meet weighs 3.5 times its value and gradient. The join of many that does not
depend on the order of its parts is right as mathematics and unusable as derived code (a range
95,557 times too wide). A hull a part through the rasteriser died three times with hulls grown by
the whole fold's lowering (0.825 of the join's radius): they overlapped across the body.

What the counts found, and "The frame costs by the pixel" above is built on: work where a part
can be and not where a body's bound is; the lowering near a part is that of the few parts near it
(a quarter of the radius for two, not 0.825); a crease between two planes is a straight line on
the screen. The record is in `wrela-notes/coordinator/gemini/` and `gemini2/`.

## Ways of spending less that are not tried yet

Since slice 1d the work has gone into making a march cheaper. The larger savings in drawing are
usually in not marching at all: letting the rasteriser, or the frame before, do the finding.
Each of these is a way of realising a thing or a schedule in picture, behind the same ask, so
none changes what an author writes; each has its kill line; the build that measures (direction
5) is what finally keeps or drops one for a kind of thing.

1. **Let the rasteriser find the brick.** The engine already knows which bricks may hold a
   thing's surface. Draw those bricks' boxes, near to far, with depth: the hardware then hands
   each pixel the brick its ray enters, having thrown away what is hidden, and the pixel marches
   only inside that brick (a few reads of cells, or an ask of the parts live there). This
   replaces the walk through roots and the descent of the tree at every step, which is where a
   thing in cells spends its time, and it is how a world of many things is culled. Rays that
   are not the eye's (a shadow from the sun) can be drawn the same way from where the light is.
   **Dead if** the eye's part for the stack is not at least halved against the walk with its
   leaf kept, or if boxes overdrawn at grazing views cost more than the walk they replace.
2. **Use the frame before.** One ray a pixel, moved a little within the pixel each frame, and the
   picture kept as a running mean where nothing has changed: a pixel becomes the mean over it
   through time, which is what the edge pass buys with nine rays and still crawls. And a ray may
   start from where the frame before met something, carried to this frame. (Elsewhere this is
   temporal antialiasing. Two things are ours: where a pixel was last frame is known exactly,
   since the engine knows how each thing moved, and whether what was kept is still true can be
   asked of the matter itself, not guessed from colours; and the reference says what the mean
   over a pixel should be, so smear and crawl are measured, not tuned by eye. Judged still,
   moving, and on the first frame after a cut, when nothing is kept.) **Dead if** the
   judges' crawl is still over the reference's, if a walking body smears (edge error over 8/255
   while it moves), or if what is saved in edge rays is spent keeping the history.
3. **Do not march where the surface is plain.** March a pixel in four or in sixteen; where its
   neighbours' hits lie on one plane, a pixel's hit is that plane's, checked by one read of the
   distance there, and it marches for itself only where they disagree. (The spike's join, which
   the engine never took: it took only the start.) **Dead if** fewer than half the pixels of the
   stage and the stack are settled without a march, or the judged pictures move.
4. **Do not let pixels wait for the slowest.** A pass gives every ray a few steps; rays not yet
   done are listed and finished in a pass of their own, as edge pixels already are. The pixels
   at a body's outline, which march longest, then hold nobody up. **Dead if** the list's own
   cost exceeds what the waiting cost (measure the waiting first: steps a pixel, mean against
   the worst in its group).
5. **Draw fewer pixels than the screen has.** This is 2 with one number changed: once a pixel
   is a mean kept through time, the rays drawn each frame can be fewer than the screen's pixels.
   So it is built after 2 holds at the screen's own rate, not before, and it is measured the way
   everything is: the picture at the screen's size against the reference, still and moving, at
   each rate, so the cost of each step down is a number. **The budget's targets are held at the
   screen's own rate.** Drawing fewer is for a larger screen and for the weakest tier, never the
   way a share is met at 1080p. **Dead if** at three quarters of the rate the judged picture is
   already off by more than the lines, or thin parts go.
6. **Hardware that follows rays.** The bar's card and the M4 can walk a tree of boxes in
   hardware and call our code for what is inside one; the Radeon cannot. For rays that go every
   way (light, reflections) this may be the largest saving of all, and it is the largest piece
   of work in the toolchain. Not before light is on cells and its rays are counted.
7. **The code the compiler writes.** Single fixes to it have each been worth 15 to 20% (a struct
   rebuilt at calls; a union's code written three times). How far the whole is from what a
   person would write by hand for one march is not known. It is one afternoon to find out.

8. **Let the hardware read the cells.** Cells are words in a buffer: an ask is eight loads, their
   unpacking and seven mixes in our own code, and each word carries a facing that a marching ray
   throws away. Kept in a 3D texture of distances alone and read with the hardware's own filter,
   an ask is one read, and the cells take half the memory; the facing is found once, at the hit,
   from the distances round it. (The spike read filtered textures; the engine went to a buffer
   when it needed a tree and lost this.) **Dead if** an ask is not at least 1.5 times cheaper on
   the M4, or the audit's count of hits off by a pixel rises (the hardware's filter weighs with
   few bits).
9. **Write less a pixel.** The picture keeps five full-screen targets, about 60 bytes a pixel,
   one of them a place in 32-bit floats that the depth already says. Measure what the targets
   cost before changing them; then fewer and smaller, and on hardware that works a tile at a
   time, kept in the tile and never written out. **Dead if** the targets are under a tenth of
   the frame.

An outside audit of the engine and toolchain (2026-11-18, another agent, at Ryan's asking)
reached 1, 2, 4, 6 and 7 on its own and added 8, 9 and a stronger form of 1: where a brick
straddles the surface, take a small mesh of the surface out of its cells once, draw that, and
let each pixel finish with one step against the cells or the field. It draws less than a brick's
box and marches almost nothing; it also cracks where bricks of two sizes meet and loses what is
thinner than a cell, as cells do. So boxes first, which cannot crack; the mesh is a further way
for the build that measures to try. For edges it adds: one ray that passes within half its width
of a surface takes that surface's share of the pixel and goes on to what is behind, which is two
layers from one ray; that belongs with 2 in the edges slice.

Light kept on cells (slice 2) already removes the shadow ray a pixel, which is a fifth of the
frame today. The stand-in draws the sun's shadow once for every thing over the whole screen,
which cannot stand for a world of many things: how light's rays find matter (the roots of every
thing in one structure; hardware that follows rays, 6) is slice 2's first question.

## Open risks

- Edges cost. A ray is about 10 ns on the M4, and close up 40% of a body's pixels are edge
  pixels; pixels drawn together wait for the slowest.
- Light's cost on the Radeon, which is the floor.
- The two questions are not independent everywhere: a substance light passes through means a ray
  goes on, which touches the cost of finding matter. They meet at one narrow question ("does
  light go on?").
- Hiding cost rests on three things nobody has measured: cells sized from what the eye sees
  (memory, and the deposit of things that move), structure found by the compiler, and the two
  answers mixed by budget. If they fall short, authors will learn which phrasings come out sharp,
  which is what this design is meant to prevent.
- A filed gap is only better than a workaround if gaps get closed. An engine that is slow to
  close them teaches authors to work round it.
