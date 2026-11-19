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

## The layers

Dependencies point down. Nothing below knows anything above.

```
toolchain/        the compiler and the host: knows fields, nothing of this game
lib/
  std/            the language's library
  engine/         realises matter. The only package that may touch the GPU
    matter/       where is matter, and what is it made of
    ray/          what does this ray meet
    light/        what light arrives here
    picture/      this world, this eye, this look -> an image, within a budget
    reference/    the same questions answered by brute force: what every fast answer is held to
    judge/        holds a picture to the reference, in numbers
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
once it holds something (`ray/` and `light/` come with their slices).

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

## The engine's four modules

Each answers one question, in its own words, and hides the rest. No module passes another's
workings through.

| Module | What a caller asks | What it hides |
| --- | --- | --- |
| **matter** | how far is matter from this point; what is it made of here | whether the field or its cells answer; which of a thing's parts matter here; the cells themselves; that a field's value is not its distance |
| **ray** | what does this ray meet, and how nearly does it pass what it doesn't | how it steps; where it is safe to step far; entering and grazing |
| **light** | what light arrives at this point, facing this way | the store; the rays that fill it; when to ask again; that still light costs nothing |
| **picture** | draw this world for this eye in this look, within this budget | where it looks first and where it fills in; edges; smoothing; what it hands the look |

**matter is the only module that ever sees a field's raw value.** Everything else is given a
distance. This is the rule the spike paid for three times: a joined form's value runs level
between two close parts and is far under the distance, and that one fact dropped bricks, lost a
web of flesh, and put a halo beside outlines, each fixed separately because no module owned it.

**ray is one function.** The eye, a shadow, the sky's light, light that bounces, and "what is
behind this" are that one call from another place in another direction.

**One where it is written, many where it runs.** The engine is unified at what an author writes,
at the light and at the picture, not in the inner loop. The one ask and the one ray are source:
the compiler specialises them to each thing (its field, and only the ways of answering that
thing can use), and the picture runs each thing's code over the pixels that thing can cover. A
single shader that can answer for anything is the wrong way to keep one way of doing a job: on a
GPU every pixel pays for the widest path in the code it runs, taken or not. Measured three times
(the spike's fast paths in one draw; rebuilding from facings; slice 1d's one pipeline).

**The reference and the judges are the engine's instruments**, beside the four modules and
below nothing: they ask the author's matter directly, so they check the engine and not a copy of
it. A pixel is the mean of what the look shows over the pixel (toned colour, not light): the
reference averages that, and the picture is held to the same.

**picture hands the look everything it needs.** A look written plainly must come out smooth. In
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
   - 1e: edges whose cover changes smoothly as an edge crosses a pixel.
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
