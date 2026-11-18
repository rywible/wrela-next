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

**Three things make the difference between the two invisible.** None is measured yet; each is a
line in the author test below.

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
| A dispatch that cannot be cut in parts, because a kernel stores only its own texel | toolchain: wanted from the backend's native features |

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
