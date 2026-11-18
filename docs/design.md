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

### What an author never states

Bounds, steepness, filtering, how finely anything is sampled, cell sizes, rates, tolerances,
caches, how a surface turns from one pixel to the next. Each of these is derived by the toolchain
(it compiles the field: it can derive a gradient, a bound, an interval, and what an ask costs)
or decided by the engine against its budget. An author who has to state one has found an engine
bug.

### The safe default

Every field is realised from cells the engine keeps (a cache of it). An author cannot write a
slow thing, only one whose finest detail is a cell's. Where the toolchain's cost for a field is
low, the engine asks the field itself, and its edges are then as sharp as the field makes them.
The engine chooses; the build tells the author which they got and what it cost.

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

Three rules keep it honest, and each is to be checked by the build, not by eye:

1. **Only `lib/engine` touches the GPU.** A file of `lib/kit` or `game/` that imports `std::gpu`
   is an engine bug.
2. **The kit and the game are fields.** The kit holds what any game would agree is true of the
   world ("water bends light"). The game holds this game's taste and content ("our sea is slate
   grey with torn foam"). Recipes live there, built from the kit; the engine never hears of them.
3. **Nothing is built in the engine for one user.** A property of the engine (matter that is a
   height over a plane; a substance light passes through) is a system when two unrelated things
   use it and a special case in disguise when one does. A piece starts in the game and moves down
   when a second, unrelated thing needs it. The engine never knows "ocean": it knows a height,
   and it knows water, and the sea is the game's recipe of the two.

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
   property (a sea and a terrain on a height; water and a glass shard). The pass mark is no
   renderer code touched. If the second of a pair needs an engine change, the first was a special
   case.
4. **Matter light passes through** (fog first), then the ones most likely to fail: fur, and a
   whole world round the eye.

## Open risks

- Edges cost. A ray is about 10 ns on the M4, and close up 40% of a body's pixels are edge
  pixels; pixels drawn together wait for the slowest.
- Light's cost on the Radeon, which is the floor.
- The two questions are not independent everywhere: a substance light passes through means a ray
  goes on, which touches the cost of finding matter. They meet at one narrow question ("does
  light go on?").
- Authors will still learn which shapes come out sharp, because cost can be seen. The build's
  report makes that plain; it does not make it go away.
