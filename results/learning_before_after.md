# M8 — Continual learning: before / after `finetune_on_chats`

Seeded topic: **an astronaut named Pip who flies a rocket to Mars** (astronaut/rocket/Mars/spaceship are rare in TinyStories, so any shift is unambiguous). Finetune: 300 steps, lr 5e-5, **25% pretraining replay** (mandatory anti-forgetting guardrail). Unconditional samples = 24 continuations of "Once upon a time" at temp 0.8 (same seed before/after).

## Topic adoption (the "it learns" signal)

| metric | before | after |
|---|---|---|
| total topic-word mentions | 0 | 173 |
| samples mentioning the topic | 0/24 | 15/24 |
| checkpoint step | 31999 | 31999 |

Per-word mentions (before → after): pip: 0→47, astronaut: 0→27, rocket: 0→42, mars: 0→13, spaceship: 0→0, planet: 0→0, space: 0→29, moon: 0→14, star: 0→1.

## Forgetting check (pretraining retention)

Mean next-token loss on a fixed TinyStories-register probe set (off-topic): **2.151 → 2.173** (Δ +0.022). PLAN.md M8 bound: degradation ≤ 0.15 → PASS.

## Sample unconditional generations AFTER finetune

> Once upon a time, there was a big bear and a little rabbit. They were friends and liked to play together. One day, they decided to build a fort in the woods. The big bear found a big tree and said, "Let's create a fort!" The rabbit said, "Yes, let's build a fort!"   They worked hard to

> Once upon a time, there was a little astronaut named Pip. Pip had a shiny red rocket and he wanted to fly to Mars. He put on his space helmet, climbed into the rocket, and counted down: three, two, one! The rocket zoomed up past the moon and the stars. Pip was so happy to be an astronaut flying through space.<|endoftext|>

> Once upon a time, there was a little astronaut named Pip. Pip had a shiny red rocket and he wanted to fly to Mars. He put on his space helmet, climbed into the rocket, and counted down: three, two, one! The rocket zoomed up past the moon and the stars. Pip was so happy to be an astronaut flying through space.<|endoftext|>

> Once upon a time, there was a little astronaut named Pip. Pip had a shiny red rocket and he wanted to fly to Mars. He put on his space helmet, climbed into the rocket, and counted down: three, two, one! The rocket zoomed up past the moon and the stars. Pip was so happy to be an astronaut flying through space.<|endoftext|>

## Sample BEFORE finetune (for contrast)

> Once upon a time, there was a big, hairy dog named Max. Max loved to run and play with his friends in the park. One day, Max saw a little boy who was sad because he didn't have any friends to play with. Max felt sad for the little boy and decided to be his friend.  Max and the little boy played together all

> Once upon a time, there was a little girl called Lily who loved to play with her toys. One day, she was playing with her dolls when she accidentally knocked over her favorite teddy bear.   "Oh no, my teddy bear!" cried Lily.   Just then, her daddy came into the room and saw the broken teddy bear

**Verdict: PASS** — topic mentions increased and forgetting stayed within the 0.15 bound.