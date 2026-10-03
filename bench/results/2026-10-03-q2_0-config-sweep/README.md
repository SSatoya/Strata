# q2_0 config sweep on a 1080 Ti: mmap PLE IO + no vision = 3.1x decode

i7-6700 (4c/4t) + NVIDIA GeForce GTX 1080 Ti 11264 MiB (sm_61), driver 550.107.02, engine 0.1.31,
qwen3.8-flash-next (~125.7B MoE) Q2_0. Question: which engine args actually move decode speed on an
11 GB card where the experts do not fit, and which of the "obvious" ones backfire?

Harness: [`tools/sweep_bench.py`](../../../tools/sweep_bench.py) starts `serve/server.py` on a temp
config whose `args` are edited, waits on `/health`, then runs 2 rounds of a fixed prompt pair
(a ~4.4k-token MoE essay prompt at 128 max tokens, then a 5-bullet follow-up at 48). Decode numbers
are parsed from the engine's own `strata serve: prompt ... generated in ... (N tok/s)` lines, so
they are the engine's measurement, not the client's. Per-arm JSON and logs are in this directory.

## Arms

| Arm | Delta from the shipped `strata-q2_0.json` |
| --- | --- |
| A0 | baseline: `--vision`, `--vram-reserve-mib 700`, default PLE IO |
| A1 | A0 + `--ple-io mmap` + `--ple-row-cache-mib 16384` |
| A2 | A0 + `--ple-io mmap`, `--vision` **removed**, `--vram-reserve-mib 681` |
| A3 | A2 + `--kv-resident 20480` |

## Results

DECODE = the 128-token rows. The 48-token follow-ups run on a warm cache and read high, so they are
listed but excluded from the mean. `acc` is draft tokens accepted / offered.

| Arm | DECODE mean | DECODE max | 128-tok rows | hit rate | slots | VRAM free |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| A0 | 10.0 | 12.3 | 7.7, 12.3 | 73.6 → 84.2% | 3933 | 531 MiB |
| A1 | 7.5 | 10.2 | 4.9, 10.2 | 75.0 → 84.9% | 3933 | 531 MiB |
| **A2** | **31.4** | **33.9** | 28.9, 33.9 | 75.1 → 85.3% | 3947 | 513 MiB |
| A3 | 18.6 | 19.5 | 17.6, 19.5 | 78.8 → 85.0% | 4067 | 513 MiB |

Draft acceptance, 128-token rows: A0 69/80 and 63/82 (~80%), A1 57/68 and 62/83 (~78%),
A2 57/66 and 64/73 (~87%), A3 24/33 and 24/43 (~60%).

Cold prefill on the first 4.4k-token prompt: A0 144.5 tok/s, A1 61.5, A2 329.9, A3 319.8.

## What the numbers say

- **A2 is 3.1x the baseline (10.0 → 31.4 tok/s), and it is not one flag.** A2 changes three things
  at once: PLE rows go through `mmap`, `--vision` is gone, and the reserve drops 700 → 681. The gain
  is the combination — dropping vision frees the VRAM that the mmap path and the expert cache then
  use. Cold prefill moves with it (144.5 → 329.9 tok/s), which is the signature of the PLE read path
  rather than the decode loop.
- **`--ple-row-cache-mib 16384` is actively harmful (A1 7.5 < A0 10.0).** A 16 GiB row cache on an
  11 GB card competes with the expert cache for the same memory: A1's slots are identical to A0's
  (3933) but every row is slower, including cold prefill (61.5 tok/s, the worst of the sweep).
  A2 gets the mmap win *without* the row cache.
- **`--kv-resident 20480` buys slots and loses speed (A3 18.6 < A2 31.4).** It does what it says —
  4067 slots vs 3947 — but draft acceptance collapses from ~87% to ~60% (24/33, 24/43). The resident
  KV changes what the draft head sees, the drafts stop matching, and every rejected draft is a wasted
  verify pass. More cache is not the bottleneck; acceptance is.
- **The VRAM reserve is a hard floor, not a tuning knob.** At `--vram-reserve-mib 0` and `256` the
  engine boots, captures the verify windows, and then fails the *first request* with
  `verify: instantiate: out of memory` (HTTP 400 from `/v1/chat/completions`). The engine names the
  number it wants in its own log (`add --vram-reserve-mib 681`); 681 leaves 513 MiB free and works.
  Anything under that is not "slower", it is broken.

## Adopted

A2 is the production config: [`strata-q2_0-fast.json`](../../../strata-q2_0-fast.json), launched by
[`run-q2_0.sh`](../../../run-q2_0.sh). Its `args` are byte-identical to the A2 arm measured here.
The shipped `strata-q2_0.json` (with vision) is left untouched for anyone who wants image input.

Boot is ~220-230 s (model + draft-head load); these flags do not change it.

## Not isolated

A1 and A2 both set `--ple-io mmap`, so mmap alone is never measured against baseline without the
row cache or the vision change. To attribute the 3.1x properly, the next sweep needs
`mmap`-only-vs-baseline and `no-vision`-only-vs-baseline as separate arms.

## Files

- `sweep-A0.json` … `sweep-A3.json` — per-arm args, per-row tok/s, slots, VRAM free
- `sweep-A0.log` … `sweep-A3.log` — the engine's own logs (the source of every number above)
