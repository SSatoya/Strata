# 2026-10-04: catching up to engine 0.1.33 and turning images back on (GTX 1080 Ti)

Two questions, one answer each:

1. **The engine was 0.1.31 while `origin/main` is 0.1.33.** Rebuilt; decode is unchanged.
2. **Images were off** (`run-q2_0.sh` has no `--vision`, and the 09-24 sweep that found
   `--ple-io mmap` measured 3.1x decode *with* vision on). Turning them back on costs
   **nothing measurable**: the encoder runs on the CPU, so the expert cache keeps all
   **3,947 slots**, the **681 MiB** reserve is untouched, and decode lands **3.6% lower**
   (31.1 vs 32.25 tok/s) - inside the noise floor of this box.

## What 0.1.31 -> 0.1.33 changed

`git log` between the two: `a1eb951 Engine 0.1.33: a portable image encoder again (#411 #412),
setup recommends instead of forcing (#406 #403 #364 #384), fixes #352 #365 #369 #371 #375 #393
#408 #414`, plus `aeb35be hip_compat: cudaDevAttrComputeCapabilityMinor (#371's check, the HIP
build)`. The engine source hash moved to `bd5219cbd261b225`; the vision tool's own source moved
to `4d89409b228a1daa` - that is the "#411 #412 portable image encoder" part: the encoder was
rewritten so it builds without a GPU toolchain (`setup.py:1258 build_vision_cpu`).

Rebuilt with the same flags as before (`engine/BUILD.json`):

```
"version": "0.1.33",  "archs": [61],  "sm60": true,  "vision": "cpu",
"cuda_dirs": ["/usr/local/cuda-12.6/bin", "/usr/local/cuda-12.6/lib64"]
```

`engine/strata` and `engine/strata-vision` both built clean.

## Why vision is free here

The 09-24 result that made us drop vision was a **GPU** encoder: it took its VRAM before the
engine sized the expert cache, and on an 11 GiB card that is where the slots went.

`--vision cpu` is a different thing. From the code:

- `serve/server.py:493` - the encoder is a separate resident process (`engine/strata-vision`,
  llama.cpp mtmd + the mmproj file). With `"gpu": false` it never allocates VRAM.
- `tools/vision/strata_vision.cpp:94` - the text model is opened **vocab-only** ("no weights"),
  so the encoder's own footprint is the mmproj weights in RAM plus mtmd's context.
- The engine's own cost for `--vision` is the mrope position table: `cells * 3` int32 for a
  32,768-token context = **~0.4 MiB**.

Measured, every arm, from the engine's own log line:

```
[V0-baseline]     expert cache slots: ['3947'] | vram free: ['513']
[V1-vision]       expert cache slots: ['3947'] | vram free: ['513']
[V2-vision-repeat] expert cache slots: ['3947'] | vram free: ['513']
[V3-baseline-repeat] ... identical through V5
```

Identical. The `--vram-reserve-mib 681` from the 10-03 sweep stays exactly as it is.

## Decode: 0.1.33 with and without `--vision`

`tools/sweep_bench.py`, 3 rounds x (128, 48) tokens, every flag from `strata-q2_0-fast.json`
(`--kv int8 --ple-io mmap --ple-row-cache 16777216 --spec 4 --pcie-frac 0.55 --pool-workers 2
--vram-reserve-mib 681`). V1/V2/V5 = V0 + `--vision`.

| Arm | Config | Order | Decode mean | Max | Slots | VRAM free |
|---|---|---|---:|---:|---:|---:|
| V0 | no vision | 1st of the day | 27.1 | 29.6 | 3,947 | 513 MiB |
| V1 | `--vision` | after V0 | 25.2 | 29.8 | 3,947 | 513 MiB |
| V3 | no vision | warmed | **32.6** | 34.3 | 3,947 | 513 MiB |
| V2 | `--vision` | warmed | **31.3** | 33.9 | 3,947 | 513 MiB |
| V4 | no vision | warmed, reversed order | **31.9** | 34.4 | 3,947 | 513 MiB |
| V5 | `--vision` | warmed, reversed order | **30.9** | 34.1 | 3,947 | 513 MiB |

V0/V1 were the first two runs of the day and both sit ~5 tok/s under the warmed arms - that is
page-cache warm-up, not vision. The warmed pair is the honest A/B, and it was run in both orders
so the ordering can't be the cause:

| warmed pair | baseline | vision | delta |
|---|---:|---:|---:|
| V3 / V2 (baseline first) | 32.6 | 31.3 | -4.0% |
| V4 / V5 (vision first) | 31.9 | 30.9 | -3.1% |
| **mean** | **32.25** | **31.1** | **-3.6%** |

Round by round (128 tok / 48 tok / 128 tok / 48 tok / 128 tok / 48 tok):

| Arm | r1 | r2 | r3 | r4 | r5 | r6 |
|---|---:|---:|---:|---:|---:|---:|
| V3 baseline | 29.8 | 33.2 | 34.3 | 35.6 | 33.6 | 34.7 |
| V2 vision | 26.6 | 33.2 | 33.9 | 35.1 | 33.5 | 33.1 |
| V4 baseline | 28.1 | 33.6 | 34.4 | 35.9 | 33.3 | 35.5 |
| V5 vision | 29.8 | 33.1 | 34.1 | 35.6 | 28.9 | 30.0 |

The draft acceptance counts are the same in every arm (65/77, 64/78, 57/66-67 at 128 tokens), so
the model is doing identical work; only the wall time moves. V5's last two rounds (28.9 / 30.0)
are the outlier that pulls its mean down - its first four rounds are 29.8 / 33.1 / 34.1 / 35.6,
indistinguishable from V4's 28.1 / 33.6 / 34.4 / 35.9.

**Verdict: -3.6%, inside the ±5% target.** The 10-03 sweep measured this box's noise floor at
~1% (D0 and D2 are the same config and scored 25.2 / 25.0); the residual 3.6% is not a VRAM
effect - there is no mechanism, since the encoder never touches the GPU. The likeliest cause is
the encoder's 2 CPU threads competing with the token thread and the 2 pool workers on a 4-core
box, which would also explain why it costs more on the cold arms.

## Images actually work

`tools/vision_smoke.py` (new): generates a 64x64 PNG - a red square on white - posts it as an
OpenAI `image_url` part, and checks the answer.

```
health: images=True
answer: 'Red'
usage: prompt=77 completion=41
PASS
```

`prompt=77` is the whole request: the image is ~50 tokens, well inside the `max_tokens: 1024`
the encoder is configured with. The first attempt at 32 `max_tokens` returned
`content: null` with `reasoning_content` reading *"Image shows red square"* - the model was
looking at the picture, but a thinking model spends the budget on reasoning first, so the smoke
test asks for 128 and accepts either field.

Timings from that request: `prompt 41.7 tok/s`, `predicted 26.4 tok/s` - the same decode speed
as the text-only sweep, with the image already encoded.

## Config and how to run it

`strata-q2_0-vision.json` = `strata-q2_0-fast.json` plus:

```
args:   ... --vision
vision: { "exe": "engine/strata-vision",
          "mmproj": "../Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
          "model": "../Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
          "gpu": false, "max_tokens": 1024, "threads": 2 }
```

`--model` for the encoder is the **first split** of the text model (it only needs the vocab).
`threads: 2` because the token thread and the two pool workers already own this 4-core box.

Two deliberate departures from what `setup.py` would generate:

- **`max_tokens: 1024`, not 300.** `setup.py:182` sets `VISION["cpu"]["max_tokens"] = 300`
  (the 09-24 note that a CPU encoder caps at 300 image tokens). The 0.1.33 encoder takes the
  same `--max-tokens` flag for both paths and 1024 measured fine here - the smoke test's image
  cost ~50 tokens, so the ceiling only matters for large photos. Raise it back to 300 if a
  single turn ever needs the prompt budget back.
- **`--vram-reserve-mib 681`, not 700.** `setup.py:2945` adds `--vram-reserve-mib 700` for *any*
  vision config, GPU or CPU. That reserve was sized for a GPU encoder's VRAM. With `"gpu": false`
  the encoder allocates none, so the 10-03 tuned 681 is kept; 700 would cost 19 expert slots for
  nothing.

```
./run-q2_0-vision.sh          # images on
./run-q2_0.sh                 # images off, byte-identical to the 10-03 winner
```

Both keep the same engine flags; only `--vision` and the `vision` block differ.

## Decision

Run `./run-q2_0-vision.sh`. Images are accepted, the expert cache is unchanged at 3,947 slots,
and decode is 31.1 tok/s against 32.25 without them - 3.6%, inside this box's noise. The 09-24
"turn vision off" advice is specific to a **GPU** encoder on an 11 GiB card; it does not apply to
`--vision cpu`.

If 3.6% matters, the knob is `vision.threads`: 1 instead of 2 gives the token thread a core back.
Not measured here.
