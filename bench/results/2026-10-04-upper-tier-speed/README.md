# 2026-10-04: can the upper quant tiers match Q2_0's speed? (GTX 1080 Ti)

Question: after the 10-03 config sweep put Q2_0 at **31.4 tok/s warm / 28.0 cold**, do the
higher-quality tiers (IQ2_XS, IQ3_XXS) reach the same speed on this box?

**Answer: no.** Speed falls monotonically with quant size, and the gap is structural, not a
settings artifact. Measured on this box with the 10-03 winner flags held identical:

| Tier | Decode mean | Max | vs Q2_0 | Expert slots | VRAM free | Expert arena |
|---|---:|---:|---:|---:|---:|---:|
| **Q2_0** (10-03 winner, warm) | **31.4** | 33.9 | — | 3,947 | 513 MiB | 31.64 GiB |
| **Q2_0 + `--vision`** (10-04 warmed) | 31.1 | 34.1 | -1% | 3,947 | 513 MiB | 31.64 GiB |
| **IQ2_XS** (tuned here) | **24.2** | 25.9 | **-23%** | 3,829 | 513 MiB | 33.02 GiB |
| **IQ3_XXS** (tuned here) | **19.6** | 23.2 | **-38%** | 2,808 | 499 MiB | 39.97 GiB |

IQ3_S and Q4_K_S were not measured: no pack exists on this box and the memory math rules them
out (see "Q4 and above").

## How it was measured

Same harness, same flags, same box as `2026-10-03-q2_0-config-sweep`. `tools/sweep_bench.py`,
3 rounds x (128, 48) generated tokens, four unrelated ~1,483-token prompts per round (cold
prefill each round, which is what a browser session does). Every arm carries:

```
--kv int8 --expert-cache auto --spec 4 --mtp rt --max-context 32768
--pcie-frac 0.55 --spec-min-p 0.70 --pool-workers 2
--ple-io mmap --ple-row-cache 16777216 --vram-reserve-mib 681
```

Configs: `strata-iq2_xs-fast.json`, `strata-iq3_xxs-fast.json` (this repo, added today).
Per-arm JSON: `sweep-iq2xs-tuned.json`, `sweep-iq3xxs-tuned.json`. The Q2_0 rows are quoted
from `2026-10-03-q2_0-config-sweep` and `2026-10-04-vision-catchup`, which used these same flags.

Round-by-round decode (128 / 48 / 128 / 48 / 128 / 48 tokens):

| Arm | r1 | r2 | r3 | r4 | r5 | r6 |
|---|---:|---:|---:|---:|---:|---:|
| IQ2_XS | 21.0 | 28.0 | 25.6 | 29.9 | 25.9 | 32.1 |
| IQ3_XXS | 16.1 | 21.9 | 19.6 | 19.4 | 23.2 | 25.7 |

Both arms warm up across the three rounds (page cache + expert cache), so the last rounds are
the best case for them and they still do not reach Q2_0's mean.

## Why the upper tiers lose: three compounding limits

### 1. Fewer expert slots for the same VRAM

The expert cache is sized from free VRAM, and a slot is one expert blob. From the engine logs:

| Tier | Largest blob | Auto sizing said | Slots actually used | VRAM held |
|---|---:|---:|---:|---:|
| Q2_0 | 1.38 MB | 3,947 | 3,947 | 5.08 GiB |
| IQ2_XS | 1.51 MB | 3,657 | 3,829 | 5.14 GiB |
| IQ3_XXS | 2.33 MB | 2,120 | 2,808 | 4.60 GiB |

The routing profile ranks 24,576 expert pairs. Q2_0 caches 16% of them, IQ2_XS 16%, IQ3_XXS
11%. The 10-03 sweep established that Q2_0's ceiling on this box is its **13.4% miss rate** on
3,947 slots — every miss is a CPU AVX-2 row plus a PCIe Gen3 transfer. IQ3_XXS starts from a
smaller cache, so it misses more often and pays that cost more often.

### 2. More CPU work per expert row

Every arm logs:

```
this CPU has no AVX-512: the expert kernels run on AVX-2 (multi-token for the i-quant gate/up rows)
```

The i-quants (IQ2_XS, IQ3_XXS) decode to more work per row than Q2_0 on AVX-2. This is the
dominant term for IQ2_XS: it got **more** slots than its auto sizing predicted (3,829 vs 3,657)
and still lost 23% to Q2_0 at 3,947 slots. Slot count alone does not explain the gap; the
dequant cost does.

### 3. RAM runs out before VRAM does

| Tier | Expert arena | PLE table (mmap) | Total | Page cache left of 62 GiB |
|---|---:|---:|---:|---:|
| Q2_0 | 31.64 GiB | 28.8 GiB | 60.4 GiB | ~1.6 GiB |
| IQ2_XS | 33.02 GiB | 28.8 GiB | 61.8 GiB | ~0.2 GiB |
| IQ3_XXS | 39.97 GiB | 28.8 GiB | **68.8 GiB** | **negative** |

IQ3_XXS's arena plus the PLE table exceeds installed RAM. That is visible in the engine's own
sizing: it found **5.45 GiB** free for IQ3_XXS against **5.95 GiB** for IQ2_XS, because the
page cache was already squeezed. The 10-03 report documented the failure mode this produces in
a live session — a 31.64 + 28.8 GiB footprint drove swap full and the running server fell to
**5.3 tok/s** while a short sweep still read 18.7. IQ3_XXS sits further into that regime, so
the 19.6 tok/s measured here is optimistic for a real browser session.

## The ordering is intrinsic, not local

`docs/DETAILS.md` publishes the same ladder on a much faster machine (RTX 5070 12 GB, Ryzen 5
7600 6 cores, 64 GB DDR5-5200, engine 0.1.26, 4K prompt):

| Model | Output tok/s (4K) | vs Q2_0 |
|---|---:|---:|
| Q2_0 | 93.0 | — |
| IQ2_XS | 78.6 | -15% |
| IQ3_XXS | 61.6 | -34% |
| IQ3_S | 53.3 | -43% |

Same shape as this box (-23% / -38%). The gap narrows somewhat on a machine with more cores and
faster RAM, but it never closes: the published table's own note is that IQ3_XXS is "slower
(more CPU work)".

## What the 5090's 179 tok/s actually is

`docs/COMMUNITY_BENCHMARKS.md` and `bench/results/2026-09-30-community-rtx-5090` report
**179.4 / 175.7 / 165.0 tok/s** — but that is **IQ2_XS**, not an upper tier, and it is a
different class of machine:

- 32,607 MiB VRAM: **17,463 expert slots, 23.44 GiB**, profile-prefilled, hit rates
  **97.8-99.7%**. That is 71% of the 24,576-pair routing set cached, versus 16% here.
- PCIe Gen5 and 23 pool workers, against Gen3 x16 and 2 workers on 4 cores here.

So the 5090 result is the same tier with the miss rate removed. It is not evidence that a
heavier quant can match a lighter one; it is evidence that the miss rate is the thing to remove.

## Q4 and above

No Q4_K_S or higher pack exists on this box, and the memory math rules them out here:

- `docs/ORCA_Q4_K_S.md` describes a manual path over **112 GB** of Q4_K_S shards, with a
  separate opt-in build (`STRATA_ORCA_Q4KS_MMQ=ON`) and a ~1.43 GiB dense pack.
- The GPU expert kernels take Q4_K with Q5_1 or Q8_0 downs; **Q5_0 downs are not supported yet**
  and the engine refuses to start on a pack with such a layer.
- Experts that large exceed both the 11 GiB of VRAM and the 62 GiB of RAM on this box.

## Decision

Keep **Q2_0** as this box's model. `./run-q2_0-vision.sh` remains the recommended launch:
images on, 3,947 slots, 31.1 tok/s (10-04).

To run an upper tier at Q2_0-equivalent speed the requirement is hardware, not flags:

- **VRAM ≥ 24 GiB** to hold the whole 24,576-pair routing set (~31 GiB of expert cache at
  Q2_0's slot size, more at heavier quants). That is a 5090 / RTX 6000 Ada class budget, as the
  10-03 report concluded.
- **More cores and AVX-512** so the i-quant row decode is not the bottleneck, plus faster RAM
  and PCIe Gen5 for the misses that remain.
- **RAM ≥ 96 GiB** for IQ3_XXS and above, whose arena alone is 40+ GiB before the PLE table.

On this box the quality/speed trade is fixed: IQ2_XS costs 23% for "a bit better" quality,
IQ3_XXS costs 38% for "best" quality, and IQ3_XXS additionally risks the RAM-eviction failure
mode in long sessions.
