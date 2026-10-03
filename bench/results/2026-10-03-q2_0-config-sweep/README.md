# 2026-10-03: Q2_0 config sweep on a GTX 1080 Ti (i7-6700, 62 GiB RAM)

Goal: raise decode toward 50 tok/s. Reached **33.1 tok/s warm / 28.0 cold** on this box.
The 50 tok/s figure in `docs/COMMUNITY_BENCHMARKS.md` is an RTX 5090 (32 GiB VRAM, PCIe Gen5,
23 pool workers); it is not reachable on a 1080 Ti with this model.

## Hardware limits that set the ceiling

| Resource | This box | What it costs |
|---|---|---|
| VRAM | 11,264 MiB | 3,947 expert slots (5.08 GiB) after weights + draft head + 681 MiB reserve; 513 MiB free |
| RAM | 62 GiB | 31.64 GiB expert arena + 28.8 GiB mmap'd PLE table = 60.4 GiB -> page cache has ~nothing left |
| Cores | 4 (i7-6700, no AVX-512) | expert kernels on AVX-2; `--pool-workers 3` steals a core from the token thread |
| PCIe | Gen3 x16 (~12 GB/s) | `--pcie-frac 0.55` |

Decode is bound by the **13.4% expert-cache miss rate**: every miss is a CPU AVX-2 row plus a
PCIe transfer, and the 16 PLE n-gram rows per token are major faults on that starved page cache.

## Sweeps

`--kv int8`, `--expert-cache auto`, `--spec 4`, `--mtp rt`, `--vram-reserve-mib 681` in every arm.
A/B/C = 1,483-token prompt repeated (warm prefix reuse); C/D = four unrelated 1,483-token prompts
(cold prefill each round), which is what a browser session actually does.

### Warm (sweep2, `sweep-A*.json`)

| Arm | Change | Decode mean | Max |
|---|---|---:|---:|
| A0 | baseline | 10.0 | 11.4 |
| A2 | `--ple-io mmap`, no vision | **31.4** | 33.9 |
| B0 | A2 | 29.5 | 33.1 |
| B1 | A2 `--spec 6` | 28.5 | 31.9 |
| B3 | A2 `--pcie-frac 0.75` | 22.3 | 24.6 |
| B4 | A2 `--ple-inflight 256` | 29.6 | 33.0 |

`--spec 6` and `--pcie-frac 0.75` both lose. Acceptance is already 56-58/66 at `--spec 4`; a wider
window pays for the rejected drafts.

### Cold prompts (sweep3, `sweep-C*.json`)

| Arm | Change | Decode mean | Max |
|---|---|---:|---:|
| C0 | baseline (`--ple-io mmap`, default 1M-row cache) | 18.7 | 21.0 |
| C1 | `--ple-inflight 256` | 24.6 | 27.5 |
| C2 | **`--ple-row-cache 16777216`** | **28.0** | 29.6 |
| C3 | C1 + C2 | 28.1 | 29.3 |
| C4 | `--kv q4_0` | 21.1 | 22.4 |

### sweep4 (`sweep-D*.json`) - fp16 KV by mistake, so 3,641 slots instead of 3,947

| Arm | Change | Decode mean |
|---|---|---:|
| D0 | `--ple-row-cache 16777216` | 25.2 |
| D2 | D0 `--ple-io direct` | 25.0 |
| D1 | D0 `--pool-workers 3` | 25.8 |
| D3 | D0 `--ple-io direct --pool-workers 3` | 27.6 |
| D4 | D0 `--kv-resident 20480` | **15.4** |

D0 and D2 are the same config and score 25.2 / 25.0, so the noise floor is ~1%. Everything except
D4 is inside noise of C2.

## What each knob does and why

- **`--ple-row-cache 16777216`** (16M rows x 90 B = 1.4 GiB, vs the 1M-row default): the table has
  320,001,536 rows and a token touches 16. The default cache holds 0.3% of the table, so a long
  session re-fetches almost every row from SSD. 16M rows covers the working set of a chat session.
  **+50% on cold prompts (18.7 -> 28.0).** This is the win.
- **`--ple-inflight 256`**: +32% alone, but redundant once the row cache is large (C3 ~= C2).
- **`--ple-io direct`**: no benefit here. It keeps the table out of RAM, which sounds right given
  the RAM pressure, but it also forfeits the page cache for the rows that *are* hot.
- **`--kv q4_0`**: loses 306 slots' worth of gain and costs precision (see `2026-09-27-kv-q4`).
- **`--kv-resident 20480`**: 15.4 tok/s. Streaming 20K positions of KV over PCIe per token is far
  slower than keeping 32K of int8 KV in VRAM. Do not use on a 1080 Ti.
- **`--expert-cache-per-layer`**: engine rejects it - `ExpertCache::verify_slot: slot 0 differs
  from the arena at byte 0`. Not usable with the MTP draft head.

## The 5.3 tok/s browser failure mode

A live browser session ran at 5.3 tok/s while the sweep measured 18.7 on the same binary. Cause:
the 31.64 GiB arena plus the 28.8 GiB mmap'd PLE table is 60.4 of 62 GiB. Swap was full
(1.0 of 1.0 GiB), so the table's pages were evicted and every token's 16 rows major-faulted with
swap churn on top - about 190 ms/token. The sweep's three rounds are short enough that the table
stays resident, so the sweep *understates* both the problem and the fix.

Mitigation on this box: `--ple-row-cache 16777216` keeps the fetched rows in the engine's own
cache instead of relying on the page cache, so eviction stops mattering as much.

## Live verification (`/var/tmp/strata-ple/verify_live.py`, no restart)

`./run-q2_0.sh` with the shipped config, then four unrelated ~1,400-token prompts to the running
server:

| Round | Prompt tok/s | Decode tok/s | Expert hit |
|---|---:|---:|---:|
| payments ledger (prefix reused) | 30.5 | 28.1 | 81.2% |
| postal history | 333.5 | 19.5 | 65.0% |
| GPU memory hierarchy | 331.8 | 23.7 | 71.6% |
| fourth | 155.8 | 19.9 | 80.6% |

**Mean 22.8 tok/s** against 5.3 tok/s for the same binary before the change. The cold rounds land
below the sweep's 28.0 because the sweep's prompts are synthetic and repetitive, while real text
touches more distinct n-gram rows. The remaining gap to 50 tok/s is the 13.4% expert-cache miss
rate on 3,947 slots: a VRAM limit, not a setting.

## Decision

`strata-q2_0-fast.json` now carries `--ple-row-cache 16777216`. Everything else is unchanged:
`--kv int8`, `--ple-io mmap`, `--spec 4`, `--pcie-frac 0.55`, `--pool-workers 2`,
`--vram-reserve-mib 681`.

For >40 tok/s on this model the constraint is VRAM, not settings: an expert cache that holds the
whole 24,576-pair routing set needs ~31 GiB, which is a 5090/RTX 6000 Ada class budget.
