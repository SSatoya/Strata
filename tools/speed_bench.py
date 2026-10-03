"""tools/speed_bench.py - decode throughput on this PC, cold and warm, from one engine.

The engine's `DONE` line already carries every number that matters, so this harness only starts the engine the
same way the service does (`serve.server.StrataEngine`, the config's own args) and repeats a fixed prompt set:

  long    a ~6,000-token prompt read from scratch (the batched prompt path, cold PLE row cache)
  long*   the same prompt again: the conversation cache reuses the tokens, so this is the decode-only number
  short   a short prompt after it, reusing the previous context (what a chat turn feels like)

Each round reports prompt tok/s, decode tok/s, the decode expert-cache hit rate, the draft acceptance and the
expert tiers (RAM blobs vs file blobs) - the four numbers that move together with speed on a MoE box.  Rounds
are separated because the expert tier and the row cache warm up: the first round is the cold machine, the last
is what the box feels like after it has been running.

    python tools/speed_bench.py strata-q2_0.json                     # the config as it stands
    python tools/speed_bench.py strata-q2_0.json --set --ple-gguf /opt/strata-ple/x.gguf   # an A/B arm
    python tools/speed_bench.py strata-q2_0.json --drop --vision --rounds 3 --out /tmp/a.json

`--set FLAG VALUE` replaces (or adds) an engine argument; `--drop FLAG` removes one.  Nothing is written to the
config: this measures, it does not tune (that is tools/calibrate.py).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

MAX_NEW = 128
ROUNDS = 2

# A fixed, shareable long prompt: one paragraph repeated until it reaches the target token count.  The paragraph
# is ordinary technical prose so the routing it produces is ordinary too (not a pathological token pattern).
PARAGRAPH = (
    "The engine keeps a bounded cache of the n-gram table's rows and reads the rest from the model file with "
    "unbuffered I/O, sixteen rows for every token it generates.  A row is ninety bytes, so a read is one page of "
    "the file, and the pages are scattered over twenty-eight gigabytes: the seek time of the drive, not its "
    "bandwidth, is what the token loop waits for.  The same is true of the experts that do not fit in VRAM.  "
    "They live in pinned host memory and are either computed on the CPU or copied to the GPU over PCIe, and the "
    "share that is copied is a machine-dependent setting that has to be measured on the machine it runs on.  "
)
LONG_TARGET = 6000
SHORT_PROMPT = "Translate into Japanese: the cache miss is the whole cost of the token."


def prompts() -> tuple[str, str]:
    long_text, n = "", 0
    while n < LONG_TARGET:
        long_text += PARAGRAPH
        n += len(long_text.split()) * 2          # cheap upper bound; the real count is taken after encoding
    return long_text, SHORT_PROMPT


def chat_ids(tok, text: str) -> list[int]:
    return tok.encode(f"<|im_start|>user\n{text}