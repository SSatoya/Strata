"""Verify the live server (started by run-q2_0.sh, already carrying --ple-row-cache 16777216).

No restart: this measures the config the user will actually run. Four unrelated ~1,400-token
prompts, so every round is cold text and cold PLE rows - the case that ran at 5.3 tok/s in the
browser. Decode tok/s is read from the engine log, not from wall time, so it excludes the HTTP
round trip.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path("/home/users/satoya_sugimoto/Workspace/Strata")
sys.path.insert(0, str(ROOT / "tools"))
from sweep_bench import DONE_RE, HIT_RE, MEM_RE, chat, health  # noqa: E402

PORT = 8080
LOG = ROOT / "strata-q2_0-fast.log"
CFG = ROOT / "strata-q2_0-fast.json"
MAX_TOKENS = 128

sys.path.insert(0, str(Path("/var/tmp/strata-ple")))
from sweep3 import _SEEDS, prompt_for  # noqa: E402


def tail_from(offset: int) -> tuple[str, int]:
    with LOG.open("r", errors="replace") as f:
        f.seek(offset)
        data = f.read()
        return data, f.tell()


def main() -> int:
    print(health(PORT))
    api_key = json.loads(CFG.read_text())["api_key"]
    offset = LOG.stat().st_size
    rows = []
    for i in range(len(_SEEDS)):
        name = _SEEDS[i][0]
        t0 = time.time()
        chat(PORT, [{"role": "user", "content": prompt_for(i)}], MAX_TOKENS, api_key)
        wall = time.time() - t0
        chunk, offset = tail_from(offset)
        gen = tok = hit = p_tok = None
        for line in chunk.splitlines():
            m = DONE_RE.search(line)
            if m:
                p_tok, gen, tok = float(m.group(5)), int(m.group(6)), float(m.group(8))
            h = HIT_RE.search(line)
            if h:
                hit = float(h.group(1))
            mm = MEM_RE.search(line)
            if mm:
                print(f"  mem: {mm.group(1)} MiB resident, {mm.group(2)} MiB in swap, "
                      f"{mm.group(3)} MiB available; {mm.group(4)} major faults")
        rows.append((name, gen, tok, hit, wall))
        print(f"  {name:28s} gen {gen:>4} tok  {tok:>5.1f} tok/s  "
              f"prompt {p_tok:>6.1f}  hit {hit}  wall {wall:.1f} s")
    decodes = [r[2] for r in rows if r[2]]
    print(f"\nDECODE mean {sum(decodes)/len(decodes):.1f} tok/s  max {max(decodes):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
