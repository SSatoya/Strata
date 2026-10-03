"""Measure the ALREADY-RUNNING engine on a port, using sweep_bench's exact prompt pair.

The sweep harness starts its own engine; two engines cannot both hold 38 GB of experts
plus the GPU, so this drives the live production server instead and parses the engine's
own log lines (the same DONE_RE the harness uses).
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/users/satoya_sugimoto/Workspace/Strata")
sys.path.insert(0, str(ROOT / "tools"))
from sweep_bench import LONG, SHORT, DONE_RE, HIT_RE, MEM_RE  # noqa: E402

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
rounds_n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 128
log_path = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "strata-q2_0-fast.log"
cfg = json.loads((ROOT / "strata-q2_0-fast.json").read_text())
api_key = cfg.get("api_key", "")

offset = log_path.stat().st_size if log_path.exists() else 0


def chat(msgs, cap):
    body = json.dumps({"model": "m", "messages": msgs, "max_tokens": cap,
                       "temperature": 0.0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        json.loads(resp.read().decode())


t0 = time.time()
for r in range(rounds_n):
    msgs = [{"role": "user", "content": LONG}]
    wall = time.time()
    chat(msgs, max_tokens)
    msgs.append({"role": "user", "content": SHORT})
    chat(msgs, 48)
    print(f"[measure] round {r + 1} wall {time.time() - wall:.1f} s "
          f"(elapsed {time.time() - t0:.0f} s)", flush=True)

time.sleep(2)
text = log_path.read_text(errors="replace")[offset:]
rows, last_hit, last_mem = [], None, None
for line in text.splitlines():
    m = DONE_RE.search(line)
    if m:
        rows.append({"prompt_tok": int(m.group(1)), "reused": int(m.group(2)),
                     "prompt_tps": float(m.group(5)), "gen_tok": int(m.group(6)),
                     "gen_tps": float(m.group(8)), "acc": f"{m.group(9)}/{m.group(10)}",
                     "hit": last_hit})
    h = HIT_RE.search(line)
    if h:
        last_hit = float(h.group(1))
    mm = MEM_RE.search(line)
    if mm:
        last_mem = {"resident_mib": int(mm.group(1)), "swap_mib": int(mm.group(2)),
                    "avail_mib": int(mm.group(3)), "major_faults": int(mm.group(4))}

for row in rows:
    print(f"[measure]   gen {row['gen_tok']:4d} tok {row['gen_tps']:6.1f} tok/s "
          f"(prompt {row['prompt_tps']:7.1f}, hit {row['hit']}, acc {row['acc']})", flush=True)
print(f"[measure] memory: {last_mem}", flush=True)
dec = [r["gen_tps"] for r in rows if r["gen_tok"] >= 64]
if dec:
    print(f"[measure] DECODE mean {sum(dec) / len(dec):.1f} tok/s  max {max(dec):.1f}", flush=True)
