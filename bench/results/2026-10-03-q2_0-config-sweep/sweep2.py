"""Sequential arms on top of the fixed strata-q2_0-fast.json (PLE on NVMe).

Same measurement contract as tools/sweep_bench.py (fixed prompt pair, 3 rounds, the
engine's own tok/s lines), but arms may add bare boolean flags, not just FLAG=VALUE.
"""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/users/satoya_sugimoto/Workspace/Strata")
sys.path.insert(0, str(ROOT / "tools"))
from sweep_bench import LONG, SHORT, DONE_RE, HIT_RE, MEM_RE, SLOTS_RE, FREE_RE, health, chat  # noqa: E402

OUT = Path("/var/tmp/strata-ple")
PORT = 8080
ROUNDS = 3
MAX_TOKENS = 128

ARMS = [
    ("B0", []),
    ("B1", ["--spec", "6"]),
    ("B2", ["--expert-cache-per-layer"]),
    ("B3", ["--pcie-frac", "0.75"]),
    ("B4", ["--suffix-draft", "8"]),
]


def run_arm(name: str, extra: list[str]) -> dict | None:
    cfg = json.loads((ROOT / "strata-q2_0-fast.json").read_text())
    args = list(cfg["args"])
    for flag in extra:
        while flag in args:
            i = args.index(flag)
            args.pop(i)
            if i < len(args) and not args[i].startswith("--"):
                args.pop(i)
    args += extra
    cfg["args"] = args
    log = OUT / f"sweep-{name}.log"
    cfg["log"] = str(log)
    if log.exists():
        log.unlink()
    tmp = OUT / f"sweep-{name}.json"
    tmp.write_text(json.dumps(cfg, indent=1))
    print(f"[{name}] args: {' '.join(args)}", flush=True)

    proc = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / "serve/server.py"),
                             "--engine", "strata", "--config", str(tmp), "--port", str(PORT)],
                            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    try:
        while True:
            try:
                health(PORT)
                break
            except Exception:
                if proc.poll() is not None:
                    print(f"[{name}] server exited (code {proc.returncode}) after {time.time() - t0:.0f} s",
                          flush=True)
                    return None
                if time.time() - t0 > 1800:
                    print(f"[{name}] TIMEOUT after {time.time() - t0:.0f} s", flush=True)
                    return None
                time.sleep(5)
        print(f"[{name}] up in {time.time() - t0:.0f} s", flush=True)
        api_key = cfg.get("api_key", "")
        for r in range(ROUNDS):
            wall = time.time()
            msgs = [{"role": "user", "content": LONG}]
            chat(PORT, msgs, MAX_TOKENS, api_key)
            msgs.append({"role": "user", "content": SHORT})
            chat(PORT, msgs, 48, api_key)
            print(f"[{name}] round {r + 1} wall {time.time() - wall:.1f} s", flush=True)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=60)
    time.sleep(5)

    text = log.read_text(errors="replace") if log.exists() else ""
    rows, last_hit, last_mem = [], None, None
    for line in text.splitlines():
        m = DONE_RE.search(line)
        if m:
            rows.append({"prompt_tok": int(m.group(1)), "reused": int(m.group(2)),
                         "prompt_tps": float(m.group(5)), "gen_tok": int(m.group(6)),
                         "gen_ms": int(m.group(7)), "gen_tps": float(m.group(8)),
                         "acc": f"{m.group(9)}/{m.group(10)}", "hit": last_hit})
        h = HIT_RE.search(line)
        if h:
            last_hit = float(h.group(1))
        mm = MEM_RE.search(line)
        if mm:
            last_mem = {"resident_mib": int(mm.group(1)), "swap_mib": int(mm.group(2)),
                        "avail_mib": int(mm.group(3)), "major_faults": int(mm.group(4))}
    slots = SLOTS_RE.findall(text)
    free = FREE_RE.findall(text)
    dec = [r["gen_tps"] for r in rows if r["gen_tok"] >= 64]
    summary = {"name": name, "extra": extra, "rows": rows, "slots": slots,
               "vram_free": free, "mem": last_mem,
               "decode_mean": round(sum(dec) / len(dec), 1) if dec else None,
               "decode_max": max(dec) if dec else None}
    (OUT / f"sweep-{name}.json").write_text(json.dumps(summary, indent=1))
    print(f"[{name}] slots {slots} vram free {free} mem {last_mem}", flush=True)
    for row in rows:
        print(f"[{name}]   gen {row['gen_tok']:4d} tok {row['gen_tps']:6.1f} tok/s "
              f"(prompt {row['prompt_tps']:7.1f}, hit {row['hit']}, acc {row['acc']})", flush=True)
    print(f"[{name}] DECODE mean {summary['decode_mean']} tok/s  max {summary['decode_max']}", flush=True)
    return summary


def main() -> int:
    only = set(sys.argv[1:])
    results = []
    for name, extra in ARMS:
        if only and name not in only:
            continue
        r = run_arm(name, extra)
        if r:
            results.append(r)
    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['name']:4s} {' '.join(r['extra']):30s} mean {r['decode_mean']}  max {r['decode_max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
