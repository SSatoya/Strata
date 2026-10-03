"""sweep3: the PLE n-gram table is the suspect, not the expert cache.

sweep2 reused ONE prompt pair, so after round 1 the table's rows and the conversation cache were warm and
decode read 31-33 tok/s.  The browser's real chats use fresh text every time: 5.3 tok/s.  So every arm here
gets a DIFFERENT ~1,400-token prompt per round (no reuse, cold rows), and the arms are the PLE I/O knobs:

  --ple-inflight N     outstanding SSD reads (default 64)
  --ple-row-cache N    fetched rows kept in RAM, 90 B each (default 1,048,576 = 90 MiB)
  --kv q4_0            frees ~1.1 GiB of VRAM -> ~850 more expert slots

--ple-io ram is NOT an arm: the table is 28.8 GB and the experts are 31.6 GB, which does not fit 62 GB.
"""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/users/satoya_sugimoto/Workspace/Strata")
sys.path.insert(0, str(ROOT / "tools"))
from sweep_bench import DONE_RE, HIT_RE, MEM_RE, SLOTS_RE, FREE_RE, health, chat  # noqa: E402

OUT = Path("/var/tmp/strata-ple")
PORT = 8080
ROUNDS = 3
MAX_TOKENS = 128

ARMS = [
    ("C0", []),
    ("C1", ["--ple-inflight", "256"]),
    ("C2", ["--ple-row-cache", "16777216"]),
    ("C3", ["--ple-inflight", "256", "--ple-row-cache", "16777216"]),
    ("C4", ["--kv", "q4_0"]),
]

# distinct ~1,400-token prompts: each round is cold text, like a real chat
_SEEDS = [
    ("payments ledger service", """You are reviewing a payments ledger service. Here is the code and the incident notes;
answer with the root cause and the fix.\n\n""" + "\n".join(
        f"def settle_{i}(account, amount, currency):\n"
        f"    if amount <= 0 or currency not in SUPPORTED:\n"
        f"        raise ValueError(f'bad settle_{i} argument for {{account}}')\n"
        f"    entry = LedgerEntry(account=account, amount=amount, currency=currency,\n"
        f"                        posted_at=clock.now(), idempotency_key=key_{i})\n"
        f"    with db.transaction(isolation='serializable') as tx:\n"
        f"        if tx.exists(entry.idempotency_key):\n"
        f"            return tx.lookup(entry.idempotency_key)\n"
        f"        tx.insert(entry)\n"
        f"        tx.debit(account, amount, reason='settle_{i}')\n"
        f"    return entry" for i in range(28)) + "\n\n"
     "Incident: 03:14 UTC the reconciliation job reported 41 double settlements on account ranges 900-999, "
     "and the retry counter for settle_7 climbed to 12,000. The ledger's own invariant check passed at 03:00 "
     "and failed at 03:20. Explain which invariant broke, why the idempotency key did not save us, and what "
     "single change to the code above removes the class of failure without dropping throughput."),
    ("19th century postal history", """Summarise and then evaluate the following passage about postal history;
keep your answer under 200 words and end with one open question.\n\n""" + " ".join(
        f"In {1840 + i}, the {['Penny Post','Uniform Penny Rate','Penny Postage Act','Parcel Post','Postcard','Money Order'][0] if i % 6 == 0 else 'district'} of {['LONDON','EDINBURGH','DUBLIN','CALCUTTA','SYDNEY','CAPE TOWN'][i % 6]} "
        f"recorded {1200 + 37 * i % 900:,} letters carried at a uniform rate of {1 + (i % 4)}d, with {3 + (i % 5)} "
        f"receiving offices and a {['single','double','treble'][i % 3]} handstamp; the surviving cover bears a "
        f"{['Maltese Cross','numeral','cds','ship mark'][i % 4]} in {['black','red','blue','green'][i % 4]} and the "
        f"backstamp of the {['GPO','PO','District Office','Sub-Office'][i % 4]} dated {(i % 28) + 1} "
        f"{['January','March','May','July','September','November'][i % 6]} {1840 + i}." for i in range(34)) +
     "\n\nWhat does this sequence of records tell us about the administrative reach of the uniform rate, and "
     "where would a historian expect the series to be unrepresentative?"),
    ("GPU memory hierarchy", """Explain the memory hierarchy of a modern consumer GPU for someone who knows C++
but has never written a CUDA kernel. Use the numbers below as the worked example.\n\n""" + "\n".join(
        f"kernel_{i}: reads {64 * (i + 1)} KB of weights per token, {2 * (i + 1)} KB of activations, "
        f"writes {i + 1} KB; L1 hit {55 + (i % 40)}%, occupancy {25 + (i % 50)}%, "
        f"achieved bandwidth {40 + (i % 180)} GB/s of 300 GB/s peak." for i in range(30)) +
     "\n\nWhy do the small kernels sit so far below peak, and which of the three fixes (fusion, larger tiles, "
     "persistent kernels) would you try first and why?"),
    ("supply chain model", """Build a small model of this supply chain and tell me the bottleneck.\n\n""" + "\n".join(
        f"node {i}: {['factory','warehouse','port','depot','retail'][i % 5]}, lead time {2 + (i % 9)} days, "
        f"capacity {500 + 120 * (i % 7)} units/day, defect rate {i % 5}%, buffer {1 + (i % 4)} days."
        for i in range(32)) +
     "\n\nGive the throughput of the chain as it stands, the single node whose capacity increase buys the most, "
     "and what happens to the answer if every lead time doubles."),
    ("jazz theory", """Analyse the following chord changes in C minor: name the functions, the voice leading,
and one reharmonisation.\n\n""" + " | ".join(
        f"{['Cm7','Fm7','Bb7','Ebmaj7','Ab7','G7alt','Ddim7','Abmaj7','Fm7b5','Cm7','Gm7','C7'][i % 12]} "
        f"(bar {i + 1}, {['4/4','3/4','4/4','6/8'][i % 4]}, tempo {120 + 3 * (i % 20)})" for i in range(30)) +
     "\n\nWhere does the progression leave C minor, and what is the earliest bar where a tritone substitution "
     "would be idiomatic?"),
    ("database vacuum", """Postgres is bloating. Diagnose from this output and give the fix.\n\n""" + "\n".join(
        f"table {i}: rows {100000 * (i + 1):,}, dead {10000 * (i % 9):,}, bloat {5 + 3 * (i % 12)}%, "
        f"autovacuum {'on' if i % 3 else 'off'}, last vacuum {i % 30} days ago, toast {'yes' if i % 4 == 0 else 'no'}."
        for i in range(26)) +
     "\n\nWhich tables need intervention now, which settings are wrong, and what is the risk of the fix you propose?"),
]


def prompt_for(round_index: int) -> str:
    """A different long prompt each round, so nothing is warm and the cache reuses nothing."""
    name, body = _SEEDS[round_index % len(_SEEDS)]
    return f"[{name}] {body}\n\n(round {round_index + 1}: answer directly, no preamble)"


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
    log = OUT / f"sweep3-{name}.log"
    cfg["log"] = str(log)
    if log.exists():
        log.unlink()
    tmp = OUT / f"sweep3-{name}.json"
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
            chat(PORT, [{"role": "user", "content": prompt_for(r)}], MAX_TOKENS, api_key)
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
    (OUT / f"sweep3-{name}.json").write_text(json.dumps(summary, indent=1))
    print(f"[{name}] slots {slots} vram free {free} mem {last_mem}", flush=True)
    for row in rows:
        print(f"[{name}]   gen {row['gen_tok']:4d} tok {row['gen_tps']:6.1f} tok/s "
              f"(prompt {row['prompt_tps']:7.1f}, reused {row['reused']}, hit {row['hit']}, acc {row['acc']})",
              flush=True)
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
    print("\n=== SUMMARY (cold prompts) ===")
    for r in results:
        print(f"{r['name']:4s} {' '.join(r['extra']):34s} mean {r['decode_mean']}  max {r['decode_max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
