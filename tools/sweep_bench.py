"""tools/sweep_bench.py - one-shot A/B harness for the engine on this PC.

Starts `serve/server.py` with a config whose `args` have been edited, waits for the engine,
repeats a fixed prompt set, and prints the decode numbers parsed from the engine's own
`strata serve: prompt ... generated in ... (N tok/s)` lines.  Nothing is written to the
real config; each arm gets a temp config.

    python tools/sweep_bench.py strata-q2_0.json --name A0
    python tools/sweep_bench.py strata-q2_0.json --name A1 \
        --set --ple-io mmap --set --ple-row-cache-mib 16384
    python tools/sweep_bench.py strata-q2_0.json --name A2 --drop --vision --set --vram-reserve-mib 0
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DONE_RE = re.compile(
    r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)"
)
HIT_RE = re.compile(r"decode expert cache hit rate: ([\d.]+)%")
MEM_RE = re.compile(r"memory: (\d+) MiB resident, (\d+) MiB in swap, (\d+) MiB RAM available; (\d+) major page faults")
SLOTS_RE = re.compile(r"expert cache (\d+) slots")
FREE_RE = re.compile(r"(\d+) MiB of VRAM free")

LONG = ("Explain, in detail and with worked examples, how a mixture-of-experts transformer routes "
        "tokens, how the router is trained, and what the practical failure modes are when the expert "
        "weights do not fit in GPU memory. Cover load balancing, auxiliary losses, expert parallelism, "
        "offloading strategies, and the trade-offs between them. ") * 22
SHORT = "Summarize the three most important points in five bullet points."


def _preparse(argv: list[str]) -> list[str]:
    """argparse refuses to read a value that starts with `--`, so `--set --ple-io mmap` dies with
    "expected 2 arguments".  Rewrite the space-separated form into the `=` form before argparse
    sees it: `--set --ple-io mmap` -> `--set=--ple-io=mmap`, `--drop --vision` -> `--drop=--vision`."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--set" and i + 2 < len(argv):
            out.append(f"--set={argv[i + 1]}={argv[i + 2]}")
            i += 3
        elif a == "--drop" and i + 1 < len(argv):
            out.append(f"--drop={argv[i + 1]}")
            i += 2
        else:
            out.append(a)
            i += 1
    return out


def _split_pair(spec: str) -> tuple[str, str]:
    flag, sep, val = spec.partition("=")
    if not sep:
        raise SystemExit(f"--set wants FLAG=VALUE, got {spec!r}")
    return flag, val


def health(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=30) as r:
        return json.loads(r.read().decode())


def chat(port: int, msgs: list[dict], max_tokens: int, api_key: str = "") -> dict:
    body = json.dumps({"messages": msgs, "max_tokens": max_tokens, "stream": False}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        # the server's 400/409/500 body names the reason; without this the traceback hides it
        detail = e.read().decode(errors="replace")[:2000]
        raise SystemExit(f"chat failed: HTTP {e.code} {e.reason}: {detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--name", default="arm")
    ap.add_argument("--set", action="append", metavar="FLAG=VALUE", default=[],
                    help="set an engine arg (also accepts the space-separated `--set FLAG VALUE` form)")
    ap.add_argument("--drop", action="append", default=[], help="remove an engine flag")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--boot-timeout", type=int, default=1800, help="seconds to wait for the engine to answer")
    ap.add_argument("--out", default="")
    a = ap.parse_args(_preparse(sys.argv[1:]))
    port = a.port
    sets = [_split_pair(s) for s in a.set]

    cfg = json.loads((ROOT / a.config).read_text())
    args = list(cfg["args"])
    for flag in a.drop:
        while flag in args:
            i = args.index(flag)
            args.pop(i)
            if i < len(args) and not args[i].startswith("--"):
                args.pop(i)
    for flag, val in sets:
        while flag in args:
            i = args.index(flag)
            args.pop(i)
            if i < len(args) and not args[i].startswith("--"):
                args.pop(i)
        args += [flag, val]
    cfg["args"] = args

    log = ROOT / f"sweep-{a.name}.log"
    cfg["log"] = str(log)
    tmp = ROOT / f"sweep-{a.name}.json"
    tmp.write_text(json.dumps(cfg, indent=1))

    if log.exists():
        log.unlink()

    api_key = cfg.get("api_key", "")
    print(f"[{a.name}] args: {' '.join(args)}", flush=True)
    proc = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / "serve/server.py"),
                             "--engine", "strata", "--config", str(tmp), "--port", str(port)],
                            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        # wait for the engine to answer: /health needs no key and generates nothing, so it says
        # "the server is up" without paying for a cold prompt
        t0 = time.time()
        while True:
            try:
                health(port)
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                if proc.poll() is not None:
                    print(f"[{a.name}] server exited (code {proc.returncode}) after {time.time() - t0:.0f} s",
                          flush=True)
                    return 1
                if time.time() - t0 > a.boot_timeout:
                    print(f"[{a.name}] TIMEOUT waiting for engine after {a.boot_timeout} s", flush=True)
                    return 1
                time.sleep(5)
        print(f"[{a.name}] up in {time.time() - t0:.0f} s", flush=True)

        rounds = []
        msgs = [{"role": "user", "content": LONG}]
        for r in range(a.rounds):
            t = time.time()
            chat(port, msgs, a.max_tokens, api_key)
            msgs.append({"role": "user", "content": SHORT})
            chat(port, msgs, 48, api_key)
            rounds.append(round(time.time() - t, 1))
            print(f"[{a.name}] round {r + 1} wall {rounds[-1]} s", flush=True)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()

    text = log.read_text(errors="replace") if log.exists() else ""
    rows = []
    last_hit = None
    last_mem = None
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
    print(f"[{a.name}] expert cache slots: {slots} | vram free: {free}", flush=True)
    print(f"[{a.name}] memory: {last_mem}", flush=True)
    for row in rows:
        print(f"[{a.name}]   gen {row['gen_tok']:4d} tok {row['gen_tps']:6.1f} tok/s "
              f"(prompt {row['prompt_tps']:7.1f}, hit {row['hit']}, acc {row['acc']})", flush=True)
    dec = [r["gen_tps"] for r in rows if r["gen_tok"] >= 64]
    if dec:
        print(f"[{a.name}] DECODE mean {sum(dec) / len(dec):.1f} tok/s  max {max(dec):.1f}", flush=True)

    if a.out:
        Path(a.out).write_text(json.dumps({"name": a.name, "args": args, "rows": rows,
                                           "slots": slots, "vram_free": free, "mem": last_mem}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
