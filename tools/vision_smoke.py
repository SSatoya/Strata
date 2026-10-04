#!/usr/bin/env python3
"""Image round-trip against a running Strata server: a small generated PNG goes through the
vision encoder (strata-vision + mmproj) and the engine's GENI path; the answer must describe
the picture.  Usage: vision_smoke.py [--port 8080] [--api-key KEY] [--image FILE]"""
import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path


def png_bytes() -> bytes:
    """A 64x64 PNG: a red square on a white background, drawn with zlib (no Pillow needed)."""
    import struct
    import zlib

    w = h = 64
    rows = []
    for y in range(h):
        row = b"\x00"
        for x in range(w):
            row += b"\xff\x00\x00" if 16 <= x < 48 and 16 <= y < 48 else b"\xff\xff\xff"
        rows.append(row)
    raw = b"".join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--image", default="", help="a file to send instead of the generated PNG")
    a = ap.parse_args()

    key = a.api_key
    if not key:
        try:
            import os
            key = os.environ.get("STRATA_API_KEY", "")
        except Exception:
            pass

    data = Path(a.image).read_bytes() if a.image else png_bytes()
    b64 = base64.b64encode(data).decode()

    health = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=30).read())
    print(f"health: images={health.get('images')} vision={health.get('vision', {}).get('enabled')}")
    if not health.get("images"):
        print("FAIL: this server was started without the vision encoder")
        return 1

    body = json.dumps({
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What color is the square in this image? Answer in one word."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        # a thinking model spends the budget on reasoning_content first, so 32 tokens leaves
        # content null; 128 leaves room for both
        "max_tokens": 128, "stream": False,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read().decode())

    msg = out["choices"][0]["message"]
    text = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
    usage = out.get("usage", {})
    print(f"answer: {msg.get('content')!r}")
    print(f"usage: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")
    ok = "red" in text.lower()
    print("PASS" if ok else "FAIL (the answer does not mention red)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
