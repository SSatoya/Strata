"""tools/mtp_fetch.py - plan v0.3, P0.3/P6: the MTP block from the BF16 checkpoint, without the checkpoint.

The GSQ-RCO GGUF ships no MTP head. The BF16 checkpoint (Qwen/Qwen3.8-Flash-Next, 360 GB in 131 shards) does:
31 `mtp.*` tensors scattered over 28 shards. Safetensors puts a JSON header (name -> dtype, shape, byte range)
at the start of each shard, so HTTP range requests can read the headers and then only the MTP tensors.

    python tools/mtp_fetch.py inventory --out DIR          # headers only (a few KB per shard)
    python tools/mtp_fetch.py fetch --out DIR [--only SUBSTR]  # the MTP tensors themselves, resumable
    python tools/mtp_fetch.py verify --out DIR           # re-hash what is on disk: no download

`fetch` writes one raw file per tensor plus `mtp-manifest.json` (dtype, shape, source shard, byte range,
sha256). It never downloads anything but the ranges named in the headers. Nothing here runs a model.

A range request is only believed when the server answers 206 with that exact range: a mirror or proxy that
drops the Range header answers 200 with the whole file, and the first bytes of a shard are its own JSON
header - saving those as a tensor loads garbage, decodes fine, and leaves the drafter at 0 acceptance with
no error anywhere (issue #327). `fetch` therefore refuses such a server, and re-reads a tensor already on
disk whose bytes are a shard header or hash differently from what an earlier fetch of the same source
recorded, so a corrupt install repairs itself instead of being kept.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request

# #214: a fixed commit of the checkpoint (its `sha` from https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next
# on 2026-09-30), so every install reads the same tensors; STRATA_MTP_REVISION overrides it (e.g. main).  When the
# repository no longer has it, the current files are read instead, with a message (resolve_repo).  STRATA_MTP_REPO
# names another host serving the same files, for when this one is reached through a proxy that drops Range (#327).
REVISION = os.environ.get("STRATA_MTP_REVISION") or "de4b8e4d43b917e7706784d8bb445c9af86a3540"
REPO = os.environ.get("STRATA_MTP_REPO") or "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/%s/" % REVISION
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I64": 8, "I32": 4}


class RangeRefused(IOError):
    """A range request answered with something other than that range: the whole file (200), or another slice.
    Keeping those bytes is what made MTP acceptance 0 with no visible error (#327)."""


def content_range_slice(value):
    """(start, end) of a `Content-Range: bytes 8-100/4096` header, or None when it is missing or malformed."""
    m = re.match(r"\s*bytes\s+(\d+)-(\d+)/(\d+|\*)\s*$", value or "")
    return None if m is None else (int(m.group(1)), int(m.group(2)))


def get(url, start=None, end=None, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "strata-mtp-fetch"})
            if start is not None:
                req.add_header("Range", "bytes=%d-%d" % (start, end))
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
                status, headers = getattr(r, "status", None) or r.getcode(), getattr(r, "headers", None)
            if start is None:
                return data
            # The length check alone cannot see this: a proxy truncates its 200 to the length asked for, and the
            # manifest hashes the bytes that were downloaded, so both agree while the tensor is the shard header.
            # Only a 206 carrying exactly the requested slice is a range (#327).
            if status != 206:
                raise RangeRefused("%s: HTTP %s instead of 206 for bytes %d-%d - the server ignored the Range "
                                   "header (a mirror or proxy?); set STRATA_MTP_REPO to one that answers ranges"
                                   % (url, status, start, end))
            got = content_range_slice(headers.get("Content-Range") if headers is not None else None)
            if got != (start, end):
                raise RangeRefused("%s: Content-Range %r is not the requested bytes %d-%d"
                                   % (url, headers.get("Content-Range") if headers is not None else None,
                                      start, end))
            if len(data) != end - start + 1:
                raise IOError("short range read: %d of %d" % (len(data), end - start + 1))
            return data
        except RangeRefused:
            raise                        # a server that drops Range drops it every time: say so, don't retry it
        except Exception as e:  # network errors are retried, then surfaced
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            print("retry %s: %s" % (url, e), file=sys.stderr)


def resolve_repo():
    """REPO, or the repository's current files when the pinned revision is gone from it (a 404 on its index)."""
    global REPO
    try:
        req = urllib.request.Request(REPO + "model.safetensors.index.json", method="HEAD",
                                     headers={"User-Agent": "strata-mtp-fetch"})
        urllib.request.urlopen(req, timeout=120).close()
    except urllib.error.HTTPError as e:
        if e.code == 404 and "/resolve/main/" not in REPO:
            print("the checkpoint's pinned revision %s is gone: reading its current files (main)" % REVISION,
                  file=sys.stderr)
            REPO = re.sub(r"/resolve/[^/]+/", "/resolve/main/", REPO)   # works for a STRATA_MTP_REPO host too
    except OSError:
        pass                                        # no answer: get() retries and reports it
    return REPO


def shard_header(shard):
    """(byte where the tensor data starts, the header, the header's own length) of a shard.  The length is
    kept in the inventory so `fetch` can recognise a saved tensor that is really this header (#327)."""
    url = REPO + shard
    n = struct.unpack("<Q", get(url, 0, 7))[0]
    header = json.loads(get(url, 8, 8 + n - 1))
    return 8 + n, header, n


def inventory(out):
    index = json.loads(get(REPO + "model.safetensors.index.json"))["weight_map"]
    mtp = {k: v for k, v in index.items() if k.startswith("mtp.")}
    rows, total = [], 0
    for shard in sorted(set(mtp.values())):
        base, header, header_len = shard_header(shard)
        for name, meta in header.items():
            if name in mtp and mtp[name] == shard:
                a, b = meta["data_offsets"]
                rows.append(dict(name=name, shard=shard, dtype=meta["dtype"], shape=meta["shape"],
                                 start=base + a, end=base + b - 1, bytes=b - a, header_len=header_len))
                total += b - a
    missing = sorted(set(mtp) - set(r["name"] for r in rows))
    if missing:
        sys.exit("tensors named in the index but absent from their shard headers: %s" % missing)
    rows.sort(key=lambda r: r["name"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "mtp-inventory.json"), "w", encoding="utf-8") as f:
        json.dump(dict(repo=REPO, total_bytes=total, tensors=rows), f, indent=1)
    lines = ["# MTP block in the BF16 checkpoint", "", "%d tensors, %.3f GB, in %d shards." % (len(rows), total / 1e9, len(set(r["shard"] for r in rows))), "",
             "| tensor | dtype | shape | MB |", "|---|---|---|---:|"]
    for r in rows:
        lines.append("| `%s` | %s | %s | %.1f |" % (r["name"], r["dtype"], "x".join(map(str, r["shape"])), r["bytes"] / 1e6))
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "mtp-inventory.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return rows


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def is_shard_header(path, header_len):
    """True when the file starts with the shard's own safetensors header: that header's length as a u64, then
    a '{'.  A server that answers a range request with the whole file leaves exactly these bytes where a
    tensor should be, which is how 20 of 31 MTP tensors were lost in silence (#327)."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return False
    return len(head) >= 9 and head[8:9] == b"{" and struct.unpack("<Q", head[:8])[0] == header_len


def load_inventory(out):
    """The saved inventory, read again when it came from another repository or revision, or was written before
    the header lengths were recorded: a byte range in one source is not the same bytes in another."""
    path = os.path.join(out, "mtp-inventory.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            inv = json.load(f)
        rows = inv.get("tensors") or []
        if inv.get("repo") != REPO or any("header_len" not in r for r in rows):
            print("the saved inventory was read from %s: reading it again from %s"
                  % (inv.get("repo"), REPO), file=sys.stderr)
        else:
            return rows
    return inventory(out)


def saved_is_usable(path, row, before):
    """Whether the bytes already on disk for `row` may be kept (resumed, or left alone).  Prints why not; the
    caller then starts that tensor over.  `before` is the same tensor in the previous manifest, or None."""
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if not have:
        return True
    if is_shard_header(path, row.get("header_len")):
        print("%s: the %d bytes on disk are %s's own header, not the tensor: fetching it again"
              % (path, have, row["shard"]), file=sys.stderr)
        return False
    # the manifest hashes what was downloaded, so it cannot prove the bytes are the tensor - but a complete
    # file that no longer hashes to it was truncated, edited, or fetched from somewhere else (#327).
    if have == row["bytes"] and before and before.get("sha256") and sha256_file(path) != before["sha256"]:
        print("%s: sha256 is not the one the earlier fetch recorded: fetching it again" % path, file=sys.stderr)
        return False
    return True


def fetch(out, only=None):
    rows = load_inventory(out)
    tdir = os.path.join(out, "tensors")
    os.makedirs(tdir, exist_ok=True)
    man_path = os.path.join(out, "mtp-manifest.json")
    before = {}
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            before = {r["name"]: r for r in json.load(f)}
    manifest = []
    chunk = 64 << 20
    for r in rows:
        if only and only not in r["name"]:
            # an --only run leaves the tensors it skipped as the earlier fetch of this same source recorded them,
            # so the manifest still names everything on disk and `verify` checks all of it.
            if r["name"] in before and before[r["name"]].get("repo") == REPO:
                manifest.append(before[r["name"]])
            continue
        path = os.path.join(tdir, r["name"] + ".bin")
        if not saved_is_usable(path, r, before.get(r["name"])):
            open(path, "wb").close()
        have = os.path.getsize(path) if os.path.exists(path) else 0
        with open(path, "ab") as f:
            pos = r["start"] + have
            while pos <= r["end"]:
                end = min(pos + chunk - 1, r["end"])
                f.write(get(REPO + r["shard"], pos, end))
                pos = end + 1
                print("%s %.0f%%" % (r["name"], 100 * (pos - r["start"]) / r["bytes"]), file=sys.stderr)
        if os.path.getsize(path) != r["bytes"]:
            sys.exit("%s: size %d != %d" % (path, os.path.getsize(path), r["bytes"]))
        if is_shard_header(path, r.get("header_len")):
            sys.exit("%s: bytes %d-%d of %s came back as the shard's header: the server does not answer range "
                     "requests; set STRATA_MTP_REPO to one that does" % (path, r["start"], r["end"], r["shard"]))
        manifest.append(dict(r, repo=REPO, file=os.path.relpath(path, out), sha256=sha256_file(path)))
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)


def why_unverified(path, r):
    """Why the file on disk is not the tensor the manifest recorded, or None when it is: missing, short, a shard's
    header where the tensor should be (#327), or bytes that no longer hash to what the fetch recorded."""
    if not os.path.exists(path):
        return "missing"
    if os.path.getsize(path) != r["bytes"]:
        return "size %d != %d" % (os.path.getsize(path), r["bytes"])
    if is_shard_header(path, r.get("header_len")):
        return "holds %s's header, not the tensor" % r["shard"]
    digest = sha256_file(path)
    if digest != r["sha256"]:
        return "sha256 %s != %s recorded" % (digest, r["sha256"])
    return None


def verify(out):
    """Re-hash every tensor the manifest names against the manifest itself - no download.  A corrupt install
    (short, edited, or a shard header where a tensor should be) exits non-zero, so setup fetches it again."""
    man_path = os.path.join(out, "mtp-manifest.json")
    if not os.path.exists(man_path):
        sys.exit("no mtp-manifest.json in %s: run fetch first" % out)
    with open(man_path, encoding="utf-8") as f:
        rows = json.load(f)
    bad = 0
    for r in rows:
        why = why_unverified(os.path.join(out, r["file"]), r)
        if why:
            print("%s: %s" % (os.path.join(out, r["file"]), why), file=sys.stderr)
            bad += 1
    print("%d of %d MTP tensors verified" % (len(rows) - bad, len(rows)))
    if bad:
        print("run fetch again: it re-reads those tensors", file=sys.stderr)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inventory", "fetch", "verify"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only")
    a = ap.parse_args()
    if a.cmd == "verify":
        return verify(a.out)                      # reads only what is on disk: no network, no resolve_repo
    resolve_repo()
    inventory(a.out) if a.cmd == "inventory" else fetch(a.out, a.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
