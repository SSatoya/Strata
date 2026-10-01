"""Tests for tools/mtp_fetch.py (#327): a mirror that ignores the Range header is refused instead of saving the
shard's own header over a tensor, an inventory read from another repository is read again, and a tensor on disk
that is a shard header or no longer hashes to what was recorded is fetched again.  A fake mirror serves a
synthetic safetensors shard - nothing is downloaded, no model runs.

    python -m unittest tools.test_mtp_fetch
"""
from __future__ import annotations

import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import mtp_fetch  # noqa: E402

REPO = "https://mirror.invalid/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/"
SHARD = "model-00001-of-00002.safetensors"


def blob(seed, n):
    """Deterministic tensor bytes that are never a safetensors header (byte 8 is never '{')."""
    return bytes(((seed * 37 + i * 11) % 251) for i in range(n))


TENSORS = {
    "mtp.layers.0.fc_embedding.weight": ("BF16", [8, 8], blob(1, 128)),
    "mtp.layers.0.norm.weight": ("F32", [16], blob(2, 64)),
    "model.layers.0.mlp.weight": ("BF16", [4, 4], blob(3, 32)),      # not an MTP tensor: never fetched
}
MTP_NAMES = sorted(n for n in TENSORS if n.startswith("mtp."))


def shard_bytes():
    """A safetensors shard: u64 header length, the JSON header padded to 8 bytes, then the tensor data."""
    header, off, data = {}, 0, b""
    for name, (dtype, shape, raw) in TENSORS.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        data += raw
        off += len(raw)
    js = json.dumps(header, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 8)
    return struct.pack("<Q", len(js)) + js + data


SHARD_BYTES = shard_bytes()
HEADER_LEN = struct.unpack("<Q", SHARD_BYTES[:8])[0]
FILES = {SHARD: SHARD_BYTES,
         "model.safetensors.index.json": json.dumps(
             {"metadata": {"total_size": 0}, "weight_map": {n: SHARD for n in TENSORS}}).encode()}


class Response(io.BytesIO):
    def __init__(self, body=b"", status=200, content_range=None):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body))}
        if content_range:
            self.headers["Content-Range"] = content_range

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Mirror:
    """The fake endpoint.  `ranges=False` is the mirror from #327: it answers a range request with the whole
    file (200), which the old code saved as the tensor because the length and its own hash both agreed."""

    def __init__(self, files=FILES, ranges=True, content_range=None, truncate=False):
        self.files, self.ranges, self.content_range, self.truncate = files, ranges, content_range, truncate
        self.seen = []                                    # (file, Range header) of every request it answered

    def urlopen(self, req, timeout=None):
        body = self.files[req.full_url.rsplit("/", 1)[-1]]
        rng = req.get_header("Range")
        self.seen.append((req.full_url.rsplit("/", 1)[-1], rng))
        if req.get_method() == "HEAD":
            return Response(b"", status=200)
        if rng is None:
            return Response(body, status=200)
        start, end = (int(x) for x in rng[len("bytes="):].split("-"))
        if not self.ranges:                               # the server dropped the Range header
            return Response(body[: end - start + 1] if self.truncate else body, status=200)
        return Response(body[start:end + 1], status=206,
                        content_range=self.content_range or "bytes %d-%d/%d" % (start, end, len(body)))


def quiet(fn, *args, **kw):
    """Runs fn with stdout swallowed, returning (what it returned, what it printed to stderr)."""
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        return fn(*args, **kw), err.getvalue()


class MTPFetch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def net(self, mirror, fn, *args, repo=REPO, **kw):
        """Runs fn against the fake mirror, with the retry sleeps switched off, and returns its stderr."""
        with mock.patch.object(mtp_fetch, "REPO", repo), \
                mock.patch.object(mtp_fetch.urllib.request, "urlopen", mirror.urlopen), \
                mock.patch.object(mtp_fetch.time, "sleep", lambda s: None):
            return quiet(fn, *args, **kw)[1]

    def fetch(self, mirror):
        return self.net(mirror, mtp_fetch.fetch, str(self.dir))

    def tensor(self, name):
        return (self.dir / "tensors" / (name + ".bin")).read_bytes()

    def manifest(self):
        return json.loads((self.dir / "mtp-manifest.json").read_text())

    def test_a_mirror_that_ignores_range_is_refused(self):
        mirror = Mirror(ranges=False, truncate=True)      # 200 cut to the length asked for: what #327 saw
        with mock.patch.object(mtp_fetch, "REPO", REPO), \
                mock.patch.object(mtp_fetch.urllib.request, "urlopen", mirror.urlopen):
            with self.assertRaises(mtp_fetch.RangeRefused) as cm:
                mtp_fetch.get(REPO + SHARD, 0, 7)
        msg = str(cm.exception)
        self.assertIn("HTTP 200 instead of 206 for bytes 0-7", msg)
        self.assertIn("ignored the Range header", msg)
        self.assertIn("STRATA_MTP_REPO", msg)
        self.assertEqual(len(mirror.seen), 1)             # not retried: that server answers the same way every time

    def test_a_fetch_stops_at_such_a_mirror_and_saves_nothing(self):
        mirror = Mirror(ranges=False)
        with self.assertRaises(mtp_fetch.RangeRefused):
            self.net(mirror, mtp_fetch.inventory, str(self.dir))
        self.assertFalse((self.dir / "tensors").exists())

    def test_a_wrong_content_range_is_refused(self):
        mirror = Mirror(content_range="bytes 0-7/%d" % len(SHARD_BYTES))   # 206, but not the slice asked for
        with mock.patch.object(mtp_fetch, "REPO", REPO), \
                mock.patch.object(mtp_fetch.urllib.request, "urlopen", mirror.urlopen):
            with self.assertRaises(mtp_fetch.RangeRefused) as cm:
                mtp_fetch.get(REPO + SHARD, 8, 20)
        self.assertIn("Content-Range 'bytes 0-7/", str(cm.exception))

    def test_a_short_206_is_still_reported_as_a_short_read(self):
        class Short(Mirror):
            def urlopen(self, req, timeout=None):
                r = super().urlopen(req, timeout)
                if r.status == 206:
                    r.seek(0)
                    r.truncate(3)
                return r

        with mock.patch.object(mtp_fetch, "REPO", REPO), \
                mock.patch.object(mtp_fetch.urllib.request, "urlopen", Short().urlopen), \
                mock.patch.object(mtp_fetch.time, "sleep", lambda s: None):
            with self.assertRaises(IOError) as cm:
                mtp_fetch.get(REPO + SHARD, 0, 7)
        self.assertIn("short range read: 3 of 8", str(cm.exception))

    def test_the_inventory_records_what_a_shard_header_is(self):
        self.net(Mirror(), mtp_fetch.inventory, str(self.dir))
        inv = json.loads((self.dir / "mtp-inventory.json").read_text())
        self.assertEqual(inv["repo"], REPO)
        self.assertEqual([r["name"] for r in inv["tensors"]], MTP_NAMES)
        self.assertTrue(all(r["header_len"] == HEADER_LEN for r in inv["tensors"]))
        self.assertTrue(all(r["start"] >= HEADER_LEN + 8 for r in inv["tensors"]))

    def test_fetch_saves_the_tensor_bytes(self):
        self.fetch(Mirror())
        for name in MTP_NAMES:
            self.assertEqual(self.tensor(name), TENSORS[name][2])
        manifest = self.manifest()
        self.assertEqual([r["name"] for r in manifest], MTP_NAMES)
        for row in manifest:
            self.assertEqual(row["repo"], REPO)
            self.assertEqual(row["bytes"], len(TENSORS[row["name"]][2]))
            self.assertEqual(len(row["sha256"]), 64)
        self.assertFalse((self.dir / "tensors" / "model.layers.0.mlp.weight.bin").exists())

    def test_a_good_install_is_not_downloaded_again(self):
        self.fetch(Mirror())
        second = Mirror()
        self.fetch(second)
        self.assertEqual(second.seen, [])

    def test_a_saved_shard_header_is_fetched_again(self):
        """Exactly what #327 left behind: the tensor file holds the shard's first bytes, and the manifest hash
        matches those bytes - so only recognising the header finds it."""
        self.fetch(Mirror())
        name = "mtp.layers.0.fc_embedding.weight"
        row = next(r for r in self.manifest() if r["name"] == name)
        (self.dir / "tensors" / (name + ".bin")).write_bytes(SHARD_BYTES[: row["bytes"]])
        err = self.fetch(Mirror())
        self.assertIn("are model-00001-of-00002.safetensors's own header", err)
        self.assertEqual(self.tensor(name), TENSORS[name][2])

    def test_a_tensor_that_no_longer_hashes_to_what_was_recorded_is_fetched_again(self):
        self.fetch(Mirror())
        name = "mtp.layers.0.norm.weight"
        (self.dir / "tensors" / (name + ".bin")).write_bytes(blob(99, len(TENSORS[name][2])))
        err = self.fetch(Mirror())
        self.assertIn("sha256 is not the one the earlier fetch recorded", err)
        self.assertEqual(self.tensor(name), TENSORS[name][2])

    def test_a_part_written_tensor_is_resumed_from_where_it_stopped(self):
        self.fetch(Mirror())
        name, raw = "mtp.layers.0.fc_embedding.weight", TENSORS["mtp.layers.0.fc_embedding.weight"][2]
        row = next(r for r in self.manifest() if r["name"] == name)
        (self.dir / "tensors" / (name + ".bin")).write_bytes(raw[:16])
        mirror = Mirror()
        self.fetch(mirror)
        self.assertEqual(self.tensor(name), raw)
        self.assertEqual([rng for _, rng in mirror.seen if rng], ["bytes=%d-%d" % (row["start"] + 16, row["end"])])

    def test_an_inventory_from_another_repository_is_read_again(self):
        self.fetch(Mirror())
        mirror = Mirror()
        err = self.net(mirror, mtp_fetch.fetch, str(self.dir),
                       repo=REPO.replace("/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/", "/resolve/main/"))
        self.assertIn("the saved inventory was read from", err)
        self.assertIn("reading it again", err)
        self.assertIn(("model.safetensors.index.json", None), mirror.seen)

    def test_an_only_run_keeps_the_tensors_it_skipped_in_the_manifest(self):
        self.fetch(Mirror())
        name = "mtp.layers.0.norm.weight"
        row = next(r for r in self.manifest() if r["name"] == name)
        (self.dir / "tensors" / (name + ".bin")).write_bytes(blob(9, row["bytes"]))
        mirror = Mirror()
        self.net(mirror, mtp_fetch.fetch, str(self.dir), only=name)
        self.assertEqual([rng for _, rng in mirror.seen], ["bytes=%d-%d" % (row["start"], row["end"])])
        self.assertEqual([r["name"] for r in self.manifest()], MTP_NAMES)
        self.assertEqual(quiet(mtp_fetch.verify, str(self.dir))[0], 0)

    def test_verify_reads_the_disk_and_never_the_network(self):
        self.fetch(Mirror())
        with mock.patch.object(mtp_fetch.urllib.request, "urlopen",
                               lambda req, timeout=None: (_ for _ in ()).throw(AssertionError("asked the network"))):
            rc, _ = quiet(mtp_fetch.verify, str(self.dir))
        self.assertEqual(rc, 0)

    def test_verify_names_a_corrupt_tensor(self):
        self.fetch(Mirror())
        name = "mtp.layers.0.fc_embedding.weight"
        path = self.dir / "tensors" / (name + ".bin")
        path.write_bytes(SHARD_BYTES[: len(TENSORS[name][2])])
        rc, err = quiet(mtp_fetch.verify, str(self.dir))
        self.assertEqual(rc, 1)
        self.assertIn(name + ".bin: holds model-00001-of-00002.safetensors's header", err)
        path.write_bytes(TENSORS[name][2][:8])
        rc, err = quiet(mtp_fetch.verify, str(self.dir))
        self.assertEqual(rc, 1)
        self.assertIn("size 8 != 128", err)


if __name__ == "__main__":
    unittest.main()


