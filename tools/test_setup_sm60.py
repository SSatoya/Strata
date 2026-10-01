"""Tests for setup.py's opt-in for cards below compute capability 7.5 - the community Pascal / Volta build (#236):
which cards the gate admits and what it says about them, that the CUDA toolkit it compiles with must be 12.x
(CUDA 13 removed compute_60/61/70), that CMake gets -DSTRATA_EXPERIMENTAL_SM60=ON only for such a card, that the
ready-made engine is not offered for one, and that engine/BUILD.json remembers the build so a later start of that
install keeps it.  No GPU, no compiler, no downloads.

    python -m unittest tools.test_setup_sm60
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def gpu(index=0, arch="61", vram=11.0, name="NVIDIA GeForce GTX 1080 Ti", **kw) -> dict:
    return {"index": index, "name": name, "vram_gb": vram, "arch": arch, "driver": "550.107.02", **kw}


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


class TheGate(unittest.TestCase):
    """The card floor: compute capability 7.5, or 6.0 with the opt-in."""

    def test_a_gtx_10_is_refused_without_the_opt_in(self):
        with mock.patch.object(setup, "SM60", False):
            p = setup.gpu_problem(gpu())
            self.assertIn("older than the RTX 20 series", p)
            self.assertIn("7.5", p)
            self.assertIn("--experimental-sm60", p)          # the message names the way out, as CMake's does
        self.assertIsNone(setup.gpu_problem(gpu(arch="75")))
        self.assertIsNone(setup.gpu_problem(gpu(arch="120")))

    def test_the_opt_in_admits_pascal_and_volta_but_nothing_older(self):
        with mock.patch.object(setup, "SM60", True):
            self.assertIsNone(setup.gpu_problem(gpu()))                # sm_61: a 1080 Ti
            self.assertIsNone(setup.gpu_problem(gpu(arch="60")))       # sm_60
            self.assertIsNone(setup.gpu_problem(gpu(arch="70")))       # Volta
            self.assertIn("even the experimental Pascal / Volta build", setup.gpu_problem(gpu(arch="52")))

    def test_the_table_says_which_build_a_card_needs(self):
        with mock.patch.object(setup, "SM60", True):
            _, out = quiet(setup.gpu_table, [gpu(), gpu(index=1, arch="86", name="NVIDIA GeForce RTX 3090")])
        self.assertIn("GPU 0: NVIDIA GeForce GTX 1080 Ti, 11 GB VRAM - can be used (the experimental Pascal / Volta "
                      "build: unsupported upstream)", out)
        self.assertIn("GPU 1: NVIDIA GeForce RTX 3090, 11 GB VRAM - can be used\n", out)


class TheOptIn(unittest.TestCase):
    def test_asked_on_the_command_line_in_the_env_or_already_installed(self):
        self.assertTrue(setup.sm60_requested(True, "", {}))
        self.assertTrue(setup.sm60_requested(False, "1", {}))
        self.assertTrue(setup.sm60_requested(False, " ON ", {}))
        self.assertFalse(setup.sm60_requested(False, "0", {}))
        self.assertFalse(setup.sm60_requested(False, "", {}))
        self.assertTrue(setup.sm60_requested(False, "", {"sm60": True}))   # a start of an install made with it

    def test_engine_meta(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "engine").mkdir()
            with mock.patch.object(setup, "ROOT", root):
                self.assertEqual(setup.engine_meta(), {})             # nothing installed
                (root / "engine" / "BUILD.json").write_text(
                    json.dumps({"source": "local", "archs": [61], "sm60": True}))
                self.assertTrue(setup.engine_meta().get("sm60"))
                (root / "engine" / "BUILD.json").write_text("{not json")
                self.assertEqual(setup.engine_meta(), {})

    def test_only_a_pre_sm_75_card_takes_the_build(self):
        with mock.patch.object(setup, "SM60", True):
            self.assertTrue(setup.sm60_build([61]))
            self.assertTrue(setup.sm60_build([70, 86]))               # a split across both generations
            self.assertFalse(setup.sm60_build([75, 86]))
        with mock.patch.object(setup, "SM60", False):
            self.assertFalse(setup.sm60_build([61]))


class TheToolkit(unittest.TestCase):
    """nvcc has to be one that can still compile the card."""

    def test_pick(self):
        found = [("/cuda-11.5", (11, 5)), ("/cuda-12.6", (12, 6)), ("/cuda-13.2", (13, 2))]
        self.assertEqual(setup.pick_nvcc(found), ("/cuda-13.2", (13, 2)))
        self.assertEqual(setup.pick_nvcc(found, below=(13, 0)), ("/cuda-12.6", (12, 6)))
        self.assertEqual(setup.pick_nvcc([("/cuda-13.2", (13, 2))], below=(13, 0)), (None, None))
        self.assertEqual(setup.pick_nvcc([]), (None, None))

    def build_tools(self, found, archs=("61",), out=None, answer="n"):
        """found: the toolkits this PC has, as find_nvcc would see them.  `out`: say into this buffer instead of
        returning what was printed (a fail() raises, so the caller needs the buffer it wrote to); `answer`: the
        answer to "Install them now?" - "n" keeps the test from touching the PC."""
        with mock.patch.object(setup, "SM60", True), \
                mock.patch.object(setup, "find_nvcc", lambda below=None: setup.pick_nvcc(found, below)), \
                mock.patch.object(setup.shutil, "which", lambda c: "/usr/bin/g++"), \
                mock.patch("builtins.input", lambda *a: answer):
            if out is None:
                return quiet(setup.install_build_tools, gpu(archs=list(archs)), True)
            with contextlib.redirect_stdout(out):
                return setup.install_build_tools(gpu(archs=list(archs)), False)

    def test_a_pc_with_only_cuda_13_is_asked_for_the_toolkit_that_compiles_it(self):
        out = io.StringIO()
        only13 = [("/usr/local/cuda-13.2/bin/nvcc", (13, 2))]
        with self.assertRaises(SystemExit) as got:
            self.build_tools(only13, out=out)
        self.assertEqual(got.exception.code, 1)
        self.assertIn("which needs: the NVIDIA CUDA Toolkit 12.6", out.getvalue())
        self.assertNotIn("CUDA Toolkit 13", out.getvalue())     # CUDA 13 removed compute_60/61/70 (#236)
        with self.assertRaises(SystemExit):                     # and no toolkit at all
            self.build_tools([], out=io.StringIO())

    def test_cuda_12_is_the_one_it_compiles_with(self):
        found = [("/usr/bin/nvcc", (11, 5)), ("/usr/local/cuda-12.6/bin/nvcc", (12, 6)),
                 ("/usr/local/cuda-13.2/bin/nvcc", (13, 2))]
        (nvcc, vcvars), out = self.build_tools(found)
        self.assertEqual(nvcc, "/usr/local/cuda-12.6/bin/nvcc")     # not the newest: the newest that compiles it
        self.assertIsNone(vcvars)
        self.assertIn("build tools present (CUDA 12.6)", out)

    def test_a_supported_card_still_wants_the_newest_toolkit(self):
        seen = []

        def find(below=None):
            seen.append(below)
            return "/usr/local/cuda-13.2/bin/nvcc", (13, 2)

        with mock.patch.object(setup, "find_nvcc", find), \
                mock.patch.object(setup.shutil, "which", lambda c: "/usr/bin/g++"):
            quiet(setup.install_build_tools, gpu(arch="120", archs=["120"]), True)
        self.assertEqual(seen, [None])


class TheBuild(unittest.TestCase):
    def test_the_cmake_it_compiles_with_is_the_one_setup_installed(self):
        """A Pascal card has no ready-made engine, so the compile is not optional - and Ubuntu 22.04's system cmake
        (3.22) cannot even configure Strata (CMakeLists.txt needs 3.24).  The pinned one in the venv wins."""
        with tempfile.TemporaryDirectory() as d:
            venv = Path(d) / "bin"
            venv.mkdir(parents=True)
            (venv / "cmake").write_text("")
            with mock.patch.object(setup.sys, "executable", str(venv / "python")), \
                    mock.patch.object(setup.shutil, "which", lambda n: "/usr/bin/cmake"):
                self.assertEqual(setup.find_tool("cmake"), str(venv / "cmake"))
            with mock.patch.object(setup.sys, "executable", str(Path(d) / "none" / "python")), \
                    mock.patch.object(setup.shutil, "which", lambda n: "/usr/bin/cmake"):
                self.assertEqual(setup.find_tool("cmake"), "/usr/bin/cmake")     # nothing in the venv: PATH still works

    def test_cmake_gets_the_define_only_for_a_pre_sm_75_card(self):
        with mock.patch.object(setup, "SM60", True):
            self.assertIn("-DSTRATA_EXPERIMENTAL_SM60=ON", setup.cuda_cmake_defs([61], "/nvcc", "/llama"))
            self.assertIn("-DSTRATA_EXPERIMENTAL_SM60=ON", setup.cuda_cmake_defs([61, 86], "/nvcc", "/llama"))
            self.assertNotIn("-DSTRATA_EXPERIMENTAL_SM60=ON", setup.cuda_cmake_defs([75], "/nvcc", "/llama"))
        with mock.patch.object(setup, "SM60", False):
            self.assertNotIn("-DSTRATA_EXPERIMENTAL_SM60=ON", setup.cuda_cmake_defs([61], "/nvcc", "/llama"))
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=61;86", setup.cuda_cmake_defs([61, 86], "/nvcc", "/llama"))

    def test_the_ready_made_engine_is_not_offered_for_such_a_card(self):
        def none(*a, **kw):
            raise AssertionError("went to the network for a pre-sm_75 card")

        with mock.patch.object(setup, "SM60", True), mock.patch.object(setup, "download", none), \
                mock.patch.object(setup.urllib.request, "urlopen", none):
            eng, out = quiet(setup.get_prebuilt, "https://example.com/releases", gpu(), "none")
        self.assertIsNone(eng)
        self.assertIn("compiled here with the experimental Pascal / Volta build", out)

    def test_the_installed_engine_remembers_the_build(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "engine").mkdir()
            (root / "build").mkdir()
            (root / "build" / setup.EXE).write_text("engine")
            with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "SM60", True), \
                    mock.patch.object(setup, "source_hash", lambda parts: "0" * 16), \
                    mock.patch.object(setup, "source_version", lambda: "0.1.31"), \
                    mock.patch.object(setup, "install_build_tools",
                                      lambda g, yes: ("/usr/local/cuda-12.6/bin/nvcc", None)), \
                    mock.patch.object(setup, "cmake_build", lambda *a: None):
                quiet(setup.build_engine, gpu(archs=["61"]), "none", True, "/llama")
            meta = json.loads((root / "engine" / "BUILD.json").read_text())
        self.assertTrue(meta["sm60"])
        self.assertEqual(meta["archs"], [61])
        self.assertEqual(meta["source"], "local")


if __name__ == "__main__":
    unittest.main()

