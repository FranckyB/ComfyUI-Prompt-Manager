#!/usr/bin/env python3
"""Regression tests for llama.cpp GPU support detection in Prompt Generator.

The GPU-offload decision used to be wrong on Windows: `ldd` does not exist
there, `--version` no longer names the backend, and `-h` lists `--n-gpu-layers`
on every build (CPU-only included). The result was a guaranteed "CPU-only build"
report plus a forced `--n-gpu-layers 0`, i.e. CPU inference on machines that had
a perfectly good CUDA build.

These tests exercise the real functions from nodes/prompt_generator.py with a
simulated Windows environment (no `ldd`, DLLs next to the exe). Run with:

    python3 tools/test_llama_gpu_detect.py
"""

import os
import re
import shutil
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(os.path.dirname(HERE), "nodes", "prompt_generator.py")

BLOCK_START = '_BACKEND_LIBRARY_SUFFIXES = ('
BLOCK_END = 'class PromptGenerator:'


def load_detection_code(os_name, fake_subprocess, fake_shutil=None, listdir=None):
    """Exec the detection helpers from prompt_generator.py in a stubbed namespace."""
    with open(SOURCE, "r", encoding="utf-8") as handle:
        source = handle.read()

    start = source.index(BLOCK_START)
    end = source.index(BLOCK_END, start)
    block = source[start:end]

    os_path = types.SimpleNamespace(
        isfile=lambda path: path in fake_subprocess.existing_files,
        dirname=os.path.dirname,
        abspath=os.path.abspath,
        sep=os.sep,
    )
    stub_os = types.SimpleNamespace(
        name=os_name,
        path=os_path,
        listdir=listdir or (lambda directory: []),
    )
    namespace = {
        "os": stub_os,
        "shutil": fake_shutil or shutil,
        "subprocess": fake_subprocess,
        "re": re,
    }
    exec(compile(block, SOURCE, "exec"), namespace)
    return namespace


class CompletedRun:
    def __init__(self, stdout="", stderr=""):
        self.stdout = stdout
        self.stderr = stderr


class FakeSubprocess:
    PIPE = -1
    CREATE_NO_WINDOW = 0x08000000
    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, outputs=None, existing_files=None, raise_on=None):
        self.outputs = outputs or {}
        self.existing_files = existing_files or set()
        self.raise_on = raise_on or set()
        self.calls = []

    def run(self, args, **kwargs):
        self.calls.append(list(args))
        target = args[0]
        if target in self.raise_on:
            raise FileNotFoundError(2, "The system cannot find the file specified")
        if target in self.outputs:
            return self.outputs[target]
        return CompletedRun("", "")


CUDA_LIST_DEVICES = CompletedRun(
    stdout="Available devices:\n  CUDA0: NVIDIA GeForce RTX 4090 (24564 MiB, 22100 MiB free)\n"
)
CPU_ONLY_LIST_DEVICES = CompletedRun(stdout="Available devices:\n  (none)\n")
UNSUPPORTED_FLAG = CompletedRun(stderr="error: unrecognized argument: --list-devices\n")


def test_windows_cuda_build_is_detected():
    """Windows CUDA build: --list-devices names the device, ldd must never run."""
    fake = FakeSubprocess(outputs={"llama-server.exe": CUDA_LIST_DEVICES})
    ns = load_detection_code("nt", fake)
    status, backends, note = ns["detect_llama_gpu_support"]("llama-server.exe")
    assert status is True, (status, backends, note)
    assert backends == ["CUDA"], backends
    assert note is None, note
    assert all("ldd" not in call for call in fake.calls), fake.calls


def test_windows_cuda_build_without_list_devices_falls_back_to_dlls():
    """Older Windows build without --list-devices: ggml-cuda.dll next to the exe."""
    exe = os.path.abspath(os.path.join("llama.cpp", "llama-server.exe"))
    fake = FakeSubprocess(
        outputs={exe: UNSUPPORTED_FLAG},
        existing_files={exe},
    )
    fake_shutil = types.SimpleNamespace(which=lambda name: exe)
    listdir = lambda directory: ["llama-server.exe", "ggml-cpu.dll", "ggml-cuda.dll", "cublas64_12.dll"]
    ns = load_detection_code("nt", fake, fake_shutil=fake_shutil, listdir=listdir)
    status, backends, note = ns["detect_llama_gpu_support"](exe)
    assert status is True, (status, backends, note)
    assert "CUDA" in backends, backends
    assert all("ldd" not in call for call in fake.calls), fake.calls


def test_confirmed_cpu_only_build():
    """A CPU-only build must be reported as such, with llama.cpp as the witness."""
    fake = FakeSubprocess(outputs={"llama-server.exe": CPU_ONLY_LIST_DEVICES})
    ns = load_detection_code("nt", fake)
    status, backends, note = ns["detect_llama_gpu_support"]("llama-server.exe")
    assert status is False, (status, backends, note)
    assert backends == [], backends
    assert note and "no GPU device" in note, note


def test_unknown_never_claims_cpu_only():
    """Nothing detectable: must return unknown (None), never False."""
    fake = FakeSubprocess(
        outputs={"llama-server.exe": UNSUPPORTED_FLAG},
        raise_on={"ldd"},
    )
    ns = load_detection_code("nt", fake)
    status, backends, note = ns["detect_llama_gpu_support"]("llama-server.exe")
    assert status is None, (status, backends, note)
    assert backends == [], backends


def test_linux_cuda_build():
    """Linux: builds without --list-devices still resolve through ldd."""
    fake = FakeSubprocess(
        outputs={
            "llama-server": UNSUPPORTED_FLAG,
            "ldd": CompletedRun(
                stdout="\tlibggml-cuda.so => /usr/local/lib/libggml-cuda.so\n"
                "\tlibggml-cpu.so => /usr/local/lib/libggml-cpu.so\n"
            ),
        },
    )
    ns = load_detection_code("posix", fake)
    status, backends, note = ns["detect_llama_gpu_support"]("llama-server")
    assert status is True, (status, backends, note)
    assert backends == ["CUDA"], backends


def test_hip_not_matched_on_substrings():
    ns = load_detection_code("posix", FakeSubprocess())
    assert ns["_backends_in_text"]("relationship shipping chip") == set()
    assert ns["_backends_in_text"]("HIP0: AMD RX 7900 XTX") == {"HIP"}
    assert ns["_backends_in_text"]("libggml-hip.so") == {"HIP"}
    assert ns["_backends_in_text"]("Vulkan0: AMD RADV") == {"Vulkan"}


def test_windows_cpu_archive_stays_unknown():
    """CPU-only archive without --list-devices: unknown, so offload is not forced off."""
    exe = os.path.abspath(os.path.join("llama.cpp", "llama-server.exe"))
    fake = FakeSubprocess(outputs={exe: UNSUPPORTED_FLAG}, existing_files={exe})
    fake_shutil = types.SimpleNamespace(which=lambda name: exe)
    listdir = lambda directory: [
        "llama-server.exe", "ggml.dll", "ggml-base.dll",
        "ggml-cpu-alderlake.dll", "ggml-cpu-x64.dll", "llama.dll",
    ]
    ns = load_detection_code("nt", fake, fake_shutil=fake_shutil, listdir=listdir)
    status, backends, note = ns["detect_llama_gpu_support"](exe)
    assert status is None, (status, backends, note)
    assert backends == [], backends


def test_missing_binary_reports_note():
    fake = FakeSubprocess(raise_on={"llama-server.exe"})
    ns = load_detection_code("nt", fake)
    status, backends, note = ns["detect_llama_gpu_support"]("llama-server.exe")
    assert status is None, (status, backends, note)
    assert note and "not found" in note, note


def main():
    tests = [(name, value) for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS  {name}")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
