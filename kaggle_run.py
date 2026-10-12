"""Run modules of this repo on a Kaggle T4, one after another in one session.
Run with ``uv run python kaggle_run.py post.check_kv_cache post.gsm8k``.

A Kaggle kernel is a single script, so the local ``post`` and ``core`` sources, uncommitted changes
included, are written into it; on Kaggle's machine it writes them back out as files and runs each
module with ``python -m``. The kernel is private, on a machine with two T4s (the modules use the
first), with internet on to download the model and datasets. It runs on Kaggle's own Python and
PyTorch, not the versions in uv.lock: theirs are built for Kaggle's GPUs, and installing ours would
download a few GB every session. The log streams here while it runs. Afterwards each module's log,
and any files a module wrote to ``out/``, come back to ``out/kaggle/output/``.
The session runs on without this script: ``uv run python kaggle_run.py --follow`` picks up a running
session's log again, then its output.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest
from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus

OWNER = "marcoshernanz"
SLUG = "deepseek-v41-flash-run"
PACKAGES = ["post", "core"]
KERNEL_DIR = Path("out/kaggle/kernel")
OUTPUT_DIR = Path("out/kaggle/output")
TIMEOUT_SECONDS = 3 * 60 * 60  # stops a stuck run long before Kaggle's 12-hour limit
POLL_SECONDS = 15

# What runs on Kaggle, after the lines that set SOURCES and MODULES
KERNEL = r"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SRC = Path("/tmp/src")
OUT = Path("/kaggle/working")  # what Kaggle keeps after the session

for name, text in SOURCES.items():
    path = SRC / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
print("Python", sys.version.split()[0])
subprocess.run([sys.executable, "-c", "import torch; print('PyTorch', torch.__version__)"])

env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_DISABLE_PROGRESS_BARS="1")
failed = []
for module in MODULES:
    print(f"\n$ python -m {module}", flush=True)
    start = time.time()
    with open(OUT / f"{module}.log", "w") as log:
        run = subprocess.Popen(
            [sys.executable, "-u", "-m", module],
            cwd=SRC,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in run.stdout:
            print(line, end="", flush=True)
            log.write(line)
    code = run.wait()
    print(f"({module}: exit code {code}, {(time.time() - start) / 60:.1f} minutes)", flush=True)
    if code != 0:
        failed.append(module)
if (SRC / "out").exists():
    shutil.copytree(SRC / "out", OUT / "out")
print(f"\nfailed: {', '.join(failed)}" if failed else "\nevery module ran to the end")
"""


def build(modules: list[str]) -> None:
    """Write the kernel's script, every source file embedded, and its metadata."""
    paths = [path for package in PACKAGES for path in sorted(Path(package).rglob("*.py"))]
    sources = {str(path): path.read_text() for path in paths}
    script = f"SOURCES = {sources!r}\nMODULES = {modules!r}\n" + KERNEL
    metadata = {
        "id": f"{OWNER}/{SLUG}",
        "title": SLUG,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",  # two T4s
    }
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    (KERNEL_DIR / "run.py").write_text(script)
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"{len(paths)} source files, {len(modules)} modules: {', '.join(modules)}")


def wait_while(api: KaggleApi, statuses: list[KernelWorkerStatus]) -> KernelWorkerStatus:
    """Poll the session until its status is none of statuses; return the new one."""
    while True:
        status = api.kernels_status(f"{OWNER}/{SLUG}").status
        if status not in statuses:
            return status
        time.sleep(POLL_SECONDS)


def stream_log(api: KaggleApi, skip: int) -> int:
    """Print the running session's log, without its first skip entries, until it ends or the
    connection drops. Return how many entries have been printed in all."""
    request = ApiGetKernelSessionLogsStreamRequest()
    request.user_name = OWNER
    request.kernel_slug = SLUG
    request.wait_for_logs_url_seconds = 30
    entries = 0
    with api.build_kaggle_client() as kaggle:
        response = kaggle.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
        if "json" in response.headers.get("Content-Type", ""):  # ended: the whole log at once
            lines = [f"data: {json.dumps(entry)}" for entry in json.loads(response.text)]
        else:
            response.encoding = "utf-8"  # it names no charset: requests would guess Latin-1
            lines = response.iter_lines(decode_unicode=True)
        try:
            for line in lines:
                if line.startswith("data: {"):
                    entries += 1
                    if entries > skip:
                        print(json.loads(line.removeprefix("data: "))["data"], end="", flush=True)
                elif "END_OF_LOG" in line:
                    break
        except requests.RequestException:  # the connection dropped; the caller connects again
            pass
    return max(entries, skip)


def follow(api: KaggleApi) -> None:
    """Print the session's log until it ends, then download its output."""
    status = wait_while(api, [KernelWorkerStatus.QUEUED])  # the log exists once it runs
    printed = 0
    while status == KernelWorkerStatus.RUNNING:
        printed = stream_log(api, printed)  # a stream that starts again replays the whole log
        time.sleep(POLL_SECONDS)
        status = api.kernels_status(f"{OWNER}/{SLUG}").status
    status = wait_while(api, [KernelWorkerStatus.CANCEL_REQUESTED])
    failure = api.kernels_status(f"{OWNER}/{SLUG}").failure_message
    print(f"\nsession ended: {status.name}" + (f", {failure}" if failure else ""))

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    files, _ = api.kernels_output(f"{OWNER}/{SLUG}", str(OUTPUT_DIR), force=True)
    print(f"saved {len(files)} files to {OUTPUT_DIR}/")


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    if sys.argv[1:] == ["--follow"]:
        follow(api)
        return
    modules = sys.argv[1:]
    assert modules, "name at least one module, e.g. post.gsm8k"
    build(modules)
    pushed = api.kernels_push(str(KERNEL_DIR), timeout=str(TIMEOUT_SECONDS))
    assert not pushed.error, pushed.error
    print(f"pushed version {pushed.version_number}: {pushed.url}")
    follow(api)


if __name__ == "__main__":
    main()
