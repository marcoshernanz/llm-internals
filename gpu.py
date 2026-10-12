"""Run modules of this repo on a cloud GPU, an NVIDIA L4 on Modal, one after another in one
container. Run with ``uv run modal run gpu.py --modules post.check_kv_cache,post.eval_gsm8k``.

The local ``core`` and ``post`` packages are copied in on every run, so uncommitted changes run too,
on the versions in uv.lock. Each module runs in its own process, so one module's compiled code or
memory doesn't carry into the next. Its output streams here and is saved to ``out/<module>.log``;
files a module writes to ``out/`` come back to the local ``out/`` too. One container for all the
modules pays the startup and the model download once. At the end it prints how long it waited
for a GPU, how long it ran, and what that cost: Modal bills the running time only.
"""

import subprocess
import sys
import time
from pathlib import Path

import modal

GPU = "L4"  # the cheapest GPU on Modal that computes in bf16
DOLLARS_PER_HOUR = 0.80  # the L4's rate; CPU and memory add about 1%
TIMEOUT_SECONDS = 60 * 60  # stops a stuck run before it costs more than $0.80
OUT_DIR = Path("out")

image = modal.Image.debian_slim(python_version="3.14")
image = image.uv_sync()  # uv.lock's versions, dev group too: post.check_qwen3 needs transformers
image = image.env({"HF_HUB_DISABLE_PROGRESS_BARS": "1"})
image = image.add_local_python_source("core", "post")
app = modal.App("deepseek-v4-1-flash", image=image)


@app.function(gpu=GPU, timeout=TIMEOUT_SECONDS)
def run(modules: list[str]) -> tuple[dict[str, bytes], float]:
    """Run the modules; return the files in out/ and the seconds this took."""
    started = time.time()
    OUT_DIR.mkdir(exist_ok=True)
    for module in modules:
        print(f"\n$ python -m {module}", flush=True)
        start = time.time()
        with open(OUT_DIR / f"{module}.log", "w") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", module],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
        code = process.wait()
        print(f"({module}: exit code {code}, {(time.time() - start) / 60:.1f} minutes)", flush=True)
    files = {path.name: path.read_bytes() for path in OUT_DIR.glob("*")}
    return files, time.time() - started


@app.local_entrypoint()
def main(modules: str) -> None:
    start = time.time()
    files, seconds = run.remote(modules.split(","))
    waited = time.time() - start - seconds  # for a free GPU, the container to start, the transfer
    hours = seconds / 3600  # the time billed, apart from the container's start
    OUT_DIR.mkdir(exist_ok=True)
    for name, data in files.items():
        path = OUT_DIR / name
        path.write_bytes(data)
        print(f"saved {path}")
    print(f"waited {waited / 60:.1f} minutes, then ran {hours * 60:.1f} minutes on an {GPU}")
    print(f"about ${hours * DOLLARS_PER_HOUR:.2f}")
