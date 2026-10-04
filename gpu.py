"""Run one module of this repo on a cloud GPU. Run with ``modal run gpu.py --module core.bench``.

The local ``core`` package is copied in on every run, so uncommitted changes run too. Files the
module writes to ``out/`` are copied back to the local ``out/``.
"""

import runpy
from pathlib import Path

import modal

GPU = "L4"
TIMEOUT_SECONDS = 10 * 60  # longer than Modal's 5-minute default, still stops a stuck run
OUT_DIR = Path("out")

image = modal.Image.debian_slim(python_version="3.14")
image = image.uv_pip_install("torch==2.14.0", "numpy==2.5.3")  # the versions in uv.lock
image = image.add_local_python_source("core")
app = modal.App("deepseek-v4-1-flash", image=image)


@app.function(gpu=GPU, timeout=TIMEOUT_SECONDS)
def run(module: str) -> dict[str, bytes]:
    runpy.run_module(module, run_name="__main__")
    return {path.name: path.read_bytes() for path in OUT_DIR.glob("*")}


@app.local_entrypoint()
def main(module: str) -> None:
    files = run.remote(module)
    OUT_DIR.mkdir(exist_ok=True)
    for name, data in files.items():
        path = OUT_DIR / name
        path.write_bytes(data)
        print(f"saved {path}")
