"""Run one module of this repo on a cloud GPU. Run with ``modal run gpu.py --module core.bench``.

The local ``core`` package is copied in on every run, so uncommitted changes run too.
"""

import runpy

import modal

GPU = "L4"
TIMEOUT_SECONDS = 10 * 60  # longer than Modal's 5-minute default, still stops a stuck run

image = modal.Image.debian_slim(python_version="3.14")
image = image.uv_pip_install("torch==2.14.0", "numpy==2.5.3")  # the versions in uv.lock
image = image.add_local_python_source("core")
app = modal.App("deepseek-v4-1-flash", image=image)


@app.function(gpu=GPU, timeout=TIMEOUT_SECONDS)
def run(module: str) -> None:
    runpy.run_module(module, run_name="__main__")
