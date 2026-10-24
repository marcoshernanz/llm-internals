# Agent Notes

- Optimize for learning and first-principles understanding, not speed of delivery.
- Keep experiments minimal and pedagogical.
- Only learning matters. Results don't need to be accurate, fast or competitive: small models, short runs and free or old hardware are fine as long as the mechanism runs. Don't rule out a setup for reasons that only matter in production.
- The goal is to learn how modern LLMs work under the hood. Learn by imitating what the frontier actually does: follow what most major model families agree on, and research current practice before recommending a design. Where they disagree, pick one, say so, and name the alternative.
- Experiments are a way to see a mechanism run, not evidence for a design decision. Implement the frontier's choice and explain it, even when a simpler variant would score the same at this scale.
- If production does something, it's worth doing here even when its benefit doesn't show at our scale: a kernel that is no faster on our small tensors, or a mechanism that only pays off in large models. Build it to learn how it works, and don't skip it or rank it by what our measurements say.
- Learn problem-first and bottom-up: start from a problem we can see, understand why it happens, then build the fix and check the problem is gone.
- Explain GPUs with one NVIDIA datacenter GPU as the reference and its real numbers (the H100), not ranges across chips. Mention a newer NVIDIA GPU only where it differs in a big way. No other vendors, no legacy designs.
- Keep the model to its core, listed in `README.md`. Don't add mechanisms unless asked.
- Keep code simple and uncluttered, written to be read. Give each meaningful quantity its own line and name, so the computation reads step by step. Fold into that line a step that only selects or reshapes, and keep a formula on one line rather than naming its pieces. A shape comment on every tensor line, and a docstring of shape letters at the top of model files.
- The repo has two parts: the real project (`core/`, `post/`), which trains and measures, and the learning part (`learn/`), which holds each mechanism at its simplest, written to be read, not run. Its files are numbered and standalone, and each is the file before it plus one mechanism, as small as possible. Don't add anything so that an effect shows up when run or a comparison is fair, and don't bring in a concept the file before doesn't have, even one the mechanism needs to matter: say in chat what the toy can't show. Read `learn/README.md` before adding or changing a file there, and follow it.
- `core/` is the active model. `deepseek_v41_flash/` is the earlier reimplementation of DeepSeek-V4.1-Flash, kept frozen for reference.
- After every code change, run `uv run ruff format . && uv run ruff check . && uv run ty check` and fix every error before reporting back. ty is the only type checker.
- Don't commit. The user commits.
