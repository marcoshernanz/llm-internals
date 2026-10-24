# Learning

The repo has two parts. The real project (`core/`, `post/`) trains and measures: Qwen3 in our own
code, packing, the fused loss, GSM8K, runs on Modal, and their results in `post/ROADMAP.md`. This
part, `learn/`, holds the same mechanisms at their simplest, to read: one file per mechanism, each
the file before it with that one mechanism added.

## Layout

Each folder is one ladder, numbered in the order to read it:

| Folder | Ladder |
|---|---|
| [`1_finetuning/`](1_finetuning/) | SFT, then the same training in fewer bytes: LoRA, NF4, QLoRA |
| [`2_rl/`](2_rl/) | learning from the model's own samples: pass@k, rejection sampling, REINFORCE, then GRPO and the fixes later recipes made |

Each folder's `README.md` is its index: every file, the file it builds on, the problem it solves,
the fix, the lines that changed, and its full version in the real project.

## How a file is written

- **One mechanism per file,** numbered `001`, `002`, … in reading order and named after the
  mechanism.
- **Each file is the one before it, copied whole, plus its mechanism.** `diff` two neighbours and
  you see the mechanism and nothing else. A file that starts something new instead says so in its
  folder's index: it builds on "—".
- **Standalone.** No imports from other numbered files: a name that starts with a digit can't be
  imported, and a whole file reads top to bottom. The only import from the rest of the repo is the
  model and tokenizer loaders in `post/qwen3.py`.
- **Written to be read, not run.** Nothing is added so that an effect shows up when run, or so that
  a comparison is fair: no second runs, extra logging, new rewards, stop rules or masks. No concept
  the file before doesn't have, either, even one the mechanism needs to matter. Where a toy can't
  show a mechanism's problem, its folder's index says so and points to the full version.
- **As small as possible.** Plain PyTorch, a toy prompt and a toy reward. No packing, fused losses,
  caches or batching tricks: those belong to the real project.
- **Shape comments only,** one on every tensor line, and a docstring of shape letters at the top.
  Constants first, then the functions, then `main`.

## Adding a file

A mechanism usually lands in both parts: its full version in `post/`, with its results in
`post/ROADMAP.md`, and its file here.

1. Copy the folder's last file to the next number, named after the new mechanism:
   `cp learn/2_rl/012_truncated_importance_sampling.py learn/2_rl/013_<mechanism>.py`.
2. Change only what the mechanism needs.
3. `diff` it against the file before. Every changed line must belong to the mechanism.
4. Add its row and its section to the folder's `README.md`: the problem, the fix, the change, and
   what the toy can't show, if anything.
5. Run `uv run ruff format . && uv run ruff check . && uv run ty check`.

A new ladder gets the next folder number, such as `3_reward_models/`, and starts again at `001`.

## Running one

```bash
uv run modal run --detach gpu.py --modules learn.2_rl.003_reinforce
```
