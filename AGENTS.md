# Agent Notes

- Optimize for learning and first-principles understanding, not speed of delivery.
- Keep experiments minimal and pedagogical.
- Only learning matters. Results don't need to be accurate, fast or competitive: small models, short runs and free or old hardware are fine as long as the mechanism runs. Don't rule out a setup for reasons that only matter in production.
- The goal is to learn how modern LLMs work under the hood. Learn by imitating what the frontier actually does: follow what most major model families agree on, and research current practice before recommending a design. Where they disagree, pick one, say so, and name the alternative.
- Experiments are a way to see a mechanism run, not evidence for a design decision. Implement the frontier's choice and explain it, even when a simpler variant would score the same at this scale.
- Keep the model to its core, listed in `README.md`. Don't add mechanisms unless asked.
- Keep code simple and uncluttered: one step per line, at most one dimension change per line, a shape comment on every tensor line, and a docstring of shape letters at the top of model files.
- `core/` is the active model. `deepseek_v41_flash/` is the earlier reimplementation of DeepSeek-V4.1-Flash, kept frozen for reference.
- Don't commit. The user commits.
