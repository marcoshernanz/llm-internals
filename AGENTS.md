# Agent Notes

- Optimize for learning and first-principles understanding, not speed of delivery.
- The goal is to learn how modern LLMs work under the hood. Follow what most major model families agree on, and research current practice before recommending a design. Where they disagree, pick one, say so, and name the alternative.
- Keep the model to its core, listed in `README.md`. Don't add mechanisms unless asked.
- Keep code and experiments minimal and pedagogical: one step per line, at most one dimension change per line, a shape comment on every tensor line, and a docstring of shape letters at the top of model files.
- `core/` is the active model. `deepseek_v41_flash/` is the earlier reimplementation of DeepSeek-V4.1-Flash, kept frozen for reference.
- Don't commit. The user commits.
