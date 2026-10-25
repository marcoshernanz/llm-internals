# 3. Tools and environments: the model acts, something else answers

Until now the model wrote every token after the prompt. Here the generation pauses: the model
writes a call, something outside it answers, and the answer goes into the sequence as tokens the
model didn't write. The calls use Qwen's own tool tokens, and the toys prompt Qwen3-0.6B, Qwen's
chat model, with a few examples of the format: its post-training taught it those tokens, which the
base model never learned (001). The full version will teach the base model itself, fine-tuning it
first on traces of tool calls, from GSM8K's calculator notes. The full versions belong in `post/`,
stage 5 of `post/ROADMAP.md`.

| # | File | Builds on | Adds | Full version |
|---|---|---|---|---|
| 001 | [`001_tool_call.py`](001_tool_call.py) | — | the generation pauses for a calculator, whose result goes in as tokens the model didn't write | not yet |
| 002 | [`002_tool_mask.py`](002_tool_mask.py) | 001 | RL on those rollouts, the loss on the model's tokens only | not yet |
| 003 | [`003_environment.py`](003_environment.py) | 002 | an environment with state, which answers every call and gives the reward at the end | not yet |

## 001 Tool call

**Problem.** One calculation in eight in the model's GSM8K answers is wrong
(`post/eval/arithmetic_slips.py`): it does arithmetic in its head, one token at a time.

**Fix.** Let it call a calculator. It writes `<tool_call>37 * 48</tool_call>`, and the generation
pauses there. Python computes it, `<tool_response>1776</tool_response>` goes into the sequence, and
the model carries on, reading the result as context, until it ends its turn with `<|im_end|>`. The
sequence now has two writers, and `model_wrote` records which wrote each token.

**Why special tokens.** The pause has to see where the call ends, and the text arrives a token at
a time. `<tool_call>`, `</tool_call>` and `<|im_end|>` are one id each in Qwen's vocabulary, 151657,
151658 and 151645, never split or merged with their neighbours and never meaning anything else, so
the loop only compares the last id. A tag made of ordinary text fails: a first version wrote
`<calc>37 * 48</calc><result>`, and the tokenizer splits `</calc><result>` into `</`, `calc`, `><`,
`result`, `>`. The text never ended in `</calc>`, the calculator never ran, and the model wrote the
result itself, from its head.

**Why the chat model.** The base model never trained these tokens' rows of its embedding table,
which is also its head (`post/eval/embeddings.py`). Each is the same short vector as 290 other
tokens' rows, of length 0.350 against a median of 0.968, so its logit always equals theirs and it
never gets more than 1 in 291 of the probability. Where the marker goes, the base model wrote a
different junk token in each sample, " lucrative" or "严重等级", and used it as its marker.
Fine-tuning on the format is what trains those rows.

**How Qwen's real format differs.** The call is JSON, a function's name and its arguments. The
engine stops at `<|im_end|>`, the end of the turn, and the result comes back in a new turn. Here
the call is the bare expression, and the loop pauses at `</tool_call>` itself, as gpt-oss's
`<|call|>` token does.

**The change.** `calculator`, and `rollout`, which samples one token at a time and checks each id:
`<tool_call>` marks where a call starts, `</tool_call>` runs it and appends the response, and
`<|im_end|>` ends the rollout.

**Builds on —.** It samples one rollout at a time, as rollouts now stop at different places.

**Run.** The chat model calls the calculator in 6 samples of 8, but mostly doesn't end its turn:
after the answer it makes up another question and calls the calculator again. 1 sample in 8 ends
at the answer, the only one the reward counts.

## 002 The tool's tokens out of the loss

**Problem.** The RL loss pushes up every token of a rollout that beat its group. The calculator's
tokens are in the rollout too. Counted, they would teach the model to write
`<tool_response>1776</tool_response>` itself, making up what the tool says instead of calling it.
And the model never chose them, so they say nothing about how it acted.

**Fix.** Count only the tokens the model wrote, a_i in rollout i: the same mask as SFT's
(`1_finetuning/001`), where the prompt was context, not a target.

$$
\mathcal{L} = -\frac{1}{\sum_j |a_j|} \sum_i A_i \sum_{t \in a_i} \log p(y_{i,t} \mid y_{i,<t})
$$

**The change.** `model_log_probs` keeps the model's tokens only, `picked[counted]`, and the RL step
is the simplest of `2_rl`'s: the baseline (004) and the token-level mean (007). The rest of
`2_rl`'s recipe applies the same way, each piece weighting the model's tokens only.

**Run.** The reward rises from 0.12 to 0.88 in 10 steps, as more samples end their turn at the
answer.

**What the toy can't show.** What the mask prevents, a model that makes up the tool's results,
takes far longer to grow than 10 steps.

## 003 An environment

**Problem.** A calculator answers once and remembers nothing. Most environments have state, such as
a game, a codebase or a person, answer every turn, and decide when the episode ends and what it
was worth.

**Fix.** An environment with a `step`: the model's call goes in; an observation, a reward and
whether it's done come out. Here a guessing game holds a number, each `<tool_call>50</tool_call>`
is a guess, and the game's "Higher." or "Lower." comes back as the tool's response, the way an
agent reaches a codebase through a shell. The alternative, for a game or a simulated person, is an
environment that replies as the user, in a new turn. The game ends with reward 1 at the right
guess, and 0 after 7 guesses or one that isn't a number. A step's samples all play the same
number, so they form one group for the baseline. The reward comes only at the end, and every token
the model wrote in the episode shares its advantage; rewards for each turn are the alternative,
harder to define.

**The change.** `Game` replaces `calculator` and `reward`, and `rollout` hands each call to
`game.step` and appends its reply as the tool's response.
