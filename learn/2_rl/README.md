# 2. RL: learning from the model's own samples

One prompt, "My favorite animal is the", 8 new tokens per sample, and a reward of 1 for a sample
that says "dog". It starts with sampling and counting, then trains on the model's own samples, one
mechanism at a time, up to GRPO and the fixes later recipes made to it. The full versions run on
GSM8K in `post/`, with their results in `post/ROADMAP.md`, stages 3 and 4.

| # | File | Builds on | Adds | Full version |
|---|---|---|---|---|
| 001 | [`001_pass_at_k.py`](001_pass_at_k.py) | — | the chance that k samples include a right one | `post/eval/pass_at_k.py` |
| 002 | [`002_rejection_sampling.py`](002_rejection_sampling.py) | — | keep the right samples, fine-tune on them, repeat | `post/rejection_sampling.py` |
| 003 | [`003_reinforce.py`](003_reinforce.py) | — | each sample's log-probability weighted by its reward | `post/reinforce.py` |
| 004 | [`004_baseline.py`](004_baseline.py) | 003 | the group's mean reward subtracted | `post/reinforce.py`, `baseline=True` |
| 005 | [`005_ratio_clipping.py`](005_ratio_clipping.py) | 004 | each batch reused for 4 updates, each token's ratio clipped | `post/ratio_clipping.py` |
| 006 | [`006_grpo.py`](006_grpo.py) | 005 | GRPO as DeepSeekMath wrote it: the standard deviation, each sample's tokens averaged, a KL penalty | `post/grpo.py` |
| 007 | [`007_token_mean.py`](007_token_mean.py) | 006 | every token in the batch averaged at once | `post/rlvr.py`, fix 1 |
| 008 | [`008_no_std.py`](008_no_std.py) | 007 | no division by the standard deviation | `post/rlvr.py`, fix 2 |
| 009 | [`009_no_kl.py`](009_no_kl.py) | 008 | no KL penalty | `post/rlvr.py`, fix 3 |
| 010 | [`010_zero_gradient_filtering.py`](010_zero_gradient_filtering.py) | 009 | no update on a group whose rewards are all equal | `post/rlvr.py`, fix 4 |
| 011 | [`011_clip_higher.py`](011_clip_higher.py) | 010 | the clip's upper bound raised, the lower kept | `post/rlvr.py`, fix 5 |
| 012 | [`012_truncated_importance_sampling.py`](012_truncated_importance_sampling.py) | 011 | each token weighted by the trainer's probability over the sampler's, capped | `post/rlvr.py`, fix 6 |

## 001 pass@k

**Problem.** One answer per question hides what the model can do. A question it gets right some of
the time looks the same as one it never gets right, and RL can only push up right answers the model
samples.

**Fix.** Sample n answers and count the c right ones. The chance that k of them, drawn without
replacement, include a right one is:

$$
\text{pass@}k = 1 - \binom{n-c}{k} \Big/ \binom{n}{k}
$$

**The change.** `pass_at_k`, and counting the samples that say the prompt's word.

## 002 Rejection sampling

**Problem.** SFT only learns from answers someone wrote. The model's own right answers are free
training data, in its own words.

**Fix.** Sample, keep the samples the reward accepts, fine-tune on them, and repeat with the trained
model.

**The change.** `kept = ids[right]`, and the loss is minus the kept samples' mean log-probability.

## 003 REINFORCE

**Problem.** Rejection sampling trains in rounds, on samples an older model wrote, and drops the
wrong ones.

**Fix.** Every step, sample, and weight each sample's log-probability by its reward:

$$
\mathcal{L} = -\frac{1}{N}\sum_i r_i \log p(y_i \mid x)
$$

On average over the samples, its gradient is minus the gradient of the expected reward. With rewards
of 1 and 0, it's rejection sampling on every step's own samples.

**The change.** `loss = -(rewards * log_probs).mean()`.

## 004 Baseline

**Problem.** With rewards of 1 and 0, wrong samples get weight 0 and teach nothing. Once every
sample is right, every sample is pushed up, whatever it says.

**Fix.** Subtract the group's mean reward, the group being the samples for the same prompt:

$$
A_i = r_i - \mathrm{mean}(r)
$$

Samples above the mean are pushed up and the rest down, and a group whose rewards are all equal
pushes nothing. On average this changes nothing: the probabilities always add up to 1, so the
gradient of their sum is 0.

**The change.** `advantages = rewards - rewards.mean()`, used in the loss in place of the rewards.

## 005 Ratio clipping

**Problem.** Sampling is most of each step, and its samples get one gradient step.

**Fix.** Reuse each batch for 4 updates. From the second on, the model training on the samples is
no longer the one that wrote them, so each token is weighted by its ratio: its probability now over
its probability when sampled. The ratio is clipped to 1 ± ε in the direction its advantage pushes:

$$
\mathcal{L} = -\frac{1}{N}\sum_i \sum_t \min\big(r_{i,t} A_i,\ \mathrm{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) A_i\big)
$$

A token pushed past its bound gets no more gradient from this batch. On the first update every
ratio is 1, and the gradient is the baseline's.

**The change.** `token_log_probs` keeps each token's log-probability. The old ones are computed
right after sampling, then the `ratios`, `clipped` and `objective` lines.

## 006 GRPO

**Problem.** None of its own: this is the original as DeepSeekMath wrote it, and the files after it
undo its additions one at a time.

**Fix.** Three things on top of ratio clipping:
- the advantage divided by the standard deviation of the group's rewards;
- each sample's tokens averaged, so every sample counts the same however long;
- a KL penalty toward a frozen copy of the starting model. Each token's estimate is p_ref/p minus
  log(p_ref/p) minus 1: never negative, and 0 where the two models agree.

Weight decay is 0 too. With every reward 0 and the model equal to the reference, AdamW's weight
decay alone moved the model off the reference. That gave the KL a tiny gradient, and Adam scales
each weight's step by that weight's own gradient size, so the step was far bigger than the gradient:
the KL reached 423 per token in one batch.

**The change.** The `reference` model, the `advantages` line, the `log_ref_ratios` and `kl` lines,
`- BETA * kl` in the objective, and `objective.mean(dim=-1).mean()`.

## 007 The token-level loss

**Problem.** Averaging each sample's tokens gives every token of sample i the weight:

$$
\frac{A_i}{N\,|y_i|}
$$

In a long sample the push is spread thin. Long wrong samples are punished less per token and short
right ones rewarded more, so wrong answers drift longer and right ones shorter.

**Fix.** Divide every token by the batch's total tokens, so a token's push depends only on its
sample's advantage:

$$
\frac{A_i}{\sum_j |y_j|}
$$

**The change.** `loss = -objective.mean()`, where GRPO had `-objective.mean(dim=-1).mean()`.

**What the toy can't show.** Every toy sample is 8 tokens, so the two lines give the same number.
They differ only when lengths differ, as GSM8K answers do.

## 008 No division by the standard deviation

**Problem.** With rewards of 1 and 0, and p the share of a group's samples that are right, the
standard deviation is:

$$
\sqrt{p(1-p)}
$$

Dividing by it scales up the groups whose samples are most one-sided, nearly all right or nearly
all wrong. Training then no longer follows the gradient of the expected reward.

**Fix.** No division: the advantage is the baseline's again.

$$
A_i = r_i - \mathrm{mean}(r)
$$

**The change.** The `advantages` line.

**What the toy can't show.** One prompt makes one group, so the division scales the whole batch by
one number, which Adam all but ignores. The problem is between questions.

## 009 No KL penalty

**Problem.** The KL penalty pulls the model back toward the model it started from. It guards
training on a learned reward model, which the model can learn to fool with samples it scores high
but that are wrong. A checked reward has nothing to fool, so the pull only holds the model back, and
the frozen copy costs a second model's memory and a forward pass over every sample.

**Fix.** No penalty. The objective is ratio clipping's again:

$$
\min\big(r_{i,t} A_i,\ \mathrm{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) A_i\big)
$$

With this, all three of GRPO's additions are undone: `diff 005_ratio_clipping.py 009_no_kl.py`
shows the token-level loss line, weight decay at 0, and the print.

**The change.** The `reference` model, `BETA`, `ref_log_probs`, the `log_ref_ratios` and `kl` lines,
and `- BETA * kl` go.

## 010 Zero-gradient filtering

**Problem.** When every sample in a group gets the same reward, the mean is that reward, so:

$$
A_i = r_i - \mathrm{mean}(r) = 0 \quad \text{for every sample}
$$

The group pushes nothing, but it still cost its sampling and its updates. And the token-level loss
divides every token by all the batch's tokens, these included, so they shrink the push of the
rest.

**Fix.** Drop every group whose rewards are all equal, and sample more until the batch is full of
groups whose rewards differ.

**The change.** `if (advantages == 0).all(): continue`.

**What the toy can't show.** One prompt makes one group, so dropping it drops the whole batch:
there's no rest to shrink, and refilling it means sampling the same prompt again, which the next
batch does. `post/rlvr.py` samples more questions until it has 8 whose rewards differ.

## 011 Clip-higher

**Problem.** The clip lets a token pushed up grow to at most:

$$
p_\text{new} \le (1 + \epsilon)\, p_\text{old}
$$

so its room to grow is a share of how likely it already was:

$$
p_\text{new} - p_\text{old} \le \epsilon\, p_\text{old}
$$

An unlikely token in a right sample can barely grow while the likely ones grow freely, so the
samples narrow onto what the model already prefers.

**Fix.** Raise only the upper bound: ratios clipped to 1 - 0.2 and 1 + 0.28, DAPO's values.
Loosening the lower bound too would let the tokens of wrong samples be pushed toward 0, which
narrows the samples as well.

**The change.** `EPSILON` becomes `EPSILON_LOW` and `EPSILON_HIGH`, and the clamp takes both.

## 012 Truncated importance sampling

**Problem.** The loss assumes the samples came from the model it trains. In practice a separate
copy samples, at lower precision or with other code, and gives the same tokens slightly different
probabilities. Tokens the sampler draws more often than the trainer would are pushed more often
than they should be, and the rest less.

**Fix.** Weight each token by how much likelier the trainer finds it than the sampler did. The
sampler's probability is the one it drew the token with; the trainer's is the old one. A token the
sampler found far less likely would get a huge weight and dominate, so the weight is capped:

$$
w_t = \min\left(\frac{p_\text{old}(y_t)}{p_\text{sampler}(y_t)},\ C\right)
$$

The weight is a number: no gradient goes through it.

**The change.** `sample` also returns each drawn token's log-probability, the `importance` line, and
`importance *` in the objective.

**What the toy can't show.** Its sampler is the model itself, at the same precision, so every weight
is 1 up to rounding. In `post/rlvr.py` the sampler is a bf16 copy.
