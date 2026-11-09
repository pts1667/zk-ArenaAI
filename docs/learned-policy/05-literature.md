# What the literature suggests for the learned policy

2026-11-09.  A literature review asked for by the user: architectural changes, optimizations and insights for the training speed,
efficiency and strength of the model of `docs/model.md`, and for the policy running the economy (`04-economy.md`).  Made by a
research agent from the arXiv API (one request at a time, at least 3 s apart, cached) and a few web pages; every arXiv id here was
fetched in that session.  AlphaStar's Nature paper and ROA-Star aren't on arXiv: what's said of them comes from AlphaStar Unplugged,
SCC and the NeurIPS page.  DI-star's documentation wasn't read.

Tags: **[P]** a paper measured it, in its own setting; **[E]** extrapolated to this model, untested; **[C]** from this repo's code
and docs.

**Taken up so far** (2026-11-09, the user's order: S1-S5 and P1+P2, then a short `clone-7` baseline that P3-P8, C1 and C4-C8 are
measured against, `03-training.md` 11.21 and 11.22).  For S1-S3, speeds are one cloning step of 16 stretches, the main process
alone on samples made beforehand (`policy_clone_train.forward`, backward and the optimizer), measured while `army-6` was
recording on every core:

| | What was done | Measured |
|---|---|---|
| S1 | `--accumulate N`: a step learns from N forward passes of `--samples`.  32 stretches a pass don't fit in 85% of the 12 GB card (16 peak at 5.0 GB), so a bigger batch has to be accumulated | the baseline's runs |
| S2 | Workers send the frames without what only the heads read (`pm.HEAD_FIELDS`), and that for the counted decisions alone; floats as float16 (`Batch.half`); `join(..., device)` pads on the GPU, so no host memory or copy is padding | a sample 11.6 MB through the pipe to 4.2 MB; a step 512 ms to 341 ms (1,501 to 2,254 decisions a second); host-to-GPU copies were 29% of the GPU's time (83 ms a step, pageable), and are no longer among its 15 biggest costs |
| S3 | At the 10.5M actor (`policy_latent.SIZE`), the biggest GPU cost was the time transformer's attention scoring every pair of a stretch's ~2,640 tokens, ~93% of them masked.  `--attention flex` (the default on CUDA): flex attention with a block mask of the window, and the age bias as 18 dimensions appended to each head's queries and keys (`TimeBlock.bias_dims`).  As a score_mod the bias was 2.4x slower than dense: its gradient came a pair of tokens at a time, as atomic adds into a table of ~136.  `--compile default` / `reduce-overhead` (CUDA graphs) on the time transformer add little on top, and are off | 8 stretches a step: 230 ms dense, 157 ms flex, 155 compiled, 147 as CUDA graphs (peak 4.5 GB, 3.6 as graphs); flex matches dense to 1e-4 in outputs and gradients (`test_flex_attention_reads_and_learns_as_the_bias_does`) |
| S1b | The trainer, not the GPU, was then the limit.  It took a whole step's samples before its first forward pass, so the GPU and the workers waited on each other in turn; each part's are now taken as it's needed.  A worker cuts `--per-read` 4 stretches one after another from each game it decodes (a record is changes along the game, so it decodes from the start), each frame featurized once (`_samples`); the trainer draws them at random from 4 passes' worth made.  `featurize._masks` builds every unit's reachable regions at once (a loop over units was a quarter of featurize) | the trainer with 12 workers: 791 to 1,145 decisions a second (overlap), 1,356 (per read, masks), 1,397 (shared frames); 15 workers 1,462.  Workers alone: 1 393, 4 1,431, 8 2,137, 15 2,392 (8 cores, 16 threads); the rest is featurize, 1,500 small numpy calls a frame, and the main process's core.  Since, featurize's loops are numba kernels, the same tensors to the bit: 1.6 ms a frame to 0.4, a worker's decision 2.85 ms to 1.54 (11.22) |
| S4 | `--muon-lr`: Muon on the actor's hidden matrices (10,399,232 of 10,537,808; at the old size 1,857,280 of 1,924,568), AdamW on the rest and on the heads giving logits (`policy_latent.answers`, `frame_vae.Optimisers`) | the baseline's runs |
| S5 | PPO: `passes` 1 (was 2); `--ema 0.99`, a moving average of the weights updated each optimizer step, plays the tests and is saved; the economy heads (place, offset, wreck) join the entropy-kept heads | untested until PPO runs from a clone |
| P1 | Every controllable unit reads the frame (`unit_read`, a cross-attention block) before any is picked; its anchor key is `key_anchor` of that, with its sine code (`Context.anchor_keys`) | the baseline |
| P2 | `head_ordered` on the same reading: per unit, the logit that the script tells it anything this decision; BCE over the units an action could pick, weight `--ordered-weight` 1; held out as `ordered_bce` against `ordered_base` (guessing the share told) | the baseline |
| P3 | DAgger: `policy_clone_eval.py --record` keeps the cloned actor's games, the scripts only advising its side (libzkscripts ABI 7's label seats); `policy_clone_train.py --mix DIR --mix-share --init` fine-tunes on them with `army-6` | next, on the better clone |
| P4 | `--eval-only --blind`, `--act-dropout` | blind: NLL of actions 2.84 to 3.12, worse on every measure; no dropout needed |
| P5 | `--size small` (`clone-7s`, 1.9M against 10.5M) | worse on every measure held out, 0.24 nats on actions and widening, though it plays as well so far: width helps |
| P6 | `--finish-winners` (the last share of steps on decided games' winners), `--temperature` in play | in `clone-8`; T 0.8 changes nothing measurable in play |
| P7 | `policy_train.py --init` from a clone: frozen copy and an annealed k3 KL to it, a cloning loss on the scripts' advice, dual-clip, HL-Gauss critic, critic warm-up, `--fine-sim` | smoke run only; PPO waits for the encoder's retrain (the user) |
| P8, C5 | z: the army script's knobs and preset, and in hindsight the side's structures and army at minutes 3, 6, 10; `--z`, `--z-dropout`; `zindex` | in `clone-8`: worth 0.03 nats held out; in play, z from the variant gives the best economy of six evals, within the noise |
| C1 | not applicable: no queue in the sim or scripts | |
| C4 | `--economy-weight` | `clone-8` at 2.5: builds 7.81 to 7.56 nats, reclaims 3.80 to 3.56; fights 4.44 to 4.67 |
| C6 | `SimEnv.policy_economy_share`, `--economy-start`/`--economy-end` | |
| C7 | `--income-weight`, a zero-sum income term | smoke run only |
| C8 | `policy_clone_eval.py`: wins and each side's income, extractors and value at minutes 3, 6, 10 | six evals of 64 games: 2-5% wins, 77-80% losses whatever the clone, temperature or z; 5-6 extractors at minute 10 to the script's 15-18 |

Checks: the policy tests; decisions picked out of a bigger batch through the split, joined on the device, score as a batch of
their own (1e-4), and within 0.05 nats as float16 (`test_decisions_score_the_same_picked_out_of_a_bigger_batch`).

## A. Speed and efficiency, ranked

**S1. Much bigger cloning batches, the learning rate retuned** (speed high, quality medium to high).  `--samples 16` (768
decisions a step) to 64-128 stretches, or gradient accumulation; learning rate by about the square root, warm-up kept.  A step is
bound by the main process issuing kernels and taking in ~120 MB, per step not per sample [C, 11.20].  AlphaStar Unplugged
(2308.03526): minibatch 8,192 / 16,384 / 32,768 gave 70 / 79 / 84% against very_hard, and a learning rate tuned early misleads
[P].  SCC (2012.13169): batch 3,850 to 7,700 better in speed and peak; final 15,400 [P].  OpenAI Five (1912.06680): 123k to 983k
learned ~2.5x faster (RL) [P].  Risk: fewer updates per sample; compare at equal samples.

**S2. Less through the pipe** (speed medium, up to 11.20's 1.4x).  Don't send the decision frames twice (`seen` and `heads`); fp16
unit features; a shared-memory ring of preallocated tensors with only indices sent; pinned, non-blocking, double-buffered copies.
Sample Factory (2006.11751) is this design [P]; Pluto's engine uses shared memory.  About a day; low risk.

**S3. Fewer small kernels** (speed medium to high).  Fuse `action_vectors` (8 Linear/Embedding adds) into one concat-then-Linear;
CUDA graphs for the time blocks and heads, shapes padded to buckets; FlexAttention (2412.05496) for `TimeBlock`, the age and action
biases a `score_mod` and the frame-causal window a `BlockMask` [P].  A float `attn_mask` stops PyTorch's flash kernel and
materialises heads x Q x K and its gradient [C].  Packing without cross-contamination: 2107.02027 [P].  Risk: `torch.compile` hung
before (10.3); keep the present path as the reference, with an equality test.

**S4. Muon for the actor's matrices** (low to medium): ~2x compute efficiency over AdamW in LLMs (2502.16982) [P]; the encoder
already trains with it [C].

**S5. PPO.**  OpenAI Five: sample reuse 2-3x can halve learning speed, 8x can stop it; they kept reuse ~1 and staleness 0-1 [P].
Generals.io (2606.23348): top-advantage subsets halve an iteration, and an EMA of the weights plays best [P].  PPO-EWMA
(2110.00641): batch size changes without retuning [P].

**S6. Not worth it:** caching encoder latents on disk.  History is 6 of 54 frames, the encoder is one batched call, and the cache
would be ~38 GB.

## B. Strength, ranked

**P1. Unit keys that know the frame before the anchor is scored** (high; low to medium cost).  `anchor_logits` is
`q_anchor(slot) · [key_unit(hu) ‖ sine code]`, `hu` an MLP of the unit's own features [C]: "my region is threatened" or "I'm in a
cluster" can't reach the key.  A unit reads the frame (`read_unit`) only after it's picked; 11.6's per-unit actor read it before.
In order of cost: every controllable unit cross-attends to `now` once a decision (`read_unit` for all), `key_anchor =
key_unit(hu + read)`; then one self-attention layer over the ≤256 own units; then an MLP / concat scorer on [slot ‖ key] as SCC's.
AlphaStar Unplugged: a units transformer feeds both pointer heads [P]; SCC: group transformer and a selection pointer conditioned
on the action [P]; Relational Deep RL (1806.01830); Perceiver IO (2107.14795) per-output queries [P].  Test: one short run against
`clone-6`'s curve at the same step.

**P2. A dense per-unit side loss for selection** (medium; low cost).  For every controllable unit, BCE on "is it ordered at this
decision" (and optionally its kind): a label for every unit at every decision, not one softmax gradient an action [E].  Also says
how predictable selection is.  By analogy: UNREAL (1611.05397), JueWu-SL multi-task (2011.12582), Memory Gym's TrXL winning only
with an auxiliary loss (2309.17207) [P].

**P3. DAgger with the script as the oracle** (high in closed loop; medium cost).  The cloned actor plays in the sim, the C++
scripts label the states it reaches; 10-30% such data mixed in, repeated.  DAgger (1011.0686): no-regret, fixes errors compounding
with the horizon [P].  Every check so far is open-loop on the script's own states [C].  Risk: the scripts' internal state
(`group_plans`, roles, timers) must update properly while their orders aren't carried out.  It also labels the economy in states
the script never reaches.

**P4. Check the action tokens for copying** (medium, a guard; very cheap).  Imitators learn to repeat the previous action
(2010.14876, 1905.11979) [P]; AlphaStar Unplugged dropped its previous map-location argument for "too much overfitting" [P].
Compare `--blind` with normal at the same checkpoint, later in closed loop; if needed, drop whole frames' action tokens with
p 0.2-0.5.

**P5. A wider action side, trained longer** (medium to high).  Unplugged: half the width cost 83 to 76%, and play grows about
with the log of frames trained [P]; imitation scaling laws 2307.09423, 2411.04434 (coefficients depend on architecture) [P]; game
agents are usually undersized (2210.00849) [P].  Against: Gym-µRTS (2105.13807) found size didn't matter in RL from scratch [P].
D 256 for the plan, unit and heads first; one two-point test at equal samples.

**P6. Finish cloning on the best data; play at a lower temperature** (medium; very cheap).  Unplugged: fine-tuning on high-rated
wins 84 to 89%; temperature 0.8 +4-5 points [P].  SCC: winning high-rated games >97%, a quality subset ~10 points over the larger
set [P].  CS:GO BC (2104.04258): big noisy data, then a small good set [P].  Here: the last 5-10% of steps on decided games'
winners and the strongest variants.

**P7. The cloning-to-PPO recipe** (high for RL) [P]: warm the critic up with the actor frozen (SCC: 50 steps); a KL to the frozen
clone, annealed but not to zero (AlphaStar, SCC, VPT 2206.11795, piKL 2112.07544 / 2210.05492); a cloning loss on script data
kept in the PPO batch (2402.02868: NetHack 5K to 10K; Kickstarting 1803.03835); dual-clip PPO (1912.09729); an HL-Gauss
classification critic (2403.03950: the return here is bounded in (-1, 1)); an EMA of the weights to play (2606.23348).
RAISocketAI (2402.08112): BC 71% to BC+PPO 88%, about 72 GPU-days [P].

**P8. Conditioning on a strategy, z** (medium, more in RL).  AlphaStar / SCC: z (build order, counts) conditions the policy and
gives pseudo-rewards, a value head per reward; z-conditioned agents 35-65% against the main one [P].  ROA-Star: goal-conditioned
exploiters.  MGG (2110.14221): macro goals from demonstrations, more diversity [P].  Here [E]: condition on the script variant, or a
z computed in hindsight from a game's own future.

**P9. Keep:** canonical action masking (2006.14171; Gym-µRTS: 0 to 82% wins with masking) [P]; the set-likelihood anchor loss
(1911.09996, 1511.06391), with no need for a DETR-style Hungarian loss.

## C. The economy in the policy

| Agent | Economy learned? | Type and place |
|---|---|---|
| AlphaStar / Unplugged (2308.03526) | yes, one network | heads in turn: function, delay, queued, unit pointer, target pointer, a 2-D map point from upsampled feature planes; z conditions and gives pseudo-rewards |
| SCC (2012.13169) | yes | the same; z sampled per game with a matching reward; value and GAE per reward |
| Pluto (README) | yes, "a single neural network ... plays the whole game ... no scripted play" | the opening chosen by a bandit over per-opponent wins; heads not documented |
| OpenAI Five (1912.06680) | no: items, ability order and courier scripted ("could ultimately perform better" learned) | |
| Gym-µRTS / RAISocketAI (2105.13807, 2402.08112) | yes | factored produce type x direction from the worker; masking essential |
| TStarBots / HierNet-SC2 (1809.07193, 2209.11553) | through macro actions | the net picks a scripted routine; HierNet 93% against level 7 on 4 GPUs |
| Command-conditioned PPO, MicroRTS (2610.11663) | split | a bandit strategist sets economy and army commands, PPO carries them out; over flat PPO, more wins against 3 of the 4 strongest (0.55 to 0.97, 0.01 to 0.34) |
| JueWu-SL (2011.12582) | supervised macro and micro end to end | human level |

* **C1. One network, the same action loop**, constructors as anchors with build, reclaim (pointer to a wreck), assist / repair /
  guard (pointer to an own unit).  A **queued** head (append or replace), as AlphaStar's.  Re-measure actions per decision.
* **C2. Structure type** by a head like produce, masked by the constructor's options.
* **C3. Place by pointers to candidates, not a map head.**  Extractors: a pointer to site tokens (free / ours / contested; check the
  24-site cap on the pool).  The rest: region plus fine offset, snapped to a legal spot by the script's placer.  Later, candidate
  placement tokens from the placer.  Action Space Shaping (2004.00980) [P]; a dense 2-D head needs a spatial stream this model
  hasn't got.
* **C4. Labels:** the economy script's orders, a new data version, army and economy cloned together, then DAgger (P3); constructor
  metrics tracked apart, up-weighted 2-3x if they lag.
* **C5. A build-order z:** first N structures and counts at minutes 3 / 6 / 10, in hindsight; sampled or chosen by a bandit in
  play; in PPO a reward for following it early, a value head per reward term (SCC, RAISocketAI).
* **C6. From the script's economy to the policy's:** in PPO the policy runs the economy in a share p of games, from ~0.25 to 1, or
  JSRL (2204.02372): the script plays the first T minutes, T annealed to 0.  The KL to the clone kept longer on the economy heads; a
  cloning loss on opening states.
* **C7. Reward:** the present potential counts structures at cost x build progress, and an extractor pays back within its ~60 s
  horizon [E].  Gym-µRTS agents overfit shaped rewards (more barracks, no more wins) [P]; RAISocketAI annealed to win / loss [P];
  Action Guidance (2010.03956) is gentler [P].  At most a small zero-sum income term annealed by β; never a reward for structures.
* **C8. Measure the economy apart:** income at minutes 3 / 6 / 10, metal lost and stalled, extractors held, against the script's
  economy with the army fixed; then wins.

## D. The user's open questions

**Full attention over all past frames, through summary tokens.**  In cloning, Unplugged: no memory 84%, LSTM 70%, transformer 85%
[P].  Transformers help memory, not credit assignment (2307.03864) [P]; Memory Gym: TrXL won only with an auxiliary loss, GRU won
all the endless tasks (2309.17207) [P]; RLBenchNet (2505.15040): Mamba-2 matched TrXL at 8x less memory [P]; GRUs strong
(2303.01859, 2110.05038) [P].  Mechanisms: RMT 2207.06881, Block-Recurrent 2203.07852, Compressive 1911.05507, Perceiver AR
2202.07765, RATE 2306.09459, GTrXL 1910.06764, AMAGO 2310.09971.  Advice [E]: measure window 1 against window 4 first (the
featurizer already carries memory: memory and blip tokens, time since contact); if longer helps, add a few older frames (-30 s,
-60 s, -120 s) pooled to 8 tokens each, which keeps training one pass; recurrent memory tokens only if that's too short, as they
bring back BPTT or stale stored states (10.5); then Gated DeltaNet (2412.06464), or xLSTM (2410.22391) for fast recurrent play.

**FlexAttention, one token per acting unit.**  A good fit (a variable 0-16 action tokens a frame is the document-mask case), but
more detail of the script's past orders is the copying channel (P4).  Run `--blind` first: if blind ≈ normal, per-unit tokens
cost without gain.  Port to FlexAttention for speed either way (S3).

**A ~10M actor on 12 GB.**  Parameters and AdamW ~160 MB; activations at D 256 and 64 samples a step a few GB in bf16;
checkpointing if needed.  Launch-bound steps may barely slow.  For cloning: Unplugged's width result, 2307.09423, 2411.04434,
2210.00849.  For RL, scaling with normalisation: SimBa 2410.09754, BRO 2405.16158, BBF 2305.19452; MoE 2402.08609 if dense
stalls.  Put the parameters in unit context and the action decoder; learning rate by µP (2203.03466) or a retune; grow from the
trained clone (OpenAI Five's "surgeries").  Data isn't the limit (401M recorded, 15M used; repetition: 2305.16264).

**The policy running the economy.**  Yes: section C.  It's the hardest part to keep stable in RL, which is why AlphaStar and SCC
add z, a KL to the clone, and pseudo-rewards.

## E. Already right, and not to do

Right: the frozen Perceiver encoder; actions in turn with pointers (AlphaStar, SCC, MAT 2205.14953); the set-marginal anchor loss;
canonical masking; the separate privileged critic; a zero-sum potential reward annealed to win / loss; a decision every 3 s.

Not to do: label smoothing (1906.02629; Unplugged did better sharper, T 0.8); an LSTM or GRU as cloning memory (Unplugged 70 vs 84%);
offline Q-learning on the script data (Unplugged: deep offline Q 0%, CRR / BCQ-likes at best BC's 83-84%; only offline actor-critic
with a fixed behaviour critic beat it, 93 vs 88%); reusing PPO samples 4-8 times; self-play or a league before cloning-to-PPO works
(Gym-µRTS self-play underperformed; TStarBot-X: a scaled-down AlphaStar league fails without extra tricks); rewarding structures
directly; expecting memory to fix credit assignment.

## F. The order it suggests

1. S1 + S2, measured once in decisions a second.
2. P1 + P2 in one run: anchor right against `clone-6` at the same samples.
3. P4 (`--blind`), which decides the FlexAttention question.
4. C4: re-record with economy labels, re-measure slots; then C5 / P8.
5. P3 DAgger, once the actor has a sim forward.
6. P7 + C6 for PPO.
7. P5 widening.

## References (fetched from the arXiv API)

RTS and MOBA: 2308.03526 AlphaStar Unplugged; 2012.13169 SCC; 2011.13729 TStarBot-X; 1809.07193 TStarBots; 2209.11553
HierNet-SC2; 1912.06680 OpenAI Five; 1912.09729, 2011.12692 Honor of Kings RL; 2011.12582 JueWu-SL; 1812.07887 Hierarchical Macro
Strategy; 2110.14221 MGG.  MicroRTS, Lux, Generals.io: 2105.13807 Gym-µRTS; 2006.14171 invalid action masking; 2402.08112
RAISocketAI; 2610.11663 command-conditioned PPO; 2010.03956 Action Guidance; 1707.03743 macromanagement from replays; 2304.13004
Lux centralised control; 2606.23348, 2507.06825 Generals.io.  Architecture: 2103.03206, 2107.14795, 2202.07765 Perceiver family;
1810.00825 Set Transformer; 1703.06114 Deep Sets; 1806.01830 Relational Deep RL; 1506.03134 Pointer Networks; 1511.06391 Order
Matters; 1911.09996 orderless multi-label; 2005.12872 DETR; 2205.14953 Multi-Agent Transformer.  Memory: 1901.02860
Transformer-XL; 1910.06764 GTrXL; 2207.06881 RMT; 2203.07852 Block-Recurrent; 1911.05507 Compressive; 2203.08913 Memorizing;
2404.07143 Infini-attention; 2306.09459 RATE; 2310.09971 AMAGO; 2307.03864; 2309.17207 Memory Gym; 2303.01859 POPGym; 2110.05038;
2505.15040 RLBenchNet; 2412.06464 Gated DeltaNet; 2405.04517 xLSTM; 2410.22391 LRAM.  Systems and optimisation: 2412.05496
FlexAttention; 2307.08691 FlashAttention-2; 2107.02027 packing; 2006.11751 Sample Factory; 2206.10558 EnvPool; 2502.16982 Muon;
2203.03466 µP.  Scaling: 2307.09423; 2411.04434; 2210.00849; 2001.08361; 2203.15556; 2305.16264; 2410.09754 SimBa; 2405.16158 BRO;
2305.19452 BBF; 2402.08609 MoE for RL.  Imitation and RL fine-tuning: 1011.0686 DAgger; 1905.11979 causal confusion; 2010.14876
copycat; 2104.04258 CS:GO BC; 2206.11795 VPT; 1803.03835 Kickstarting; 1511.06295 Policy Distillation; 2402.02868 forgetting;
2112.07544, 2210.05492 piKL; 2204.02372 JSRL; 2311.02198 IBRL; 2302.02948 RLPD; 2412.07762 WSRL; 1709.10087 DAPG; 1812.06298
Residual Policy Learning; 1707.05300 reverse curriculum; 2004.00980 action space shaping.  PPO and value: 1802.01561 IMPALA;
2009.04416 PPG; 2006.05990; 2110.00641; 2403.03950 HL-Gauss; 1809.04474 PopArt.  Other: 1611.05397 UNREAL; 2007.05929 SPR;
2301.04104 DreamerV3; 1906.02629 label smoothing; 2108.12409 ALiBi; 2402.04494; 2311.17190 Minimax Exploiter.

Web: the Pluto README (github.com/tscmoo/pluto), ROA-Star at NeurIPS 2023, and ar5iv full texts of SCC, AlphaStar Unplugged,
Gym-µRTS, RAISocketAI and OpenAI Five.
