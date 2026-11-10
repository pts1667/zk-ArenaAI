# What the literature suggests

A literature review from 2026-11-09, which the user asked for.  It covers training speed, the model's strength, and the policy
running the economy.  A research agent made it from the arXiv API and a few web pages.  Every arXiv id here was fetched in that
session.  AlphaStar's Nature paper and ROA-Star aren't on arXiv.  What's said of them comes from AlphaStar Unplugged, SCC and the
NeurIPS page.

## Taken up, and what it gave

| | Suggestion | Where it is now | Measured |
|---|---|---|---|
| S1 | Bigger cloning batches | `--accumulate` | 16 stretches a pass fit in 5 GB; 32 don't fit in 85% of the card |
| S2 | Less through the workers' pipe | head-only fields for counted decisions, float16, padding on the GPU | a sample 11.6 MB to 4.2 MB; a step 512 ms to 341 ms |
| S3 | Fewer, bigger kernels | `--attention flex`, the age bias as extra query/key dimensions | 8 stretches 230 ms to 157 ms.  CUDA graphs and compile add little and are off |
| S4 | Muon for the hidden matrices | `--muon-lr`, also in the encoder | the baseline's runs use it |
| S5 | PPO sample reuse 1, an EMA of the weights | `passes` 1, `--ema 0.99` | not yet measured in a full run |
| P1 | Unit keys that read the frame before the anchor is scored | `unit_read`, `key_anchor` | in every clone |
| P2 | A dense per-unit loss for who is ordered | `head_ordered`, `--ordered-weight` | BCE 0.148 against guessing 0.331 |
| P3 | DAgger with the scripts as the oracle | `--record`, `--mix`, the pipeline's rounds | not run yet |
| P4 | Check for copying past actions | `--blind`, `--act-dropout` | blind is worse everywhere (NLL 2.84 to 3.12): no dropout needed |
| P5 | Width | `--size small` against the default | the small actor is worse held out, and the gap widens |
| P6 | Finish on winners; lower temperature in play | `--finish-winners`, `--temperature` | in `clone-8`; T 0.8 changes nothing measurable |
| P7 | The cloning-to-PPO recipe: critic warm-up, KL to the clone, cloning loss, dual clip, HL-Gauss | `policy_train.py --init` | smoke run only |
| P8, C5 | A strategy z | `--z`, `zindex` | 0.03 nats held out.  Told its variant in play, the best economy of six evals, within the noise |
| C4 | Economy labels, weighted | `--economy-weight` | at 2.5: builds 7.81 to 7.56 nats, fights 4.44 to 4.67 |
| C6 | The policy taking over the economy | `--economy-start` / `--economy-end` | not yet in a full run |
| C7 | A small zero-sum income term, no reward for structures | `--income-weight` | smoke run only |
| C8 | Measure the economy apart | `policy_clone_eval.py` | clones 5-6 extractors at minute 10, the script 15-18 |

The papers behind these:

* **Batch and learning rate**: AlphaStar Unplugged (2308.03526), SCC (2012.13169), OpenAI Five (1912.06680).
* **Pipeline**: Sample Factory (2006.11751).  **Attention**: FlexAttention (2412.05496).  **Muon**: 2502.16982.
* **PPO reuse and EMA**: OpenAI Five; Generals.io (2606.23348).
* **Units reading the frame**: Unplugged's units transformer; Perceiver IO (2107.14795).
* **Side losses**: UNREAL (1611.05397), Memory Gym (2309.17207).
* **DAgger**: 1011.0686.  **Copying**: 2010.14876, 1905.11979.
* **Width**: Unplugged (half the width cost 83% to 76%), 2307.09423, 2411.04434, 2210.00849.
* **Fine-tuning on the best data**: Unplugged (84% to 89%), SCC, CS:GO BC (2104.04258).
* **Cloning to RL**:
  * a KL to the clone: AlphaStar, SCC, VPT (2206.11795) and piKL (2112.07544, 2210.05492);
  * a cloning loss kept in RL: 2402.02868 and Kickstarting (1803.03835);
  * dual clip: 1912.09729;
  * HL-Gauss: 2403.03950;
  * RAISocketAI (2402.08112) went from 71% with cloning to 88% with PPO after it, in about 72 GPU-days.
* **z**: AlphaStar, SCC, MGG (2110.14221).

## Not taken up yet

* **More on unit context** (after P1):
  * one self-attention layer over the controllable units;
  * an MLP scorer on [action ‖ key], as SCC's.
* **Wider, not only narrower** (P5): Unplugged's play grew about with the log of frames trained.  Gym-µRTS (2105.13807) found
  size didn't matter in RL from scratch.  If the 10M actor stalls:
  * µP (2203.03466) or a retune of the learning rate;
  * grow it from the trained clone;
  * normalised scaling for RL: SimBa (2410.09754), BRO (2405.16158);
  * MoE (2402.08609).
* **z in RL**:
  * sample z per game, or choose it by a bandit over wins;
  * pay a pseudo-reward for following its build order early, with a value head per reward term (SCC, RAISocketAI).
* **The economy in RL**:
  * JSRL (2204.02372) as an alternative to a share of games: the script plays the first T minutes, with T annealed to 0;
  * keep the KL to the clone longer on the economy heads;
  * a cloning loss on opening states;
  * measure the economy against the script's with the army fixed.
* **Memory** (Unplugged: no memory 84%, LSTM 70%, transformer 85%):
  * the window covers 12 frames, and the featurizer carries memory tokens;
  * if longer helps, add a few older frames (-30, -60, -120 s), each pooled to 8 tokens;
  * recurrent memory tokens only if that's too short, since they bring back BPTT;
  * for fast recurrent play: Gated DeltaNet (2412.06464), xLSTM (2410.22391).
* **Placement**: candidate placement tokens from the placer, in place of place plus offset (Action Space Shaping, 2004.00980).

## Not to do

* **Label smoothing** (1906.02629): Unplugged did better sharper.
* **An LSTM or GRU as cloning memory**: Unplugged's LSTM scored 70% against its transformer's 85%.
* **Offline Q-learning on the script data**:
  * Unplugged's deep offline Q scored 0%;
  * CRR and BCQ-likes did at best what cloning did;
  * only offline actor-critic with a fixed behaviour critic beat cloning.
* **Reusing PPO samples 4-8 times**: OpenAI Five's learning slowed or stopped.
* **Self-play or a league before cloning-to-PPO works**:
  * Gym-µRTS self-play underperformed;
  * TStarBot-X (2011.13729) found a scaled-down AlphaStar league fails without extra tricks.
* **Rewarding structures directly**: Gym-µRTS agents built more barracks and won no more.
* **Expecting memory to fix credit assignment** (2307.03864).
* **Caching encoder latents on disk**: about 38 GB, for one batched call a step.

## Already right, by the literature

* the frozen Perceiver encoder;
* actions in turn with pointers (AlphaStar, SCC, Multi-Agent Transformer 2205.14953);
* the set-marginal anchor loss (1911.09996, 1511.06391);
* canonical action masking (2006.14171: Gym-µRTS went from 0% to 82% wins with it);
* the separate privileged critic;
* a zero-sum potential reward annealed to win or loss;
* a decision every 3 s.

## References

* **RTS and MOBA**: 2308.03526 AlphaStar Unplugged; 2012.13169 SCC; 2011.13729 TStarBot-X; 1809.07193 TStarBots;
  2209.11553 HierNet-SC2; 1912.06680 OpenAI Five; 1912.09729, 2011.12692 Honor of Kings RL; 2011.12582 JueWu-SL; 1812.07887
  Hierarchical Macro Strategy; 2110.14221 MGG.
* **MicroRTS, Lux, Generals.io**: 2105.13807 Gym-µRTS; 2006.14171 invalid action masking; 2402.08112 RAISocketAI; 2610.11663
  command-conditioned PPO; 2010.03956 Action Guidance; 1707.03743 macromanagement from replays; 2304.13004 Lux centralised
  control; 2606.23348, 2507.06825 Generals.io.
* **Architecture**: 2103.03206, 2107.14795, 2202.07765 Perceiver family; 1810.00825 Set Transformer; 1703.06114 Deep Sets;
  1806.01830 Relational Deep RL; 1506.03134 Pointer Networks; 1511.06391 Order Matters; 1911.09996 orderless multi-label;
  2005.12872 DETR; 2205.14953 Multi-Agent Transformer.
* **Memory**: 1901.02860 Transformer-XL; 1910.06764 GTrXL; 2207.06881 RMT; 2203.07852 Block-Recurrent; 1911.05507 Compressive;
  2203.08913 Memorizing; 2404.07143 Infini-attention; 2306.09459 RATE; 2310.09971 AMAGO; 2307.03864; 2309.17207 Memory Gym;
  2303.01859 POPGym; 2110.05038; 2505.15040 RLBenchNet; 2412.06464 Gated DeltaNet; 2405.04517 xLSTM; 2410.22391 LRAM.
* **Systems and optimisation**: 2412.05496 FlexAttention; 2307.08691 FlashAttention-2; 2107.02027 packing; 2006.11751 Sample
  Factory; 2206.10558 EnvPool; 2502.16982 Muon; 2203.03466 µP.
* **Scaling**: 2307.09423; 2411.04434; 2210.00849; 2001.08361; 2203.15556; 2305.16264; 2410.09754 SimBa; 2405.16158 BRO;
  2305.19452 BBF; 2402.08609 MoE for RL.
* **Imitation and RL fine-tuning**: 1011.0686 DAgger; 1905.11979 causal confusion; 2010.14876 copycat; 2104.04258 CS:GO BC;
  2206.11795 VPT; 1803.03835 Kickstarting; 1511.06295 Policy Distillation; 2402.02868 forgetting; 2112.07544, 2210.05492 piKL;
  2204.02372 JSRL; 2311.02198 IBRL; 2302.02948 RLPD; 2412.07762 WSRL; 1709.10087 DAPG; 1812.06298 Residual Policy Learning;
  1707.05300 reverse curriculum; 2004.00980 action space shaping.
* **PPO and value**: 1802.01561 IMPALA; 2009.04416 PPG; 2006.05990; 2110.00641; 2403.03950 HL-Gauss; 1809.04474 PopArt.
* **Other**: 1611.05397 UNREAL; 2007.05929 SPR; 2301.04104 DreamerV3; 1906.02629 label smoothing; 2108.12409 ALiBi; 2402.04494;
  2311.17190 Minimax Exploiter.
* **Web**: the Pluto README (github.com/tscmoo/pluto), ROA-Star at NeurIPS 2023, and ar5iv full texts of SCC, AlphaStar Unplugged,
  Gym-µRTS, RAISocketAI and OpenAI Five.
