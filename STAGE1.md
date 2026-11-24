# Stage 1: Does the law-core design beat the alternatives for the same compute?

**Goal of the project:** an AI that uses less power and less compute, and is more
accurate. **Goal of Stage 1:** produce trustworthy small-scale evidence, yes or no,
that the Calculus LM design gives **better accuracy per unit of compute** than
today's two leading designs. If it does, Stage 2 is to check that the advantage
grows with model size.

## The contenders

All three read raw text one byte at a time, have about 0.9 million parameters,
and share the same input, output and training recipe. Only the core differs.

| Model | Core idea | Cost as text gets longer |
|---|---|---|
| **Transformer** (modern version with rotary positions) | every letter compares itself with every earlier letter (attention) | grows with length² |
| **State-space model (SSM)** (S4D/H3-style) | a running memory governed by a linear differential equation | grows with length |
| **Calculus LM v2** (this project) | law core, see below | grows with length, plus one small attention read |

### Calculus LM v2: what's in the law core

1. **Thinking is a differential equation.** Understanding `z` flows through depth `s`:
   `dz/ds = F_s(z)`, solved in 4 Euler steps.
2. **The rate of change uses calculus signals along the text:**
   present (P), multi-timescale integral memory (I), velocity (dz/dt) and
   acceleration (d²z/dt²).
3. **Exact-recall helper.** One attention read at step 3 fixes the fading-memory
   weakness found in the first experiments (copying exact earlier text).
4. **Settle-to-stop.** A shared read-out after every step. At prediction time a
   letter stops thinking once its answer has settled (confidence above a
   threshold), so easy letters cost less. This is the "spend energy only where
   change is happening" rule.
5. **Library memory.** Facts can be stored outside the model: a library of
   (situation → what came next) pairs taken from the training text and looked up
   by similarity (kNN-LM, Khandelwal et al. 2020). Updating the library needs no
   retraining. Tested on all three models, for fairness.

## Test method (fairness rules)

* **Benchmark:** enwik8, the standard 100 MB Wikipedia test for text models
  (checksum-verified). Standard split: 90M bytes train, 5M validation.
* **Score:** bits per character (bpc) on held-out text, the official enwik8
  measure. Lower is better.
* **Same training time** for every model (10 minutes each, same machine).
* **Learning rate** chosen per model from a short sweep (1e-3 vs 3e-3), so no
  model is handicapped by a bad setting.
* **3 random seeds** per model, reported as mean and spread.
* **Compute** counted in FLOPs (multiplications and additions) per letter, not
  guessed. Counted by PyTorch, plus FFT work added by hand.
* **Ablations:** Calculus LM v2 without the recall helper, and without the
  derivative signals, to see whether each part earns its place.

Run it yourself: `cd v2 && python bench.py lrsweep && python bench.py main && python bench.py library`
(about 2.5 hours on 4 CPU cores). Raw numbers are in `v2/results/`.

## Results

All numbers are bits per character on held-out enwik8 text (**lower is better**),
after 10 minutes of training each. "Compute" is the work needed to predict one
letter (millions of FLOPs, at a 512-letter context).

### Main comparison (3 seeds each)

| Model | Seed 0 | Seed 1 | Seed 2 | **Mean** | Compute per letter | Parameters |
|---|---|---|---|---|---|---|
| Transformer | 2.033 | 2.050 | 2.041 | **2.041** | 2.82 M | 918,656 |
| State-space model | 2.087 | 2.101 | 2.091 | **2.093** | **1.72 M** | 854,144 |
| **Calculus LM v2** | 2.041 | 2.048 | 2.028 | **2.039** | 2.34 M | 935,040 |

* **Calculus LM v2 vs Transformer:** the same accuracy (the 0.002 gap is far
  smaller than the 0.02 seed-to-seed spread) with **17% less compute per letter**.
* **Calculus LM v2 vs SSM:** Calculus is clearly more accurate (−0.054 bpc, about
  5 times the seed spread) but uses 36% more compute. Neither dominates the other.
* No model was both the most accurate and the cheapest.

### Ablations (seed 0): does each part earn its place?

| Variant | Score | Change vs full model | Compute per letter |
|---|---|---|---|
| Full Calculus LM v2 | 2.041 | | 2.34 M |
| Without velocity and acceleration signals | 2.117 | **+0.077 (much worse)** | 2.34 M |
| Without the exact-recall helper | 2.054 | +0.013 (slightly worse) | **1.95 M** |

* **The derivative signals (the "calculus" part) clearly matter.** Removing them
  costs more than any other change in the study, at the same model size and compute.
* The recall helper helps only a little at this text length. Without it, the
  model is still more accurate than the SSM (2.054 vs 2.093) at 13% more compute
  (one seed only).

### Settle-to-stop (letters stop thinking once their answer is settled)

Mean over the 3 seeds:

| Confidence threshold | Score | Compute per letter | Compute saved |
|---|---|---|---|
| off | 2.039 | 2.34 M | |
| 0.99 | 2.040 | 2.32 M | 1% |
| 0.95 | 2.053 | 2.27 M | 3% |
| 0.90 | 2.070 | 2.23 M | 5% |
| 0.80 | 2.111 | 2.16 M | 8% |
| 0.50 | 2.377 | 1.90 M | 19% |

**Not worth it yet.** The savings are small and accuracy drops quickly. These
small models are rarely very confident, so few letters settle early. Only the
step's MLP is counted as skipped; the memory and recall parts are counted as
always on, so the savings shown are a lower bound.

### Library memory (kNN lookup of about 327,000 stored training situations)

Measured on a held-out half of 8 validation batches (seed 0 models), so the
"without" numbers differ from the table above.

| Model | Without library | With library | Gain |
|---|---|---|---|
| Transformer | 2.098 | 2.076 | −0.022 |
| SSM | 2.146 | 2.125 | −0.021 |
| Calculus LM v2 | 2.100 | 2.079 | −0.021 |

The library helps every model by the same amount. It is a useful add-on, but it
gives the calculus design no special advantage.

### Verdict

**A partial yes.** Calculus LM v2 reached Transformer accuracy with 17% less compute
per letter, across 3 seeds. It did **not** beat the Transformer on accuracy, and
it did not beat the SSM on cost. The derivative signals are the best-supported new
idea. Settle-to-stop and library memory did not give the calculus design an edge
at this scale.

### Honest caveats

* **Tiny scale:** 0.9M parameters and 10 minutes on a 4-core CPU. Results can change at larger sizes.
* **Learning rate:** every model preferred the higher of the two rates tried
  (3e-3), so a still-higher rate might help some models more than others.
* **Short context only (512 letters).** The earlier long-passage test, where the
  calculus design looked strongest, was not repeated here.
* **Equal time, not equal training FLOPs.** Speed on this CPU decides how many steps each model got.
* **Not a top-of-the-line SSM:** the SSM baseline is S4D/H3-style, not the selective (Mamba) kind.
* **FLOPs are a stand-in for power.** Real energy depends on the hardware.
* The ablations and library test used one seed.
