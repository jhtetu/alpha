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

*(filled in when the runs finish)*
