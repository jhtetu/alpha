# Calculus Language Model (CLM)

An experimental language model built on **rates of change** (calculus) instead
of a fixed stack of layers (the Transformer design used by today's LLMs).

* **[DESIGN.md](DESIGN.md)**: the full design, the math, honest caveats, and a roadmap
* **[calculus_lm.py](calculus_lm.py)**: the model (heavily commented)
* **[train.py](train.py)**: trains it on Shakespeare and compares it to a Transformer

## How it works, in plain words

A Transformer is like an **assembly line**: text passes through a fixed number of
stations (layers), and at each one every word re-reads every earlier word.

The Calculus LM is like a **ball rolling downhill toward the right answer**. At
every moment it asks: *"Which way should my understanding move, and how fast?"*
It takes a small step, asks again, and repeats. To decide the direction, it
combines three signals, like a PID controller (the device behind cruise control
and thermostats):

| Signal | Meaning |
|---|---|
| **P – Present** | what the text says right now |
| **I – Integral** | a fading memory of the past, at many speeds (fast, medium, slow) |
| **D – Derivative** | how fast the meaning is changing, and whether that change is speeding up |

## First experiment (small, honest test)

Both models: about 650k parameters, same data (Tiny Shakespeare, letter by
letter), same 1500 training steps, run on 4 CPU cores.
Score = *bits per character* on text the models never saw (**lower is better**).

| | Calculus LM | Transformer (3 layers) |
|---|---|---|
| Parameters | 641,601 | 677,313 |
| **Error on unseen text** | **2.218** | 2.451 |
| Training time | 1213 s (~20 min) | 125 s (~2 min) |

**What this shows:**
* ✅ With the same number of training steps, the Calculus LM learned **more** (about 10% lower error).
* ❌ But it was about **10× slower** per step. With equal *time*, the Transformer
  could train 10× longer, and this test does not show who would win then.
* ✅ **Adjustable thinking effort after training** (something Transformers can't do):

| Solver setting | Compute | Error |
|---|---|---|
| Euler, 2 steps | 2 | 2.826 |
| Euler, 4 steps | 4 | 2.365 |
| RK4, 2 steps | 8 | 2.244 |
| RK4, 4 steps (trained) | 16 | 2.218 |
| RK4, 8 steps | 32 | 2.219 |

Half the compute costs only about 1% accuracy.

### Sample writing (prompt: `ROMEO:`)

**Calculus LM**
```
ROMEO:
And I am found me to the lord, you been my name
More and success with been to meet me:
The subore the words mildren awake our revermite
Thou art thou shamely on the thief and singuled
```

**Transformer**
```
ROMEO:
Was go thou are for mill
The was name whom such and grounders bond,
Some to the will not bleed the much of more.
```

Both are tiny models trained for minutes, so neither makes much sense yet. That is expected.

## Follow-up experiments: equal training time

The first test gave both models the same number of *steps*, which favoured the
slower Calculus LM. These tests give each model the same **12 minutes** of
training instead (`experiments/` holds the raw numbers).

### Test 1: short passages (128 letters)

| | Calculus LM | Transformer |
|---|---|---|
| Training steps in 12 min | 720 | 6,791 |
| **Error on unseen text** | 2.312 | **2.212** |
| Prediction time (batch of 32) | 314 ms | **34 ms** |

**The Transformer wins.** It fits about 9× more steps into the same time, which
outweighs the Calculus LM learning more per step.

### Test 2: long passages (1,024 letters)

| | Calculus LM | Transformer |
|---|---|---|
| Training steps in 12 min | 670 | 3,530 |
| **Error on unseen text** | **2.344** | 2.738 |
| Prediction time (full effort, batch of 4) | 264 ms | **186 ms** |
| Error at half effort | **2.363** | (no such setting) |

**The Calculus LM wins clearly.** Its accuracy barely changed going from short to
long passages (2.31 → 2.34). The Transformer got much worse (2.21 → 2.74): its
cost grows with the square of the length, so it managed fewer steps, and it
must learn a separate position marker for each of the 1,024 places.
In a separate speed test at 1,024 letters, the Calculus LM at half effort was
already faster than the Transformer (47 ms vs 74 ms), and at 4,096 letters it was
about 4.5× faster (200 ms vs 892 ms).

### Caveats

* Each test was run **once**, with one random seed, on a small model.
* The Transformer used the same settings as in the short test. A Transformer
  tuned for long passages (for example a different learning rate or position
  method) would likely do better than 2.738.
* Results on tiny models do not reliably predict results at billions of parameters.

**Takeaway:** for short text, today's Transformer design is faster and better.
For long text, this first evidence suggests the calculus design is more accurate
and, at longer lengths, faster.

## Run it yourself

```bash
pip install torch
python train.py              # trains both models (about 25 minutes on a laptop CPU)
python train.py --minutes 12 --block 1024 --batch 4   # long-passage, equal-time test
```
