# Design: a Calculus Language Model (CLM)

## 1. The idea in one paragraph

A Transformer treats text as a list of separate snapshots. It runs every
snapshot through a fixed stack of layers (layer 1, layer 2, … layer 96) and
compares every word with every other word ("attention"). The Calculus Language
Model treats text as a **signal flowing through time**. The model does not ask
"what does layer 7 output?". It asks **"at this moment, how fast and in which
direction should my understanding be changing?"** A numerical solver, the same
kind used to simulate orbits or weather, then follows that rate of change until
the model has "thought" enough to predict what comes next.

## 2. Two kinds of "time"

The model has two continuous axes:

| Axis | Symbol | Meaning | Transformer equivalent |
|------|--------|---------|------------------------|
| Text time | `t` | position in the text (can be real clock time) | token position |
| Thinking depth | `s ∈ [0, 1]` | how far the model has refined its understanding | layer number |

The model's internal state is a function `z(t, s)`: a vector describing what the
text means at moment `t`, after thinking for depth `s`.

## 3. The governing equation

The whole model is a single differential equation:

```
∂z/∂s (t)  =  F( P, I, D₁, D₂, s )

  P  = z(t)                                   the present ("proportional")
  I  = ∫₋∞ᵗ λ e^{-λ(t-τ)} z(τ) dτ             fading memory of the past ("integral")
  D₁ = ∂z/∂t                                  how fast meaning is changing ("derivative")
  D₂ = ∂²z/∂t²                                whether that change is speeding up or slowing down
```

Start: `z(t, 0)` = the embedding of the character or word at time `t`.
End: solve the equation from `s = 0` to `s = 1`, then read the next-token
probabilities off `z(t, 1)`.

The P/I/D naming comes from the **PID controller**, the most widely used
feedback-control design in engineering, found in thermostats, cruise control and
drones. A PID controller steers a system using its present value, its
accumulated past and its current trend. The CLM applies the same idea to
meaning.

### 3.1 The integral term: memory without attention

Instead of looking back at every earlier word (attention, whose cost grows with
the square of the text length), the model keeps a **bank of leaky integrators**.
Each one solves

```
dm/dt = -λ · m(t) + λ · u(t)
```

Every channel has its own learned rate `λ`:

* large λ: a fast memory (the last few letters)
* small λ: a slow memory (the whole paragraph)

This equation is linear, so it has an exact closed-form solution: a convolution
with an exponential kernel. The prototype computes it for all positions at once
with an FFT. Cost grows as `T log T` instead of `T²`.

### 3.2 The derivative terms: sensing change

`D₁` and `D₂` are backward finite differences. They look only at the past, so
the model cannot cheat by peeking at the future. They let the model notice
things like "the topic is shifting", "a sentence is accelerating toward its
end" or "this word broke the pattern".

### 3.3 The vector field `F`

`F` is a small neural network shared across all depths. It:
1. concatenates `[P, I, D₁, D₂]`,
2. is told the current depth `s` (sinusoidal features → scale and shift),
3. uses a learned **gate** to decide which of the four signals matter at this
   moment,
4. outputs the rate of change `∂z/∂s`.

The output layer starts at zero, so at initialisation the model "does not
change its mind". Training teaches it how to flow.

### 3.4 The solver

The prototype uses fixed-step Runge–Kutta 4 (4 steps, so 16 evaluations of `F`).
Because the model is an ODE and not a stack of layers, **you can change the
solver after training**:

* fewer steps: faster and cheaper, slightly less accurate
* more steps: more careful "thinking"

A layer stack cannot do this. It has exactly as many layers as it was built
with.

## 4. What is actually new, and what isn't (honest framing)

* **Today's LLMs already use calculus.** Training any neural network depends on
  derivatives (backpropagation = the chain rule). The CLM's difference is that
  the *model itself* is defined by rates of change and integrals, not only its
  training.
* **Computers still run it with arrays of numbers.** Every differential
  equation solved on a computer turns into arithmetic on vectors. Linear algebra
  is the engine. Calculus is the blueprint.
* **This builds on real research:**
  Neural ODEs (Chen et al., 2018), Neural Controlled Differential Equations
  (Kidger et al., 2020), Liquid Time-Constant Networks (Hasani et al., 2021), and
  state-space models such as S4 and Mamba (Gu et al., 2021–2023), which are
  discretised linear ODEs and already compete with Transformers. The CLM combines
  these into one design: *continuous depth + continuous-time integral memory +
  explicit derivative sensing, organised as a PID-style field.*

## 5. Where this design should beat a Transformer, in principle

1. **Irregular, real-world time.** Sensor readings, medical records, markets
   and conversations don't arrive at evenly spaced steps. The CLM accepts a real
   `Δt` between tokens (the `dt` argument in the code). A Transformer has no
   built-in notion of elapsed time.
2. **Long inputs.** Memory cost grows as `T log T` (or `T` when run
   step-by-step), not `T²`.
3. **Adjustable compute.** It can think harder on hard inputs (more solver
   steps) and less on easy ones, without retraining.
4. **Streaming.** The integral memory can be updated one token at a time with
   a fixed-size state, like a running total.

## 6. Known weaknesses and risks

* **Exact recall is harder.** Attention can copy a specific earlier word
  exactly. A fading integral blurs the past. A hybrid (a few attention layers
  plus the calculus field) is a likely practical answer.
* **Solver cost.** RK4 with 4 steps evaluates the field 16 times, so it is slower
  per token than a 3-layer Transformer of similar size.
* **Scale is untested.** The prototype is tiny (about 640k parameters,
  character-level). Nobody knows yet whether this exact design scales to
  billions of parameters.

## 7. Roadmap

1. Adaptive solvers (Dormand–Prince) that pick their own step count per input.
2. Real timestamps: train on time-stamped data (chat logs, sensor streams).
3. Hybrid model: calculus field plus 1–2 attention layers for exact recall.
4. Adjoint-method training (constant memory regardless of solver steps).
5. Scale up: word-piece tokens, larger `dim`, bigger datasets, GPU training.
