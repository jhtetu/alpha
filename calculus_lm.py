"""
calculus_lm.py — the "Calculus Language Model" (CLM).

PLAIN-ENGLISH SUMMARY
---------------------
A normal Transformer LLM is a tall stack of separate layers. Each layer
looks at every earlier word ("attention") and adjusts its guess.

This model replaces that with a differential equation. Instead of asking
"what is the next layer's output?", it asks "how fast, and in which
direction, should the model's understanding be CHANGING right now?"
A numerical solver (like the ones used to simulate planets or weather)
then follows that rate of change from "depth 0" to "depth 1".

The rate of change at every word is built from three calculus ideas,
much like a PID controller in engineering:

    P  - the value itself      : what the text means right now
    I  - an integral           : a fading memory of everything said before
                                 (computed at many speeds: fast, medium, slow)
    D  - derivatives           : how fast meaning is changing (1st derivative)
                                 and whether that change is speeding up or
                                 slowing down (2nd derivative)

Written as math, the model is a partial differential equation:

    dz/ds (t) = F( z(t),  ∫ e^{-λ(t-τ)} z(τ) dτ,  dz/dt,  d²z/dt²,  s )

where  t = position in the text (time),  s = depth ("how long it has thought").

Honest note: computers still execute this with arrays of numbers (that
part is unavoidable on today's hardware). What changes is the MODEL — it is
defined by rates of change and integrals instead of a fixed stack of layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. The INTEGRAL: a multi-timescale fading memory of the past
# ---------------------------------------------------------------------------
class LeakyIntegral(nn.Module):
    """
    Solves, for every channel c, the ordinary differential equation

        dm_c/dt = -λ_c * m_c(t) + λ_c * u_c(t)

    whose exact solution is a weighted integral of the past:

        m_c(t) = ∫_{-∞}^{t} λ_c e^{-λ_c (t-τ)} u_c(τ) dτ

    Small λ = long, slow memory (remembers the whole paragraph).
    Large λ = short, fast memory (remembers the last few letters).
    The λ values are LEARNED, so the model decides its own memory speeds.

    Because the equation is linear we can solve it EXACTLY for all
    positions at once (no step-by-step loop) using a convolution.
    """

    def __init__(self, dim, min_timescale=1.0, max_timescale=256.0):
        super().__init__()
        # Spread the initial memory lengths log-uniformly between 1 and 256 steps.
        timescales = torch.exp(torch.linspace(math.log(min_timescale),
                                              math.log(max_timescale), dim))
        self.log_lambda = nn.Parameter(-torch.log(timescales))  # λ = 1/timescale

    def forward(self, u, dt=1.0):
        # u: (batch, time, channels)
        B, T, C = u.shape
        lam = torch.exp(self.log_lambda)                      # (C,) always > 0
        decay = torch.exp(-lam * dt)                          # e^{-λ Δt}
        steps = torch.arange(T, device=u.device, dtype=u.dtype)
        # Kernel k[j] = (1 - e^{-λΔt}) * e^{-λ Δt j}: exact discretisation of the ODE.
        kernel = (1 - decay)[None, :] * decay[None, :] ** steps[:, None]  # (T, C)
        # Causal convolution via FFT (only looks at the past, never the future).
        n = 2 * T
        U = torch.fft.rfft(u, n=n, dim=1)
        K = torch.fft.rfft(kernel, n=n, dim=0)
        m = torch.fft.irfft(U * K[None], n=n, dim=1)[:, :T]
        return m


# ---------------------------------------------------------------------------
# 2. The DERIVATIVES: how fast meaning changes along the text
# ---------------------------------------------------------------------------
def time_derivatives(z, dt=1.0):
    """
    Backward finite differences (they only use the past, so no cheating):
        velocity     dz/dt   ≈ (z_t - z_{t-1}) / Δt
        acceleration d²z/dt² ≈ (z_t - 2 z_{t-1} + z_{t-2}) / Δt²
    """
    z1 = F.pad(z, (0, 0, 1, 0))[:, :-1]          # z shifted back by one step
    z1[:, 0] = z[:, 0]                           # derivative is 0 at the very start
    z2 = F.pad(z1, (0, 0, 1, 0))[:, :-1]
    z2[:, 0] = z1[:, 0]
    velocity = (z - z1) / dt
    acceleration = (z - 2 * z1 + z2) / (dt * dt)
    return velocity, acceleration


# ---------------------------------------------------------------------------
# 3. The VECTOR FIELD: "how should understanding change right now?"
# ---------------------------------------------------------------------------
class DepthEmbedding(nn.Module):
    """Tells the vector field how far along in 'depth' s ∈ [0,1] we are."""

    def __init__(self, dim):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(16) * 3.0, requires_grad=False)
        self.to_scale_shift = nn.Linear(32, 2 * dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, s):
        x = s * self.freqs
        feats = torch.cat([torch.sin(x), torch.cos(x)])
        scale, shift = self.to_scale_shift(feats).chunk(2)
        return scale, shift


class CalculusField(nn.Module):
    """
    F(z, s): returns dz/ds — the rate at which the model's internal
    representation z should change as it 'thinks deeper'.
    """

    def __init__(self, dim, hidden_mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.integral = LeakyIntegral(dim)
        self.depth = DepthEmbedding(4 * dim)
        self.mix = nn.Linear(4 * dim, hidden_mult * dim)       # combine P, I, D, D²
        self.out = nn.Linear(hidden_mult * dim, dim)
        self.gate = nn.Linear(4 * dim, 4 * dim, bias=False)    # decides which signals matter
        nn.init.zeros_(self.out.weight)                        # start as "no change"
        nn.init.zeros_(self.out.bias)
        self.nfe = 0                                           # function-evaluation counter

    def forward(self, s, z, dt=1.0):
        self.nfe += 1
        zn = self.norm(z)
        P = zn                                   # proportional: the present
        I = self.integral(zn, dt)                # integral: fading memory of the past
        D1, D2 = time_derivatives(zn, dt)        # derivatives: velocity & acceleration
        h = torch.cat([P, I, D1, D2], dim=-1)
        scale, shift = self.depth(s)
        h = h * (1 + scale) + shift              # depth-aware
        h = h * torch.sigmoid(self.gate(h))    # gate: choose which signals matter
        return self.out(F.gelu(self.mix(h)))


# ---------------------------------------------------------------------------
# 4. The ODE SOLVER: follows the rate of change from depth s=0 to s=1
# ---------------------------------------------------------------------------
def odeint(field, z0, steps=4, method="rk4", dt=1.0):
    """
    Fixed-step numerical integration of dz/ds = field(s, z) from s=0 to s=1.
    More steps = more careful thinking (more compute), and you can choose a
    different number of steps AFTER training — something a layer stack can't do.
    """
    h = 1.0 / steps
    z = z0
    for i in range(steps):
        s = torch.tensor(i * h, device=z.device, dtype=z.dtype)
        if method == "euler":
            z = z + h * field(s, z, dt)
        elif method == "midpoint":
            k1 = field(s, z, dt)
            z = z + h * field(s + h / 2, z + h / 2 * k1, dt)
        elif method == "rk4":
            k1 = field(s, z, dt)
            k2 = field(s + h / 2, z + h / 2 * k1, dt)
            k3 = field(s + h / 2, z + h / 2 * k2, dt)
            k4 = field(s + h, z + h * k3, dt)
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(method)
    return z


# ---------------------------------------------------------------------------
# 5. The full language model
# ---------------------------------------------------------------------------
class CalculusLM(nn.Module):
    def __init__(self, vocab_size, dim=128, steps=4, method="rk4"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.field = CalculusField(dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.steps, self.method = steps, method

    def forward(self, idx, steps=None, method=None, dt=1.0):
        z0 = self.embed(idx)                                  # initial condition z(s=0)
        z1 = odeint(self.field, z0, steps or self.steps,
                    method or self.method, dt)                # solve the ODE to s=1
        return self.head(self.norm(z1))                       # next-character scores


# ---------------------------------------------------------------------------
# Baseline for an honest comparison: a small standard Transformer
# ---------------------------------------------------------------------------
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, dim=128, layers=2, heads=4, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, **_):
        T = idx.shape[1]
        x = self.embed(idx) + self.pos(torch.arange(T, device=idx.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        return self.head(self.norm(x))
