"""
Stage 1 models: Calculus LM v2 and two fair baselines.

All three read raw bytes (vocab 256) and share the same width, embedding,
output head and optimiser, so differences come from the core design.

  TransformerLM   - modern small Transformer (pre-norm, rotary positions)
  SSMLM           - state-space baseline (S4D/H3-style: multi-timescale
                    linear ODE memory + gating). NOT selective like Mamba.
  CalculusLMv2    - the "law core" design:
                      * depth is an ODE   dz/ds = F_s(z)   solved with Euler steps
                      * F sees Present, Integral (multi-timescale memory),
                        Velocity and Acceleration along the text
                      * optional exact-recall helper (one attention read)
                      * a shared read-out at every step, so each letter can
                        stop thinking once it has settled ("settle-to-stop")
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

USE_SDPA = True  # fast attention kernel; set False when counting FLOPs


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


# ----------------------------------------------------------------------------- calculus parts
class LeakyIntegral(nn.Module):
    """Exact solution of dm/dt = -λ(m - u) for every channel, via causal FFT convolution.
    Cost grows as T·log T (not T²). λ is learned; initial memory lengths 1..max_len letters."""

    def __init__(self, d, max_timescale=1024.0):
        super().__init__()
        ts = torch.exp(torch.linspace(0, math.log(max_timescale), d))
        self.log_lam = nn.Parameter(-torch.log(ts))

    def forward(self, u):
        B, T, C = u.shape
        lam = self.log_lam.exp()
        a = torch.exp(-lam)
        k = (1 - a)[None] * a[None] ** torch.arange(T, device=u.device, dtype=u.dtype)[:, None]
        n = 2 * T
        y = torch.fft.irfft(torch.fft.rfft(u, n=n, dim=1) * torch.fft.rfft(k, n=n, dim=0)[None], n=n, dim=1)
        return y[:, :T]


def derivatives(u):
    """Backward differences along the text: velocity and acceleration (past only)."""
    u1 = torch.cat([u[:, :1], u[:, :-1]], 1)
    u2 = torch.cat([u1[:, :1], u1[:, :-1]], 1)
    return u - u1, u - 2 * u1 + u2


# ----------------------------------------------------------------------------- attention
def rope(x):
    B, H, T, dh = x.shape
    f = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=x.device, dtype=x.dtype) / dh))
    ang = torch.arange(T, device=x.device, dtype=x.dtype)[:, None] * f[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class CausalAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        if USE_SDPA:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:  # explicit version so the FLOP counter sees every multiplication
            s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
            y = s.softmax(-1) @ v
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden):
        super().__init__()
        self.w1 = nn.Linear(d_in, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_out, bias=False)

    def forward(self, x):
        a, b = self.w1(x).chunk(2, -1)
        return self.w2(F.gelu(a) * b)


# ----------------------------------------------------------------------------- shared shell
class ByteLM(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.embed = nn.Embedding(256, d)
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, 256, bias=False)

    def readout(self, z):
        h = self.norm(z)
        return self.head(h), h


# ----------------------------------------------------------------------------- baselines
class TransformerLM(ByteLM):
    def __init__(self, d=128, layers=4, heads=4):
        super().__init__(d)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleList([RMSNorm(d), CausalAttention(d, heads), RMSNorm(d), MLP(d, d, 3 * d)]))

    def forward(self, x):
        z = self.embed(x)
        for n1, att, n2, mlp in self.blocks:
            z = z + att(n1(z))
            z = z + mlp(n2(z))
        logits, h = self.readout(z)
        return {"logits": logits, "hidden": h}


class SSMBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.inp = nn.Linear(d, 2 * d, bias=False)
        self.mem = LeakyIntegral(d)
        self.skip = nn.Parameter(torch.ones(d))
        self.out = nn.Linear(d, d, bias=False)
        self.mlp = MLP(d, d, 3 * d)

    def forward(self, z):
        x, g = self.inp(self.n1(z)).chunk(2, -1)
        y = self.mem(x) + self.skip * x
        z = z + self.out(y * F.silu(g))
        return z + self.mlp(self.n2(z))


class SSMLM(ByteLM):
    def __init__(self, d=128, layers=4):
        super().__init__(d)
        self.blocks = nn.ModuleList([SSMBlock(d) for _ in range(layers)])

    def forward(self, x):
        z = self.embed(x)
        for b in self.blocks:
            z = b(z)
        logits, h = self.readout(z)
        return {"logits": logits, "hidden": h}


# ----------------------------------------------------------------------------- Calculus LM v2
class CalculusField(nn.Module):
    """F_s(z): the rate of change of understanding during one interval of depth s."""

    def __init__(self, d, use_derivs=True, use_recall=False, heads=4):
        super().__init__()
        self.use_derivs = use_derivs
        self.norm = RMSNorm(d)
        self.inp = nn.Linear(d, d, bias=False)
        self.mem = LeakyIntegral(d)
        n_sig = 4 if use_derivs else 2
        # hidden width chosen so the model has about as many parameters as the baselines,
        # with or without the derivative signals (keeps the comparison fair)
        self.mlp = MLP(n_sig * d, d, int(1.25 * d * 9 / (2 * n_sig + 1)))
        self.recall = CausalAttention(d, heads) if use_recall else None

    def forward(self, z):
        zn = self.norm(z)
        u = self.inp(zn)
        sig = [u, self.mem(u)]                         # Present, Integral
        if self.use_derivs:
            sig += list(derivatives(u))                # Velocity, Acceleration
        dz = self.mlp(torch.cat(sig, -1))
        if self.recall is not None:
            dz = dz + self.recall(zn)                  # exact-recall helper
        return dz


class CalculusLMv2(ByteLM):
    def __init__(self, d=128, steps=4, use_derivs=True, recall_step=2, deep_sup=0.1):
        super().__init__(d)
        self.steps, self.deep_sup = steps, deep_sup
        self.fields = nn.ModuleList([
            CalculusField(d, use_derivs, use_recall=(i == recall_step)) for i in range(steps)])

    def forward(self, x, exit_threshold=None):
        """exit_threshold=None: think for all steps.
        exit_threshold=p: a letter stops thinking (its z freezes) as soon as the read-out is
        at least p confident: settle-to-stop. Returns the per-step fraction of letters still thinking."""
        h = 1.0 / self.steps
        z = self.embed(x)
        inter_logits = []
        active = torch.ones(x.shape, dtype=torch.bool, device=x.device)
        final_logits = None
        active_frac = []
        for i, field in enumerate(self.fields):
            active_frac.append(active.float().mean().item())
            dz = field(z)
            z = z + h * dz * active.unsqueeze(-1)       # settled letters stay frozen
            logits, hid = self.readout(z)
            if exit_threshold is None:
                if i < self.steps - 1:
                    inter_logits.append(logits)
            else:
                if final_logits is None:
                    final_logits = logits.clone()
                final_logits[active] = logits[active]
                conf = logits.softmax(-1).max(-1).values
                active = active & (conf < exit_threshold)
        if exit_threshold is not None:
            return {"logits": final_logits, "hidden": hid, "active_frac": active_frac}
        return {"logits": logits, "hidden": hid, "inter_logits": inter_logits}
