"""
Stage 1 benchmark: Calculus LM v2 vs. a Transformer vs. a state-space model on enwik8.

Fairness rules
  * same data (enwik8 bytes), same context (512), same batch, same width (128)
  * same wall-clock training budget for every model, on the same machine
  * each model gets its best learning rate from a short sweep
  * several random seeds per model; we report mean ± spread
  * compute is counted (FLOPs per letter), not guessed

Usage
  python bench.py smoke                  # quick check that everything runs
  python bench.py lrsweep                # short runs to pick each model's learning rate
  python bench.py main                   # full comparison (resumable)
  python bench.py library                # library-memory (kNN) test on trained models
"""

import json, math, os, sys, time
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
import models
from models import TransformerLM, SSMLM, CalculusLMv2, LeakyIntegral

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
CTX, BATCH, D = 512, 16, 128

# ---------------------------------------------------------------- data (standard enwik8 split 90M/5M/5M)
raw = open(os.path.join(HERE, "..", "data", "enwik8"), "rb").read()
data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
train_data, val_data = data[:90_000_000], data[90_000_000:95_000_000]

MODELS = {
    "transformer": lambda: TransformerLM(D, layers=4),
    "ssm": lambda: SSMLM(D, layers=4),
    "calculus_v2": lambda: CalculusLMv2(D, steps=4),
    "calculus_v2_no_recall": lambda: CalculusLMv2(D, steps=4, recall_step=-1),
    "calculus_v2_no_derivs": lambda: CalculusLMv2(D, steps=4, use_derivs=False),
}


def get_batch(src, gen=None, b=BATCH):
    ix = torch.randint(len(src) - CTX - 1, (b,), generator=gen)
    x = torch.stack([src[i:i + CTX] for i in ix])
    y = torch.stack([src[i + 1:i + CTX + 1] for i in ix])
    return x, y


def val_batches(n=24):
    g = torch.Generator().manual_seed(1234)
    return [get_batch(val_data, g) for _ in range(n)]


# ---------------------------------------------------------------- compute counting
def flops_per_letter(model):
    """Forward FLOPs per letter at context 512. Matmuls/attention counted by PyTorch;
    FFTs (not counted by PyTorch) added by hand at 2.5·n·log2(n) per real FFT."""
    fft = [0.0]

    def hook(mod, inp, out):
        B, T, C = inp[0].shape
        n = 2 * T
        fft[0] += B * C * 2 * 2.5 * n * math.log2(n) + C * 2.5 * n * math.log2(n) + 6 * B * C * (n // 2 + 1)

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, LeakyIntegral)]
    models.USE_SDPA = False
    x = torch.zeros(1, CTX, dtype=torch.long)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    models.USE_SDPA = True
    for h in hs:
        h.remove()
    return (fc.get_total_flops() + fft[0]) / CTX


def field_mlp_flops_per_letter(model):
    """FLOPs per letter of the part of each calculus step a settled letter can skip (its MLP)."""
    m = model.fields[0].mlp
    return 2 * (m.w1.in_features * m.w1.out_features + m.w2.in_features * m.w2.out_features)


# ---------------------------------------------------------------- train / evaluate
def loss_fn(out, y):
    loss = F.cross_entropy(out["logits"].reshape(-1, 256), y.reshape(-1))
    inter = out.get("inter_logits") or []
    if inter:
        aux = sum(F.cross_entropy(l.reshape(-1, 256), y.reshape(-1)) for l in inter) / len(inter)
        loss = loss + 0.1 * aux
    return loss


@torch.no_grad()
def evaluate(model, batches, **kw):
    model.eval()
    tot, n, fr = 0.0, 0, []
    for x, y in batches:
        out = model(x, **kw)
        tot += F.cross_entropy(out["logits"].reshape(-1, 256), y.reshape(-1), reduction="sum").item()
        n += y.numel()
        if "active_frac" in out:
            fr.append(out["active_frac"])
    model.train()
    res = {"bpc": tot / n / math.log(2)}
    if fr:
        res["active_frac_per_step"] = [sum(c) / len(c) for c in zip(*fr)]
    return res


def train(name, lr, minutes, seed, log_every=60):
    torch.manual_seed(seed)
    model = MODELS[name]()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    budget, t0, it, last = minutes * 60, time.time(), 0, 0
    gen = torch.Generator().manual_seed(seed)
    while True:
        frac = (time.time() - t0) / budget
        if frac >= 1:
            break
        cur = lr * (frac / 0.05 if frac < 0.05 else 0.5 * (1 + math.cos(math.pi * (frac - 0.05) / 0.95)))
        for g in opt.param_groups:
            g["lr"] = cur
        x, y = get_batch(train_data, gen)
        loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        it += 1
        if time.time() - last > log_every:
            last = time.time()
            print(f"    {name} s{seed} step {it} loss {loss.item() / math.log(2):.3f} bpc ({time.time() - t0:.0f}s)", flush=True)
    return model, it


def run(name, lr, minutes, seed, tag):
    path = os.path.join(RES, f"{tag}_{name}_lr{lr}_s{seed}.json")
    if os.path.exists(path):
        print(f"  [skip] {os.path.basename(path)} exists", flush=True)
        return json.load(open(path))
    print(f"  >>> {tag}: {name} lr={lr} seed={seed} ({minutes} min)", flush=True)
    model, steps = train(name, lr, minutes, seed)
    vb = val_batches()
    res = {"model": name, "lr": lr, "seed": seed, "minutes": minutes, "steps": steps,
           "params": sum(p.numel() for p in model.parameters()),
           "flops_per_letter": flops_per_letter(model),
           "val": evaluate(model, vb)}
    if isinstance(model, CalculusLMv2) and tag == "main":
        full = res["flops_per_letter"]
        skip = field_mlp_flops_per_letter(model)
        res["settle_to_stop"] = []
        for th in [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]:
            ev = evaluate(model, vb, exit_threshold=th)
            saved = sum(1 - f for f in ev["active_frac_per_step"]) * skip
            res["settle_to_stop"].append({"threshold": th, "bpc": ev["bpc"],
                                          "active_frac_per_step": ev["active_frac_per_step"],
                                          "flops_per_letter": full - saved})
    if tag == "main":
        torch.save(model.state_dict(), os.path.join(RES, f"{name}_s{seed}.pt"))
    json.dump(res, open(path, "w"), indent=1)
    print(f"  <<< {name} s{seed}: {steps} steps, val {res['val']['bpc']:.4f} bpc, "
          f"{res['flops_per_letter'] / 1e6:.2f} MFLOPs/letter, {res['params']:,} params", flush=True)
    return res


# ---------------------------------------------------------------- library memory (kNN-LM)
@torch.no_grad()
def library_test(name, seed, lr, n_store_batches=40, n_query_batches=8, k=32):
    """Store (hidden state -> next byte) pairs from TRAINING text in an external library.
    At prediction time, look up the 32 most similar past situations and blend their
    next bytes with the model's own guess. Blend settings are tuned on half the
    queries and reported on the other half."""
    path = os.path.join(RES, f"library_{name}_s{seed}.json")
    if os.path.exists(path):
        return json.load(open(path))
    model = MODELS[name]()
    model.load_state_dict(torch.load(os.path.join(RES, f"{name}_s{seed}.pt")))
    model.eval()
    g = torch.Generator().manual_seed(99)
    keys, vals = [], []
    for _ in range(n_store_batches):
        x, y = get_batch(train_data, g)
        h = model(x)["hidden"]
        keys.append(h.reshape(-1, D).half()); vals.append(y.reshape(-1))
    keys, vals = torch.cat(keys).float(), torch.cat(vals)
    k2 = (keys ** 2).sum(-1)
    lp_model, lp_knn_d, lp_knn_v, targets = [], [], [], []
    for x, y in val_batches()[:n_query_batches]:
        out = model(x)
        q = out["hidden"].reshape(-1, D)
        lp_model.append(F.log_softmax(out["logits"].reshape(-1, 256), -1))
        for qc in q.split(4096):
            d = k2[None] - 2 * qc @ keys.T + (qc ** 2).sum(-1, keepdim=True)
            dist, idx = d.topk(k, largest=False)
            lp_knn_d.append(dist); lp_knn_v.append(vals[idx])
        targets.append(y.reshape(-1))
    lp_model, dist, nv, tgt = torch.cat(lp_model), torch.cat(lp_knn_d), torch.cat(lp_knn_v), torch.cat(targets)
    half = len(tgt) // 2
    scale = dist[:half].median().item()

    def bpc(sl, tau, lam):
        w = F.softmax(-dist[sl] / (tau * scale), -1)
        p_knn = torch.zeros(w.shape[0], 256).scatter_add_(1, nv[sl], w)
        p = (1 - lam) * lp_model[sl].exp() + lam * p_knn
        return -torch.log(p.gather(1, tgt[sl, None]).clamp_min(1e-12)).mean().item() / math.log(2)

    best = min(((bpc(slice(0, half), t, l), t, l) for t in [0.05, 0.1, 0.25, 0.5, 1.0]
                for l in [0.05, 0.1, 0.2, 0.3, 0.4]))
    res = {"model": name, "seed": seed, "library_entries": len(vals), "tau": best[1], "lambda": best[2],
           "bpc_without_library": bpc(slice(half, None), 1.0, 0.0),
           "bpc_with_library": bpc(slice(half, None), best[1], best[2])}
    json.dump(res, open(path, "w"), indent=1)
    print(f"  library {name} s{seed}: {res['bpc_without_library']:.4f} -> {res['bpc_with_library']:.4f} bpc", flush=True)
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "smoke":
        for name in MODELS:
            m = MODELS[name]()
            x, y = get_batch(train_data)
            t0 = time.time()
            for _ in range(3):
                loss = loss_fn(m(x), y); loss.backward()
            print(f"{name:24s} params {sum(p.numel() for p in m.parameters()):>9,}  "
                  f"{(time.time() - t0) / 3:.2f}s/step  {flops_per_letter(m) / 1e6:.2f} MFLOPs/letter", flush=True)
        m = CalculusLMv2(D); print("exit test", evaluate(m, val_batches(1), exit_threshold=0.5))
    elif mode == "lrsweep":
        for name in ["transformer", "ssm", "calculus_v2"]:
            for lr in [1e-3, 3e-3]:
                run(name, lr, 3, 0, "lrsweep")
    elif mode == "main":
        best = {}
        for name in ["transformer", "ssm", "calculus_v2"]:
            rs = [json.load(open(os.path.join(RES, f))) for f in os.listdir(RES)
                  if f.startswith(f"lrsweep_{name}_lr")]
            best[name] = min(rs, key=lambda r: r["val"]["bpc"])["lr"]
        best["calculus_v2_no_recall"] = best["calculus_v2_no_derivs"] = best["calculus_v2"]
        print("learning rates:", best, flush=True)
        minutes = float(os.environ.get("MINUTES", 10))
        for seed in [0, 1, 2]:
            for name in ["transformer", "ssm", "calculus_v2"]:
                run(name, best[name], minutes, seed, "main")
        for name in ["calculus_v2_no_recall", "calculus_v2_no_derivs"]:
            run(name, best[name], minutes, 0, "main")
    elif mode == "library":
        for name in ["transformer", "ssm", "calculus_v2"]:
            library_test(name, 0, None)
