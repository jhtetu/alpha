"""
train.py — train the Calculus LM (and, for comparison, a small Transformer)
on Shakespeare, one character at a time, then let it write some text.

Run:
    python train.py                 # trains both models and compares them
    python train.py --model calculus --iters 2000
    python train.py --minutes 12          # equal-time comparison (12 min each)
    python train.py --minutes 12 --block 1024 --batch 4   # long passages
"""

import argparse, json, math, time
import torch
import torch.nn.functional as F
from calculus_lm import CalculusLM, TransformerLM

p = argparse.ArgumentParser()
p.add_argument("--model", default="both", choices=["both", "calculus", "transformer"])
p.add_argument("--iters", type=int, default=1500)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--block", type=int, default=128, help="characters of context")
p.add_argument("--dim", type=int, default=128)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--minutes", type=float, default=0,
               help="if set, train for this many minutes instead of a fixed step count")
p.add_argument("--out", default="results.json")
args = p.parse_args()

torch.manual_seed(args.seed)
torch.set_num_threads(4)

# ----- data: every distinct character becomes a token --------------------------
text = open("data/tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]


def batch(src):
    ix = torch.randint(len(src) - args.block - 1, (args.batch,))
    x = torch.stack([src[i:i + args.block] for i in ix])
    y = torch.stack([src[i + 1:i + args.block + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, n=40, **kw):
    model.eval()
    g = torch.Generator().manual_seed(123)  # same validation batches for every model
    losses = []
    for _ in range(n):
        ix = torch.randint(len(val_data) - args.block - 1, (args.batch,), generator=g)
        x = torch.stack([val_data[i:i + args.block] for i in ix])
        y = torch.stack([val_data[i + 1:i + args.block + 1] for i in ix])
        logits = model(x, **kw)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses) / math.log(2)  # bits per character (lower = better)


@torch.no_grad()
def generate(model, prompt="ROMEO:\n", n=300, temperature=0.8):
    model.eval()
    idx = torch.tensor([[stoi[c] for c in prompt]])
    for _ in range(n):
        logits = model(idx[:, -args.block:])[:, -1] / temperature
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
    model.train()
    return "".join(chars[i] for i in idx[0].tolist())


def train(name, model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== {name}: {params:,} parameters ===", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    budget = args.minutes * 60

    def progress(it, elapsed):  # 0 -> 1 over the run (by steps or by time)
        return elapsed / budget if budget else it / args.iters

    t0, it, last_print = time.time(), 0, 0
    while progress(it, time.time() - t0) < 1:
        frac = progress(it, time.time() - t0)
        # learning rate: short warm-up, then smooth cosine decay to zero
        lr = args.lr * (frac / 0.1 if frac < 0.1 else 0.5 * (1 + math.cos(math.pi * (frac - 0.1) / 0.9)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch(train_data)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        it += 1
        if it == 1 or time.time() - last_print > 120:
            last_print = time.time()
            print(f"  step {it:5d}  train loss {loss.item() / math.log(2):.3f} bits/char"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    seconds = time.time() - t0
    # how long one prediction pass over a batch takes once training is done
    x, _ = batch(val_data)
    with torch.no_grad():
        model.eval(); model(x); t1 = time.time()
        for _ in range(3): model(x)
        model.train()
    result = {"params": params, "steps": it, "seconds": round(seconds),
              "predict_ms_per_batch": round((time.time() - t1) / 3 * 1000),
              "val_bits_per_char": round(evaluate(model), 3)}
    print(f"  {it} steps in {seconds:.0f}s, validation: {result['val_bits_per_char']} bits/char,"
          f" prediction {result['predict_ms_per_batch']} ms/batch", flush=True)
    return result


results = {}
if args.model in ("both", "calculus"):
    m = CalculusLM(len(chars), args.dim, steps=4, method="rk4")
    results["calculus"] = train("Calculus LM (ODE in depth, P+I+D field)", m)
    # Unique to the ODE model: change how carefully it 'thinks' AFTER training.
    results["calculus"]["solver_sweep"] = {
        f"{meth}-{st}": round(evaluate(m, steps=st, method=meth), 3)
        for meth, st in [("euler", 2), ("euler", 4), ("rk4", 2), ("rk4", 4), ("rk4", 8)]}
    print("  accuracy vs. thinking budget:", results["calculus"]["solver_sweep"], flush=True)
    results["calculus"]["sample"] = generate(m)
    print("\n--- Calculus LM writes: ---\n" + results["calculus"]["sample"], flush=True)
    torch.save(m.state_dict(), "calculus_lm.pt")

if args.model in ("both", "transformer"):
    m = TransformerLM(len(chars), args.dim, layers=3, max_len=max(512, args.block))
    results["transformer"] = train("Transformer baseline (3 layers)", m)
    results["transformer"]["sample"] = generate(m)
    print("\n--- Transformer writes: ---\n" + results["transformer"]["sample"])

results["settings"] = vars(args)
json.dump(results, open(args.out, "w"), indent=2)
print(f"\nSaved {args.out}")
