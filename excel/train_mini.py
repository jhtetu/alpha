"""
Train the MINI Calculus LM that lives in Calculus_LM.xlsx.

It is the same idea as calculus_lm.py, simplified so every step is a plain
Excel formula:
  - 4 channels, 10-character vocabulary
  - vector field  F = tanh(W · [P, I, velocity, acceleration] + b)
  - 2 Euler steps of size h = 0.5 (Python version uses RK4)
Weights are rounded to 3 decimals and saved to mini_weights.json.
"""
import json, math, random, torch

VOCAB = [" ", "a", "c", "e", "h", "m", "n", "o", "s", "t"]
D, T, DT, H = 4, 16, 1.0, 0.5


def make_corpus(n=3000, seed=0):
    rnd = random.Random(seed)
    S = ["cat", "man", "hen", "ant"]
    O = ["mat", "hat", "stone", "nest"]
    F = ["ham", "oats", "meat", "tea"]
    out = []
    for _ in range(n):
        r = rnd.random()
        if r < 0.45:
            out.append(f"the {rnd.choice(S)} sat on the {rnd.choice(O)}")
        elif r < 0.8:
            out.append(f"the {rnd.choice(S)} ate {rnd.choice(F)}")
        else:
            out.append(f"a {rnd.choice(S)} sat on a {rnd.choice(O)}")
    return " ".join(out) + " "


def forward(idx, p):
    """Exactly the computation the spreadsheet performs."""
    a = torch.exp(-p["lam"] * DT)                       # (D,)
    z = p["emb"][idx]                                   # (B,T,D)
    for _ in range(2):                                  # two Euler steps
        I, prev = [], torch.zeros_like(z[:, 0])
        for t in range(z.shape[1]):
            prev = a * prev + (1 - a) * z[:, t]
            I.append(prev)
        I = torch.stack(I, 1)
        z1 = torch.cat([z[:, :1], z[:, :-1]], 1)        # z_{t-1} (z_1 at start)
        z2 = torch.cat([z1[:, :1], z1[:, :-1]], 1)      # z_{t-2}
        V = (z - z1) / DT
        A = (z - 2 * z1 + z2) / DT ** 2
        X = torch.cat([z, I, V, A], -1)                 # (B,T,16)
        F = torch.tanh(X @ p["W"].T + p["b"])
        z = z + H * F
    return z @ p["Wout"].T + p["bout"]                  # logits (B,T,V)


if __name__ == "__main__":
    torch.manual_seed(0)
    text = make_corpus()
    data = torch.tensor([VOCAB.index(c) for c in text])
    p = {"emb": torch.randn(10, D) * 0.5, "log_lam": torch.log(torch.tensor([1.0, 0.4, 0.15, 0.05])),
         "W": torch.randn(D, 4 * D) * 0.3, "b": torch.zeros(D),
         "Wout": torch.randn(10, D) * 0.5, "bout": torch.zeros(10)}
    for v in p.values():
        v.requires_grad_(True)
    opt = torch.optim.Adam(p.values(), lr=0.03)
    for step in range(3000):
        ix = torch.randint(len(data) - T - 1, (64,))
        x = torch.stack([data[i:i + T] for i in ix]); y = torch.stack([data[i + 1:i + T + 1] for i in ix])
        p["lam"] = torch.exp(p["log_lam"])
        loss = torch.nn.functional.cross_entropy(forward(x, p).reshape(-1, 10), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(step, round(loss.item() / math.log(2), 3), "bits/char")
    w = {k: [[round(v, 3) for v in row] for row in p[k].detach().tolist()] for k in ["emb", "W", "Wout"]}
    w["b"] = [round(v, 3) for v in p["b"].tolist()]
    w["bout"] = [round(v, 3) for v in p["bout"].tolist()]
    w["lam"] = [round(v, 3) for v in torch.exp(p["log_lam"]).tolist()]
    w["vocab"] = VOCAB
    # accuracy with rounded weights
    q = {k: torch.tensor(w[k]) for k in ["emb", "W", "Wout", "b", "bout", "lam"]}
    x = torch.stack([data[i:i + T] for i in range(0, 8000, T)]); y = torch.stack([data[i + 1:i + T + 1] for i in range(0, 8000, T)])
    lg = forward(x, q)
    print("rounded weights: accuracy", (lg.argmax(-1) == y).float().mean().item(),
          "bits", torch.nn.functional.cross_entropy(lg.reshape(-1, 10), y.reshape(-1)).item() / math.log(2))
    json.dump(w, open("mini_weights.json", "w"), indent=1)
