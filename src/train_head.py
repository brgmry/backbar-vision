"""Train a projection head + brand classifier on frozen DINOv2 features of external data only.

Loss: supervised contrastive on brand (photos of the same brand pull together) + cross-entropy brand
classifier over brands with at least MIN_PER_BRAND images. Validation = held-out 10% of products and
of in-the-wild images, never a test set.
Usage: python src/train_head.py   -> data/train/head.pt
"""
import csv
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from models import DEVICE

SEED = 0
MIN_PER_BRAND = 30  # training images a brand needs to get its own classifier output
DIM = 256  # embedding size out of the head
HIDDEN = 1024
DROPOUT = 0.1
EPOCHS = 30
BATCH = 512
PER_BRAND = 8  # images per sampled brand in a batch, so every batch has positives
TEMP = 0.1  # supervised contrastive temperature
NOISE = 0.02  # feature-space noise as light augmentation
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_EVERY = 10  # one in ten products, and one in ten wild images without a product, is validation
VAL_QUERIES = 3000  # wild validation images scored per evaluation


class Head(torch.nn.Module):
    def __init__(self, n_brands: int, d_in: int = 768):
        super().__init__()
        self.proj = torch.nn.Sequential(torch.nn.Linear(d_in, HIDDEN), torch.nn.GELU(), torch.nn.Dropout(DROPOUT),
                                        torch.nn.Linear(HIDDEN, DIM))
        self.cls = torch.nn.Linear(DIM, n_brands)

    def embed(self, x):
        return F.normalize(self.proj(x), dim=-1)

    def forward(self, x):
        z = self.embed(x)
        return z, self.cls(z)


def supcon(z, y):
    """Supervised contrastive loss: every other sample with the same label is a positive."""
    sim = z @ z.T / TEMP
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has = pos.any(1)
    return -(log_prob * pos).sum(1)[has].div(pos.sum(1)[has]).mean()


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    with open("data/train/index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    feats = np.load("data/train/feats.npy").astype(np.float32)
    ok = np.load("data/train/feats_ok.npy")
    keep = [i for i in range(len(rows)) if ok[i] and rows[i]["brand_key"]]

    # split by product (catalogue) and by image (wild), so validation products are unseen
    products = sorted({rows[i]["product_key"] for i in keep if rows[i]["product_key"]})
    random.shuffle(products)
    val_products = set(products[: len(products) // VAL_EVERY])
    val = [i for i in keep if rows[i]["product_key"] in val_products
           or (not rows[i]["product_key"] and random.random() < 1 / VAL_EVERY)]
    val_set = set(val)
    train = [i for i in keep if i not in val_set]

    counts = Counter(rows[i]["brand_key"] for i in train)
    brands = sorted(b for b, c in counts.items() if c >= MIN_PER_BRAND)
    bidx = {b: k for k, b in enumerate(brands)}
    all_brands = {b: k for k, b in enumerate(sorted({rows[i]["brand_key"] for i in keep}))}
    print(f"train {len(train)}, val {len(val)}, classifier brands {len(brands)}, all brands {len(all_brands)}")

    X = torch.tensor(feats, device=DEVICE)
    yb = torch.tensor([all_brands.get(r["brand_key"], -1) for r in rows], device=DEVICE)
    yc = torch.tensor([bidx.get(r["brand_key"], -1) for r in rows], device=DEVICE)
    model = Head(len(brands)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    # balanced-ish batches: sample brands, then up to PER_BRAND images per brand, so positives exist
    by_brand = defaultdict(list)
    for i in train:
        by_brand[rows[i]["brand_key"]].append(i)
    multi = [b for b, v in by_brand.items() if len(v) >= 2]

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for _ in range(len(train) // BATCH):
            bs = random.sample(multi, BATCH // PER_BRAND)
            idx = [i for b in bs for i in random.sample(by_brand[b], min(PER_BRAND, len(by_brand[b])))]
            idx = torch.tensor(idx, device=DEVICE)
            x = X[idx] + NOISE * torch.randn_like(X[idx])
            z, logits = model(F.normalize(x, dim=-1))
            loss = supcon(z, yb[idx])
            m = yc[idx] >= 0
            if m.any():
                loss = loss + F.cross_entropy(logits[m], yc[idx][m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        if epoch % 5 == 4 or epoch == EPOCHS - 1:
            print(f"epoch {epoch + 1}: loss {np.mean(losses):.3f}  " + evaluate(model, X, rows, train, val, yc), flush=True)
    torch.save({"state": model.state_dict(), "brands": brands, "dim": DIM}, "data/train/head.pt")


@torch.no_grad()
def evaluate(model, X, rows, train, val, yc) -> str:
    """Val brand accuracy by nearest catalogue neighbour (raw vs head) and by classifier."""
    model.eval()
    gal = [i for i in train if rows[i]["gallery"] == "1" and not rows[i]["source"].startswith("wild")]
    q = [i for i in val if rows[i]["source"].startswith("wild")][:VAL_QUERIES]  # real photos vs catalogue: the hard way
    gb = [rows[i]["brand_key"] for i in gal]
    out = []
    for name, f in (("raw", lambda t: t), ("head", model.embed)):
        G, Q = f(X[gal]), f(X[q])
        nn = (Q @ G.T).argmax(1).cpu().tolist()
        out.append(f"{name} NN {np.mean([gb[j] == rows[i]['brand_key'] for i, j in zip(q, nn, strict=True)]):.3f}")
    m = [i for i in q if yc[i] >= 0]
    if m:
        pred = model(X[m])[1].argmax(1)
        out.append(f"cls {(pred == yc[m]).float().mean().item():.3f} on {len(m)}")
    return "wild->catalogue brand acc: " + ", ".join(out)


if __name__ == "__main__":
    main()
