"""Обучение CloudNet по разметке «процент облачности в ячейке сетки 4x3».

Каждый снимок разбит на 12 ячеек; эксперт указал долю облаков в каждой ячейке.
Сеть выдаёт попиксельную вероятность облака; функция потерь сравнивает среднюю
вероятность в каждой ячейке с экспертной оценкой (слабое обучение сегментации),
а регуляризатор подталкивает вероятности к 0 или 1, чтобы получалась чёткая маска.

Чтобы маска ложилась именно на облака, а не «размазывалась» по ячейке, используются
псевдомаски: внутри каждой ячейки с оценкой L% самые белые пиксели (верхние L%
по «белизне» — яркость минус насыщенность) считаются облаком, самые тёмные —
не облаком, а пограничная полоса не участвует в обучении.

Запуск:
    python train.py --images data/images --epochs 45 --out weights/cloudnet.pt
    python train.py --images data/images --cv 5          # кросс-валидация
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from cloudnet import CloudNet, load_image, to_tensor, MEAN, STD

GRID_R, GRID_C = 3, 4


def pseudo_mask(img, lab, band=0.12):
    """Псевдомаска облаков по разметке ячеек и относительной белизне пикселей."""
    from scipy.ndimage import uniform_filter
    a = img.astype(np.float32)
    mn, mx = a.min(-1), a.max(-1)
    white = uniform_filter(mn - 0.5 * (mx - mn), size=5)
    H, W = mn.shape
    th, tw = H // GRID_R, W // GRID_C
    m = np.zeros((H, W), np.float32)
    w = np.zeros((H, W), np.float32)
    for r in range(GRID_R):
        for c in range(GRID_C):
            L = float(lab[r, c])
            sl = (slice(r * th, (r + 1) * th), slice(c * tw, (c + 1) * tw))
            v = white[sl]
            if L <= 0.0:
                w[sl] = 1.0
                continue
            if L >= 1.0:
                m[sl] = 1.0; w[sl] = 1.0
                continue
            hi = np.quantile(v, min(1.0, 1 - L + band))   # выше — уверенно облако
            lo = np.quantile(v, max(0.0, 1 - L - band))   # ниже — уверенно не облако
            m[sl] = (v >= hi).astype(np.float32)
            w[sl] = ((v >= hi) | (v <= lo)).astype(np.float32)
    return m, w


def load_dataset(images_dir, ann_path):
    ann = json.load(open(ann_path))["labels"]
    xs, ys, names = [], [], []
    for name, v in sorted(ann.items()):
        p = os.path.join(images_dir, name)
        if not os.path.exists(p):
            continue
        xs.append(load_image(p))
        ys.append(np.array(v, dtype=np.float32).reshape(GRID_R, GRID_C) / 100.0)
        names.append(name)
    return np.stack(xs), np.stack(ys), names


def augment(img, lab, pm, pw):
    """Отражения (вместе с сеткой меток и псевдомаской) и фотометрические искажения."""
    if random.random() < 0.5:
        img, lab, pm, pw = img[:, ::-1], lab[:, ::-1], pm[:, ::-1], pw[:, ::-1]
    if random.random() < 0.5:
        img, lab, pm, pw = img[::-1], lab[::-1], pm[::-1], pw[::-1]
    img = img.astype(np.float32)
    img = img * random.uniform(0.85, 1.15) + random.uniform(-15, 15)          # яркость/контраст
    img = img * np.random.uniform(0.93, 1.07, size=(1, 1, 3))                 # баланс каналов
    img = np.clip(img, 0, 255)
    return (np.ascontiguousarray(img), np.ascontiguousarray(lab),
            np.ascontiguousarray(pm), np.ascontiguousarray(pw))


def batch_tensor(imgs):
    t = torch.from_numpy(np.stack(imgs)).float().permute(0, 3, 1, 2) / 255.0
    return (t - MEAN) / STD


def tile_means(prob):
    return F.adaptive_avg_pool2d(prob, (GRID_R, GRID_C)).squeeze(1)


def train_model(X, Y, epochs=45, bs=8, lr=3e-4, seed=0, log=None):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = CloudNet(pretrained=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(X) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    PM, PW = zip(*[pseudo_mask(X[i], Y[i]) for i in range(len(X))])
    history = []
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(X))
        tot, n = 0.0, 0
        for b in range(0, len(idx), bs):
            bi = idx[b:b + bs]
            pairs = [augment(X[i], Y[i], PM[i], PW[i]) for i in bi]
            x = batch_tensor([p[0] for p in pairs])
            y = torch.from_numpy(np.stack([p[1] for p in pairs]))
            pm = torch.from_numpy(np.stack([p[2] for p in pairs]))[:, None]
            pw = torch.from_numpy(np.stack([p[3] for p in pairs]))[:, None]
            logits = model(x)
            prob = torch.sigmoid(logits)
            fit = F.smooth_l1_loss(tile_means(prob), y, beta=0.1)
            pix = (F.binary_cross_entropy_with_logits(logits, pm, reduction="none") * pw).sum() / pw.sum().clamp(min=1)
            binar = (prob * (1 - prob)).mean()
            loss = fit + 0.3 * pix + 0.05 * binar
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += fit.item() * len(bi); n += len(bi)
        history.append(tot / n)
        if log:
            log(f"эпоха {ep + 1:02d}/{epochs}  потеря {tot / n:.4f}")
    return model, history


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = []
    for i in range(0, len(X), 8):
        x = batch_tensor(list(X[i:i + 8]))
        p = torch.sigmoid(model(x))
        p = (p + torch.flip(torch.sigmoid(model(torch.flip(x, [3]))), [3])
             + torch.flip(torch.sigmoid(model(torch.flip(x, [2]))), [2])) / 3
        out.append(p[:, 0].numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--ann", default="data/annotations.json")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--cv", type=int, default=0)
    ap.add_argument("--out", default="weights/cloudnet.pt")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 2)
    X, Y, names = load_dataset(a.images, a.ann)
    print(f"снимков в разметке: {len(X)}")

    if a.cv:
        rng = np.random.RandomState(42)
        order = rng.permutation(len(X))
        folds = np.array_split(order, a.cv)
        results = []
        for k, test_idx in enumerate(folds):
            tr = np.setdiff1d(order, test_idx)
            t0 = time.time()
            model, _ = train_model(X[tr], Y[tr], epochs=a.epochs, seed=k)
            P = predict(model, X[test_idx])
            for j, i in enumerate(test_idx):
                valid = X[i].max(-1) >= 30
                soft = 100.0 * P[j][valid].mean()
                hard = 100.0 * ((P[j] >= 0.5) & valid).sum() / valid.sum()
                results.append({"name": names[i], "fold": k, "true": float(Y[i].mean() * 100),
                                "pred": float(soft), "pred_mask": float(hard),
                                "tiles_true": (Y[i] * 100).round().tolist(),
                                "tiles_pred": (tile_means(torch.from_numpy(P[j])[None, None])[0] * 100).round().tolist()})
            print(f"фолд {k + 1}/{a.cv} готов за {time.time() - t0:.0f} c")
        os.makedirs("reports", exist_ok=True)
        json.dump(results, open("reports/cv_results.json", "w"), ensure_ascii=False, indent=1)
        t = np.array([r["true"] for r in results]); p = np.array([r["pred"] for r in results])
        print(f"MAE по снимку: {np.abs(t - p).mean():.1f} п.п.")
        return

    history_log = []
    model, hist = train_model(X, Y, epochs=a.epochs, log=lambda s: (print(s), history_log.append(s)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(model.state_dict(), a.out)
    os.makedirs("reports", exist_ok=True)
    json.dump(hist, open("reports/train_history.json", "w"))
    print("веса сохранены:", a.out)


if __name__ == "__main__":
    main()
