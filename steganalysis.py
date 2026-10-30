# -*- coding: utf-8 -*-
"""
Security test with a deep steganalyser (SRNet, colour input), replacing the StegExpose test of ISGI
(StegExpose targets LSB replacement and is not meaningful for deep hiding).

Input: a folder produced by  evaluate.py --save_stego DIR  (DIR/cover/*.png, DIR/stego/*.png, same names).
Pairs are split 60/10/30 (train/val/test) by image index; cover and its stego are always in the same
split and in the same mini-batch (pair constraint).

  python steganalysis.py --dir stego/coco_hinn_phase --epochs 60
Outputs test accuracy, AUC and P_E = min_t (P_FA + P_MD)/2  (0.5 = undetectable).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import load_rgb, seed_all
from models import SRNet


def auc_pe(scores, labels):
    s, l = np.asarray(scores), np.asarray(labels)
    order = np.argsort(-s)
    l = l[order]
    P, N = l.sum(), (1 - l).sum()
    tpr = np.concatenate([[0], np.cumsum(l) / P])
    fpr = np.concatenate([[0], np.cumsum(1 - l) / N])
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    pe = float(np.min(0.5 * (fpr + (1 - tpr))))
    return auc, pe


class PairSet(torch.utils.data.Dataset):
    def __init__(self, names, root, aug):
        self.names, self.root, self.aug = names, Path(root), aug

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        c = load_rgb(self.root / "cover" / self.names[i])
        s = load_rgb(self.root / "stego" / self.names[i])
        if self.aug:
            k = random.randint(0, 3)
            c, s = np.rot90(c, k), np.rot90(s, k)
            if random.random() < 0.5:
                c, s = c[:, ::-1], s[:, ::-1]
        x = np.stack([c, s]).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2), torch.tensor([0, 1])


def run_epoch(net, dl, dev, opt=None):
    train = opt is not None
    net.train(train)
    scores, labels, losses = [], [], []
    with torch.set_grad_enabled(train):
        for x, y in dl:
            x = x.flatten(0, 1).to(dev)
            y = y.flatten().to(dev)
            out = net(x)
            loss = F.cross_entropy(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            losses.append(loss.item())
            scores += torch.softmax(out, 1)[:, 1].detach().cpu().tolist()
            labels += y.cpu().tolist()
    acc = float(np.mean((np.array(scores) > 0.5) == np.array(labels)))
    return float(np.mean(losses)), acc, scores, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_pairs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    root = Path(a.dir)
    names = sorted(p.name for p in (root / "stego").glob("*.png"))
    rnd = random.Random(a.seed)
    rnd.shuffle(names)
    n = len(names)
    tr, va, te = names[:int(.6 * n)], names[int(.6 * n):int(.7 * n)], names[int(.7 * n):]
    mk = lambda ns, aug, sh: torch.utils.data.DataLoader(PairSet(ns, root, aug), batch_size=a.batch_pairs, shuffle=sh,
                                                         num_workers=a.workers, drop_last=sh)
    dtr, dva, dte = mk(tr, True, True), mk(va, False, False), mk(te, False, False)
    net = SRNet().to(dev)
    opt = torch.optim.Adamax(net.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, [int(a.epochs * .6), int(a.epochs * .85)], 0.1)
    best, best_state = -1, None
    for ep in range(a.epochs):
        lt, at, _, _ = run_epoch(net, dtr, dev, opt)
        lv, av, _, _ = run_epoch(net, dva, dev)
        sch.step()
        if av >= best:
            best, best_state = av, {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"ep {ep + 1}/{a.epochs} train loss {lt:.4f} acc {at:.4f} | val acc {av:.4f}", flush=True)
    net.load_state_dict(best_state)
    _, acc, sc, lb = run_epoch(net, dte, dev)
    auc, pe = auc_pe(sc, lb)
    res = {"dir": str(root), "n_pairs": n, "test_pairs": len(te), "best_val_acc": best, "test_acc": acc, "auc": auc, "P_E": pe}
    print(json.dumps(res, indent=2))
    json.dump(res, open(root / "steganalysis_srnet.json", "w"), indent=2)


if __name__ == "__main__":
    main()
