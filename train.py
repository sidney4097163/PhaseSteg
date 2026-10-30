# -*- coding: utf-8 -*-
"""
Training.

Key design point: with a strong cipher the hiding network never sees the plaintext, only the
ciphertext, which is (statistically) i.i.d. uniform bytes. Hence by default the "secret" during
training is a fresh uniform random byte image (--train_cipher random); the real cipher (IEPS / CPCE)
is applied at evaluation time. --train_cipher ieps|cpce trains on real ciphertexts of secret crops
instead (used once to verify the equivalence).

examples
  python train.py --name coco_hinn_phase --data data/coco/train --val data/coco/test --arch hinn --rep phase --disc srm
  python train.py --name coco_isgi_lin  --data data/coco/train --val data/coco/test --arch isgi --rep lin --disc isgi --quant 0
"""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import models
from common import (CropDataset, channel, decode_rep, encode_rep, rec_loss, make_pairs, load_rgb, expand_rep, pool_rep,
                    seed_all, circ_err, psnr_mse)
from crypto_chaos import build_cipher


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True, help="folder of training PNGs (256x256)")
    ap.add_argument("--val", required=True, help="folder of test PNGs (for quick validation)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--arch", default="hinn", choices=["hinn", "isgi", "baluja"])
    ap.add_argument("--rep", default="phase", choices=["phase", "lin", "plain"],
                    help="phase/lin: hide a ciphertext with that representation; plain: hide the natural image (no encryption)")
    ap.add_argument("--disc", default="srm", choices=["srm", "isgi", "none"])
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--cscale", type=int, default=1,
                    help="cipher-domain redundancy factor s: the secret is encrypted at 1/s resolution and every "
                         "ciphertext symbol is repeated over an s x s block (s=1: off)")
    ap.add_argument("--quant", type=int, default=1, help="8-bit quantisation-aware training (STE)")
    ap.add_argument("--noise", type=int, default=0, help="random Gaussian/JPEG noise layer (robust variant)")
    ap.add_argument("--train_cipher", default="random", choices=["random", "ieps", "cpce"])
    ap.add_argument("--lam_c", type=float, default=1.0, help="cover/stego MSE weight")
    ap.add_argument("--cover_warmup", type=int, default=0,
                    help=">0: ramp the cover (and low-freq) weight from 1%% to 100%% over this many iterations, "
                         "so the reveal path is established first (helps non-invertible baselines escape the trivial solution)")
    ap.add_argument("--lam_s", type=float, default=1.0, help="secret reconstruction weight")
    ap.add_argument("--lam_low", type=float, default=1.0, help="Haar low-frequency consistency weight")
    ap.add_argument("--lam_adv", type=float, default=1e-3)
    ap.add_argument("--isgi_weights", type=int, default=0, help="use ISGI's alpha,beta,gamma=0.3,0.4,0.3 (lam_low=0)")
    ap.add_argument("--lr", type=float, default=None, help="default 1e-4 (hinn) / 1e-3 (isgi, as in the paper)")
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.lr is None:
        a.lr = 1e-3 if a.arch == "isgi" else 1e-4
    if a.isgi_weights:
        a.lam_c, a.lam_s, a.lam_adv, a.lam_low = 0.3, 0.4, 0.3, 0.0
    return a


def make_secret(batch_img, rep, train_cipher, ciphers, device, s=1):
    """returns (secret bytes 0..255 at 1/s resolution, low-res rep, network input rep)."""
    if rep == "plain":
        return None, batch_img, batch_img
    B, C, H, W = batch_img.shape
    if train_cipher == "random":
        c = torch.randint(0, 256, (B, C, H // s, W // s), device=device).float()
    else:
        small = F.avg_pool2d(batch_img, s) if s > 1 else batch_img
        arr = (small * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        cip = ciphers[train_cipher]
        out = [cip.encrypt(a, nonce=int(np.random.randint(1 << 30))) for a in arr]
        c = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float().to(device)
    r_small = encode_rep(c, rep)
    return c, r_small, expand_rep(r_small, s)


@torch.no_grad()
def validate(net, pairs, rep, device, quant, s=1):
    net.eval()
    g = torch.Generator().manual_seed(123)
    ps_c, mse_s = [], []
    for cov_p, sec_p in pairs:
        cov = torch.from_numpy(load_rgb(cov_p)).permute(2, 0, 1)[None].float().to(device) / 255
        if rep == "plain":
            c = torch.from_numpy(load_rgb(sec_p)).permute(2, 0, 1)[None].float().to(device)
            r = c / 255
        else:
            B, C, H, W = cov.shape
            c = torch.randint(0, 256, (B, C, H // s, W // s), generator=g).float().to(device)
            r = expand_rep(encode_rep(c, rep), s)
        st, _ = net.hide(cov, r)
        st = st.clamp(0, 1)
        st = torch.round(st * 255) / 255
        ps_c.append(psnr_mse(F.mse_loss(st * 255, cov * 255).item()))
        rh = net.reveal(st)
        ch = decode_rep(pool_rep(rh, s) if rep != "plain" else rh, rep)
        if rep == "phase":
            mse_s.append((circ_err(ch, c) ** 2).mean().item())
        else:
            mse_s.append(((ch - c) ** 2).mean().item())
    net.train()
    return float(np.mean(ps_c)), float(np.mean([psnr_mse(m) for m in mse_s]))


def main():
    a = get_args()
    seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "config.json", "w"), indent=2)

    net = models.build(a.arch, a.rep, a.blocks).to(dev)
    D = models.build_disc(a.disc)
    D = D.to(dev) if D is not None else None
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, betas=(0.9, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=1e-4) if D is not None else None
    warm = 1000

    def lr_at(it):
        if it < warm:
            return (it + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, a.iters - warm))) * 0.99 + 0.01

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    start, best = 0, -1e9
    ck = out / "last.pt"
    if a.resume and ck.exists():
        s = torch.load(ck, map_location=dev)
        net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"]); sch.load_state_dict(s["sch"])
        if D is not None and s.get("D"):
            D.load_state_dict(s["D"]); optD.load_state_dict(s["optD"])
        start, best = s["it"], s.get("best", best)
        print(f"resumed from iter {start}")

    ds = CropDataset(a.data, a.crop, length=10 ** 9)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch * (1 if a.rep != "plain" else 2), num_workers=a.workers,
                                     pin_memory=(dev.type == "cuda"), drop_last=True,
                                     persistent_workers=a.workers > 0)
    it_dl = iter(dl)
    ciphers = {"ieps": build_cipher("ieps"), "cpce": build_cipher("cpce")}
    val_pairs = make_pairs(a.val, a.n_val)
    logf = open(out / "log.csv", "a", newline="")
    log = csv.writer(logf)
    if start == 0:
        log.writerow(["iter", "loss", "l_cover", "l_rec", "l_low", "l_adv", "l_D", "val_psnr_c", "val_psnr_secret", "sec"])
    bce = torch.nn.BCEWithLogitsLoss()
    t0 = time.time()
    acc = np.zeros(6)
    n_acc = 0
    net.train()
    for it in range(start, a.iters):
        x = next(it_dl).to(dev, non_blocking=True)
        if a.rep == "plain":
            cover, simg = x[:a.batch], x[a.batch:]
            c_bytes, r, r_small = None, simg, simg
        else:
            cover = x
            c_bytes, r_small, r = make_secret(x, a.rep, a.train_cipher, ciphers, dev, a.cscale)

        stego, _ = net.hide(cover, r)
        attacked = channel(stego, quant=bool(a.quant), noise=bool(a.noise))
        r_hat = net.reveal(attacked)

        l_cover = F.mse_loss(stego, cover)
        if a.rep != "plain" and a.cscale > 1:
            l_rec = rec_loss(pool_rep(r_hat, a.cscale), r_small, a.rep)
        else:
            l_rec = rec_loss(r_hat, r, a.rep)
        l_low = F.mse_loss(models.lowpass(stego), models.lowpass(cover)) if a.lam_low > 0 else torch.zeros((), device=dev)
        if D is not None and a.lam_adv > 0:
            l_adv = bce(D(stego.clamp(0, 1)), torch.ones(stego.shape[0], device=dev))
        else:
            l_adv = torch.zeros((), device=dev)
        wc = min(1.0, max(0.01, (it + 1) / a.cover_warmup)) if a.cover_warmup > 0 else 1.0
        loss = wc * (a.lam_c * l_cover + a.lam_low * l_low) + a.lam_s * l_rec + a.lam_adv * l_adv

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()

        l_D = torch.zeros(())
        if D is not None and a.lam_adv > 0:
            optD.zero_grad(set_to_none=True)
            n = cover.shape[0]
            l_D = bce(D(cover), torch.ones(n, device=dev)) + bce(D(stego.detach().clamp(0, 1)), torch.zeros(n, device=dev))
            l_D.backward()
            optD.step()

        acc += np.array([loss.item(), l_cover.item(), l_rec.item(), l_low.item(), l_adv.item(), l_D.item()])
        n_acc += 1
        last = it + 1 == a.iters
        if (it + 1) % 200 == 0 or last:
            m = acc / n_acc
            print(f"[{a.name}] it {it + 1}/{a.iters} loss {m[0]:.5f} cover {m[1]:.6f} rec {m[2]:.6f} "
                  f"adv {m[4]:.3f} D {m[5]:.3f} lr {sch.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        if (it + 1) % a.val_every == 0 or last:
            m = acc / n_acc
            pc, ps = validate(net, val_pairs, a.rep, dev, a.quant, a.cscale)
            print(f"   val: PSNR(cover,stego)={pc:.2f} dB  PSNR(secret/cipher recovered)={ps:.2f} dB", flush=True)
            log.writerow([it + 1, *[f"{v:.6g}" for v in m], f"{pc:.3f}", f"{ps:.3f}", f"{time.time() - t0:.0f}"])
            logf.flush()
            acc[:] = 0
            n_acc = 0
            state = {"net": net.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "it": it + 1,
                     "D": D.state_dict() if D is not None else None, "optD": optD.state_dict() if D is not None else None,
                     "args": vars(a)}
            score = pc + ps
            if score > best:
                best = score
                torch.save({"net": net.state_dict(), "args": vars(a), "it": it + 1}, out / "best.pt")
            state["best"] = best
            torch.save(state, ck)
    torch.save({"net": net.state_dict(), "args": vars(a), "it": a.iters}, out / "final.pt")
    print("done", out)


if __name__ == "__main__":
    main()
