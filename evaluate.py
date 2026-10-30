# -*- coding: utf-8 -*-
"""
Evaluation under the realistic protocol:
  real cipher (IEPS / CPCE with per-pair nonce) -> hide -> stego rounded to 8 bit (and optionally attacked)
  -> reveal -> decrypt -> metrics on the DECRYPTED secret.

  python evaluate.py --run runs/coco_hinn_phase --test data/coco/test --ciphers cpce,cpce0,ieps
  python evaluate.py --run runs/coco_hinn_phase --test data/coco/test --ciphers cpce --attacks none,jpeg95,jpeg90,gauss1,gauss2 --tag robust
  python evaluate.py --run runs/coco_hinn_phase --test data/coco/test --ciphers cpce --save_stego stego/coco_hinn_phase
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import models
from common import (expand_rep, pool_rep, decode_rep, encode_rep, make_pairs, load_rgb, psnr, ssim, apd, payload_bpp, circ_err,
                    psnr_mse, jpeg_np, to_t, to_np)
from crypto_chaos import build_cipher

CIPHER_NAMES = {"none": "无加密", "ieps": "IEPS", "cpce": "CPCE(g=16)", "cpce0": "CPCE(g=0)"}


def attack(st_u8, name, rng):
    if name == "none":
        return st_u8
    if name.startswith("jpeg"):
        return jpeg_np(st_u8, int(name[4:]))
    if name.startswith("gauss"):
        s = float(name[5:])
        return np.clip(np.rint(st_u8 + rng.normal(0, s, st_u8.shape)), 0, 255).astype(np.uint8)
    raise ValueError(name)


def load_net(run, ckpt, dev):
    run = Path(run)
    for c in [ckpt, "final.pt", "best.pt", "last.pt"]:
        if (run / c).exists():
            s = torch.load(run / c, map_location=dev)
            a = s["args"]
            net = models.build(a["arch"], a["rep"], a.get("blocks", 8)).to(dev)
            net.load_state_dict(s["net"])
            net.eval()
            print(f"loaded {run / c} (iter {s.get('it')})")
            return net, a
    raise FileNotFoundError(run)


def grid(rows, path, scale=1):
    h, w = rows[0][0].shape[:2]
    G = np.full((len(rows) * (h + 4), len(rows[0]) * (w + 4), 3), 255, np.uint8)
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            G[i * (h + 4):i * (h + 4) + h, j * (w + 4):j * (w + 4) + w] = im
    Image.fromarray(G).save(path, quality=90)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--ciphers", default=None, help="comma list of none,ieps,cpce,cpce0 (default by rep)")
    ap.add_argument("--attacks", default="none")
    ap.add_argument("--n_pairs", type=int, default=1000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save_stego", default="", help="folder to save cover/stego PNG pairs (first cipher, no attack)")
    ap.add_argument("--n_vis", type=int, default=4)
    ap.add_argument("--ieps_rounds", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    net, targs = load_net(a.run, a.ckpt, dev)
    rep = targs["rep"]
    s = int(targs.get("cscale", 1))
    ciphers = a.ciphers.split(",") if a.ciphers else (["none"] if rep == "plain" else ["cpce", "cpce0", "ieps"])
    if rep == "plain":
        ciphers = ["none"]
    attacks = a.attacks.split(",")
    pairs = make_pairs(a.test, a.n_pairs)
    dname = Path(a.test).parent.name
    tag = f"{dname}{('_' + a.tag) if a.tag else ''}"
    outdir = Path(a.run)
    rng = np.random.default_rng(0)
    if a.save_stego:
        sdir = Path(a.save_stego)
        (sdir / "cover").mkdir(parents=True, exist_ok=True)
        (sdir / "stego").mkdir(parents=True, exist_ok=True)

    cobj = {c: build_cipher(c, rounds=a.ieps_rounds) for c in ciphers}
    rows = []
    vis = {c: [] for c in ciphers}
    for i, (cp, sp) in enumerate(pairs):
        cover = load_rgb(cp)
        secret = load_rgb(sp)
        if s > 1:   # encrypt a 1/s-resolution secret, upsample after decryption
            H, W = secret.shape[:2]
            sec_small = np.asarray(Image.fromarray(secret).resize((W // s, H // s), Image.BICUBIC))
            up = lambda a_: np.asarray(Image.fromarray(a_).resize((W, H), Image.BICUBIC))
        else:
            sec_small, up = secret, (lambda a_: a_)
        for cn in ciphers:
            cip = cobj[cn]
            c = cip.encrypt(sec_small, nonce=i)
            r = expand_rep(encode_rep(to_t(c, dev), rep), s) if rep != "plain" else to_t(c, dev) / 255
            st, _ = net.hide(to_t(cover, dev) / 255, r)
            st_u8 = np.clip(np.rint(to_np(st) * 255), 0, 255).astype(np.uint8)
            if a.save_stego and cn == ciphers[0]:
                Image.fromarray(cover).save(sdir / "cover" / f"{i:05d}.png")
                Image.fromarray(st_u8).save(sdir / "stego" / f"{i:05d}.png")
            base = dict(pair=i, cipher=cn, psnr_c=psnr(cover, st_u8), ssim_c=ssim(cover, st_u8), apd_c=apd(cover, st_u8))
            for an in attacks:
                st_a = attack(st_u8, an, rng)
                rh = net.reveal(to_t(st_a, dev) / 255)
                ch = to_np(decode_rep(pool_rep(rh, s) if rep != "plain" else rh, rep)).astype(np.float64)
                if rep == "phase" or cn.startswith("cpce"):
                    e_enc = circ_err(ch, c.astype(np.float64))
                else:
                    e_enc = np.abs(ch - c)
                s_hat = up(cip.decrypt(ch, nonce=i))
                s_wrong = up(cip.wrong_key().decrypt(ch, nonce=i))
                row = dict(base, attack=an, psnr_enc=psnr_mse(np.mean(e_enc ** 2)),
                           psnr_s=psnr(secret, s_hat), ssim_s=ssim(secret, s_hat), apd_s=apd(secret, s_hat),
                           payload=payload_bpp(secret, s_hat), psnr_wrong=psnr(secret, s_wrong),
                           ssim_wrong=ssim(secret, s_wrong),
                           psnr_resample_ceiling=psnr(secret, up(sec_small)) if s > 1 else 100.0)
                if cn == "ieps":
                    sl = up(cip.decrypt_literal(ch, nonce=i))
                    row["psnr_literal"] = psnr(secret, sl)
                rows.append(row)
                if an == "none" and len(vis[cn]) < a.n_vis:
                    diff = np.clip(np.abs(st_u8.astype(int) - cover.astype(int)) * 10, 0, 255).astype(np.uint8)
                    vis[cn].append([cover, st_u8, diff, secret, up(c),
                                    up(np.clip(np.rint(ch), 0, 255).astype(np.uint8)), s_hat, s_wrong])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pairs)}", flush=True)

    with open(outdir / f"eval_{tag}_per_image.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0].keys()).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for cn in ciphers:
        for an in attacks:
            sub = [r for r in rows if r["cipher"] == cn and r["attack"] == an]
            d = {}
            for k in sub[0]:
                if k in ("pair", "cipher", "attack"):
                    continue
                v = np.array([r[k] for r in sub], dtype=float)
                d[k] = [float(v.mean()), float(v.std())]
            summ[f"{cn}|{an}"] = d
            print(f"{CIPHER_NAMES.get(cn, cn):12s} {an:7s} | stego PSNR {d['psnr_c'][0]:.2f} SSIM {d['ssim_c'][0]:.4f} | "
                  f"cipher PSNR {d['psnr_enc'][0]:.2f} | secret PSNR {d['psnr_s'][0]:.2f} SSIM {d['ssim_s'][0]:.4f} | "
                  f"wrong-key PSNR {d['psnr_wrong'][0]:.2f}" + (f" | literal Eq.9 {d['psnr_literal'][0]:.2f}" if 'psnr_literal' in d else ""))
    json.dump({"run": str(a.run), "train_args": targs, "test": a.test, "n_pairs": len(pairs), "summary": summ},
              open(outdir / f"eval_{tag}.json", "w"), indent=2, ensure_ascii=False)
    for cn in ciphers:
        if vis[cn]:
            grid(vis[cn], outdir / f"vis_{tag}_{cn}.jpg")
    print("saved", outdir / f"eval_{tag}.json")


if __name__ == "__main__":
    main()
