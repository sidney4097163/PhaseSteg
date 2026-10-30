# -*- coding: utf-8 -*-
"""
Network-free analysis of the encryption layer (CPU only, a few minutes).

  1. classic cipher metrics : entropy, adjacent-pixel correlation (H/V/D), histogram chi-square,
                              NPCR/UACI (1-bit plaintext change; 1-bit key change), PNG bit-rate, speed
  2. error tolerance        : Gaussian noise of std sigma added to the ciphertext (simulating the
                              reveal-network error) -> PSNR of the decrypted image
  3. error propagation      : how many plaintext pixels are wrong after a +1 error on ONE cipher pixel
  4. figures                : error curve, cipher examples + histograms

  python cipher_analysis.py --test data/coco/test --n 100 --out results/cipher
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from common import list_images, load_rgb, psnr
from crypto_chaos import IEPS, CPCE, ShakeStream

SIGMAS = [0, 0.25, 0.5, 1, 2, 4, 8, 16]


def entropy(a):
    h = np.bincount(a.reshape(-1), minlength=256).astype(np.float64)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def corr(a, n=5000, rng=None):
    g = a.astype(np.float64).mean(-1) if a.ndim == 3 else a.astype(np.float64)
    H, W = g.shape
    rng = rng or np.random.default_rng(0)
    y = rng.integers(0, H - 1, n)
    x = rng.integers(0, W - 1, n)
    out = []
    for dy, dx in [(0, 1), (1, 0), (1, 1)]:
        out.append(float(np.corrcoef(g[y, x], g[y + dy, x + dx])[0, 1]))
    return out


def chi2(a):
    h = np.bincount(a.reshape(-1), minlength=256).astype(np.float64)
    e = h.sum() / 256
    return float(((h - e) ** 2 / e).sum())


def npcr_uaci(c1, c2):
    d = c1 != c2
    return float(d.mean() * 100), float(np.abs(c1.astype(float) - c2.astype(float)).mean() / 255 * 100)


def png_bpp(a):
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG", optimize=True)
    return len(b.getvalue()) * 8 / (a.shape[0] * a.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n_curve", type=int, default=30)
    ap.add_argument("--out", default="results/cipher")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = list_images(a.test)[:a.n]
    rng = np.random.default_rng(0)
    C = {"IEPS(n=1)": IEPS(rounds=1), "IEPS(n=2)": IEPS(rounds=2), "IEPS(n=3)": IEPS(rounds=3),
         "CPCE(g=0)": CPCE(guard=0), "CPCE(g=16)": CPCE(guard=16),
         "SHAKE-add(g=16)": ShakeStream(mode="add", guard=16), "SHAKE-XOR": ShakeStream(mode="xor")}
    res = {k: {m: [] for m in ["entropy", "corr_h", "corr_v", "corr_d", "chi2", "npcr_p", "uaci_p",
                               "npcr_p_fresh", "uaci_p_fresh", "npcr_k", "uaci_k", "png_bpp", "t_enc", "t_dec",
                               "wrongkey_psnr", "roundtrip_psnr"]} for k in C}
    plain = {m: [] for m in ["entropy", "corr_h", "corr_v", "corr_d", "chi2", "png_bpp"]}
    for idx, f in enumerate(files):
        p = load_rgb(f)
        plain["entropy"].append(entropy(p))
        ch, cv, cd = corr(p, rng=rng)
        plain["corr_h"].append(ch); plain["corr_v"].append(cv); plain["corr_d"].append(cd)
        plain["chi2"].append(chi2(p)); plain["png_bpp"].append(png_bpp(p))
        p2 = p.copy()
        p2[p.shape[0] // 2, p.shape[1] // 2, 0] ^= 1                      # 1-bit plaintext change
        for k, cip in C.items():
            t = time.time(); c = cip.encrypt(p, nonce=idx); te = time.time() - t
            t = time.time(); d = cip.decrypt(c.astype(np.float64), nonce=idx); td = time.time() - t
            r = res[k]
            r["t_enc"].append(te * 1e3); r["t_dec"].append(td * 1e3)
            r["roundtrip_psnr"].append(min(psnr(p, d), 100))
            r["entropy"].append(entropy(c))
            ch, cv, cd = corr(c, rng=rng)
            r["corr_h"].append(ch); r["corr_v"].append(cv); r["corr_d"].append(cd)
            r["chi2"].append(chi2(c)); r["png_bpp"].append(png_bpp(c))
            c2 = cip.encrypt(p2, nonce=idx)                                 # same nonce (worst case for CPCE)
            n_, u_ = npcr_uaci(c, c2); r["npcr_p"].append(n_); r["uaci_p"].append(u_)
            c3 = cip.encrypt(p2, nonce=idx + 100000)                        # fresh nonce (normal CPCE usage)
            n_, u_ = npcr_uaci(c, c3); r["npcr_p_fresh"].append(n_); r["uaci_p_fresh"].append(u_)
            wk = cip.wrong_key()
            c4 = wk.encrypt(p, nonce=idx)
            n_, u_ = npcr_uaci(c, c4); r["npcr_k"].append(n_); r["uaci_k"].append(u_)
            r["wrongkey_psnr"].append(psnr(p, wk.decrypt(c.astype(np.float64), nonce=idx)))
        if (idx + 1) % 20 == 0:
            print(f"  metrics {idx + 1}/{len(files)}", flush=True)

    summary = {"plain": {m: [float(np.mean(v)), float(np.std(v))] for m, v in plain.items()}}
    for k in C:
        summary[k] = {m: [float(np.mean(v)), float(np.std(v))] for m, v in res[k].items()}

    # ---- error propagation of a single +1 error
    p = load_rgb(files[0])
    prop = {}
    for k, cip in C.items():
        c = cip.encrypt(p, nonce=0).astype(np.float64)
        ref = cip.decrypt(c, nonce=0)
        cnt = []
        for t in range(20):
            e = c.copy()
            y, x, ch = rng.integers(0, p.shape[0]), rng.integers(0, p.shape[1]), rng.integers(0, 3)
            e[y, x, ch] = (e[y, x, ch] + 1) % 256
            cnt.append(int((cip.decrypt(e, nonce=0) != ref).sum()))
        prop[k] = float(np.mean(cnt))
    summary["single_error_propagation_pixels"] = prop

    # ---- error tolerance curve
    curve = {k: [] for k in C}
    curve["IEPS(n=2) literal Eq.9"] = []
    curve["identity (no cipher)"] = []
    for s in SIGMAS:
        acc = {k: [] for k in curve}
        for idx, f in enumerate(files[:a.n_curve]):
            p = load_rgb(f)
            noise = rng.normal(0, s, p.shape) if s > 0 else np.zeros(p.shape)
            acc["identity (no cipher)"].append(psnr(p, np.clip(np.rint(p + noise), 0, 255)))
            for k, cip in C.items():
                c = cip.encrypt(p, nonce=idx).astype(np.float64) + noise
                if k.startswith("IEPS") or k == "SHAKE-XOR":
                    c = np.clip(np.rint(c), 0, 255)                      # IEPS needs bytes
                acc[k].append(psnr(p, cip.decrypt(c, nonce=idx)))
            ie = C["IEPS(n=2)"]
            c = np.clip(np.rint(ie.encrypt(p).astype(np.float64) + noise), 0, 255)
            acc["IEPS(n=2) literal Eq.9"].append(psnr(p, ie.decrypt_literal(c)))
        for k in curve:
            curve[k].append(float(np.mean(acc[k])))
        print(f"  sigma={s}: " + ", ".join(f"{k}={curve[k][-1]:.2f}" for k in curve), flush=True)
    summary["error_curve"] = {"sigma": SIGMAS, **curve}
    json.dump(summary, open(out / "cipher_analysis.json", "w"), indent=2, ensure_ascii=False)

    # ---- table print
    print("\n%-12s %8s %8s %8s %8s %9s %8s %8s %8s %8s %8s %8s" % (
        "cipher", "entropy", "corr_h", "corr_v", "corr_d", "chi2", "NPCRp", "UACIp", "NPCRp*", "NPCRk", "UACIk", "bpp"))
    for k in ["plain", *C]:
        s = summary[k]
        g = lambda m: s[m][0] if m in s else float("nan")
        print("%-12s %8.4f %8.4f %8.4f %8.4f %9.1f %8.3f %8.3f %8.3f %8.3f %8.3f %8.2f" % (
            k, g("entropy"), g("corr_h"), g("corr_v"), g("corr_d"), g("chi2"), g("npcr_p"), g("uaci_p"),
            g("npcr_p_fresh"), g("npcr_k"), g("uaci_k"), g("png_bpp")))
    print("single +1 cipher error -> wrong plaintext pixels:", prop)

    # ---- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        styles = {"identity (no cipher)": ("k", "--", "无加密(参考)"), "IEPS(n=1)": ("#d62728", "-", "IEPS n=1"),
                  "IEPS(n=2)": ("#ff7f0e", "-", "IEPS n=2"), "IEPS(n=3)": ("#8c564b", "-", "IEPS n=3"),
                  "IEPS(n=2) literal Eq.9": ("#7f7f7f", ":", "IEPS 原文式(9)"),
                  "CPCE(g=0)": ("#1f77b4", "-.", "CPCE g=0"), "CPCE(g=16)": ("#2ca02c", "-", "CPCE g=16 (本文)"),
                  "SHAKE-add(g=16)": ("#17becf", "--", "SHAKE 模加 g=16"), "SHAKE-XOR": ("#9467bd", ":", "SHAKE 异或")}
        xs = [max(s, 0.1) for s in SIGMAS]
        for k, (col, ls, lab) in styles.items():
            ax.plot(xs, [min(v, 70) for v in curve[k]], ls, color=col, marker="o", ms=3, label=lab)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(["0" if v == 0 else f"{v:g}" for v in SIGMAS], fontsize=8)
        ax.minorticks_off()
        ax.set_ylim(top=min(70, ax.get_ylim()[1]))
        ax.set_xlabel("密文噪声标准差 σ (灰度级)")
        ax.set_ylabel("解密图像 PSNR / dB")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "error_curve.png", dpi=300)
        # examples
        p = load_rgb(files[0])
        ims = [("明文", p), ("IEPS 密文", C["IEPS(n=2)"].encrypt(p)), ("CPCE 密文", C["CPCE(g=16)"].encrypt(p, 0))]
        fig, axs = plt.subplots(2, 3, figsize=(9, 5.5))
        for j, (t, im) in enumerate(ims):
            axs[0, j].imshow(im); axs[0, j].set_title(t); axs[0, j].axis("off")
            for ci, col in enumerate("rgb"):
                axs[1, j].hist(im[..., ci].ravel(), bins=256, range=(0, 255), color=col, alpha=0.5, histtype="stepfilled")
            axs[1, j].set_xlim(0, 255)
        fig.tight_layout()
        fig.savefig(out / "cipher_examples.png", dpi=200)
        print("figures saved to", out)
    except ImportError:
        print("matplotlib not installed: figures skipped")


if __name__ == "__main__":
    main()
