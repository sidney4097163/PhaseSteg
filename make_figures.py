# -*- coding: utf-8 -*-
"""
Publication figures (vector PDF/EPS/SVG + 600-dpi TIFF).

  python make_figures.py --fig 1                 # framework diagram (no data needed)
  python make_figures.py --fig 2                 # error curve, reads results/cipher_v2/cipher_analysis.json
  python make_figures.py --fig 3                 # visual comparison, needs runs/abl_nodisc and runs/coco_isgi_lin (GPU or CPU)
Outputs go to figures/.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none"})
BLUE = "#2a6fdb"
GREY = "#8a8a8a"


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ["pdf", "eps", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tif", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    print("saved", OUT / name, "(.pdf .eps .svg .tif)")


# ------------------------------------------------------------------ Fig. 1
def box(ax, x, y, w, h, lines, main=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=(0.165, 0.435, 0.859, 0.12) if main else "white",
                                ec=BLUE if main else GREY, lw=1.4 if main else 0.9))
    n = len(lines)
    for i, t in enumerate(lines):
        yy = y + h / 2 + (n - 1) * 0.11 - i * 0.22
        ax.text(x + w / 2, yy, t, ha="center", va="center", fontsize=8 if i == 0 else 6.5,
                fontweight="bold" if (i == 0 and main) else "normal", color="black" if i == 0 or main else "#444444")


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#666666"))


def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.05, 2.65), 9.9, 2.25, boxstyle="round,pad=0,rounding_size=0.1", fc="#f2f2f2", ec="none"))
    ax.add_patch(FancyBboxPatch((0.05, 0.1), 9.9, 2.3, boxstyle="round,pad=0,rounding_size=0.1", fc="#f2f2f2", ec="none"))
    ax.text(0.15, 4.75, "Sender", fontsize=7, color="#555555", va="top")
    ax.text(0.15, 2.3, "Receiver", fontsize=7, color="#555555", va="top")
    box(ax, 0.2, 3.5, 0.95, 0.6, ["Secret S"])
    box(ax, 1.4, 3.35, 2.6, 0.9, ["CPCE encryption", "1/2 downsample + guard band", "permute + mod-256 keystream"], True)
    box(ax, 4.2, 3.5, 1.35, 0.6, ["Phase φ(U)", "2×2 repeated"])
    box(ax, 4.2, 2.8, 1.35, 0.5, ["Cover C"])
    box(ax, 5.85, 2.8, 1.8, 1.45, ["HINN forward", "Haar + 8 couplings", "quantization-aware"], True)
    box(ax, 8.0, 3.2, 1.75, 0.7, ["Stego H", "8-bit PNG"])
    box(ax, 7.6, 0.75, 2.15, 0.9, ["HINN inverse", "shared weights, y₂ = 0"], True)
    box(ax, 4.7, 0.9, 2.5, 0.6, ["Average 2×2 copies", "atan2 → real-valued Û"])
    box(ax, 2.0, 0.75, 2.3, 0.9, ["CPCE decryption", "circular unwrap + clip", "upsample ×2"], True)
    box(ax, 0.2, 0.9, 1.4, 0.6, ["Recovered Ŝ"])
    box(ax, 2.0, 2.05, 2.3, 0.45, ["Key K + nonce N"])
    arrow(ax, (1.15, 3.8), (1.4, 3.8)); arrow(ax, (4.0, 3.8), (4.2, 3.8)); arrow(ax, (5.55, 3.8), (5.85, 3.8))
    arrow(ax, (5.55, 3.05), (5.85, 3.05)); arrow(ax, (7.65, 3.55), (8.0, 3.55)); arrow(ax, (8.875, 3.2), (8.875, 1.65))
    arrow(ax, (7.6, 1.2), (7.2, 1.2)); arrow(ax, (4.7, 1.2), (4.3, 1.2)); arrow(ax, (2.0, 1.2), (1.6, 1.2))
    arrow(ax, (2.7, 2.5), (2.7, 3.35)); arrow(ax, (3.15, 2.05), (3.15, 1.65))
    ax.text(8.75, 2.45, "public channel", fontsize=7, color="#555555", ha="right")
    save(fig, "fig1_framework")


# ------------------------------------------------------------------ Fig. 2
def fig2(path):
    d = json.load(open(path))["error_curve"]
    s = d["sigma"][1:]
    series = [("identity (no cipher)", "No encryption (reference)", "#555555", "--", 1.2),
              ("CPCE(g=16)", "CPCE, g = 16 (ours)", BLUE, "-", 2.0),
              ("SHAKE-XOR", "SHAKE-256 XOR", "#9a9a9a", ":", 1.4),
              ("CPCE(g=0)", "CPCE, g = 0", "#9a9a9a", "-", 1.2),
              ("IEPS(n=1)", "IEPS, n = 1", "#e07a7a", "-", 1.0),
              ("IEPS(n=2)", "IEPS, n = 2 (ISGI)", "#c62828", "-", 1.4),
              ("IEPS(n=2) literal Eq.9", "IEPS, Eq. (9) of ISGI", "#c62828", ":", 1.0)]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for key, lab, col, ls, lw in series:
        ax.plot(s, d[key][1:], ls, color=col, lw=lw, marker="o", ms=2.5, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks(s)
    ax.set_xticklabels([f"{v:g}" for v in s])
    ax.set_ylim(0, 65)
    ax.set_xlabel("Ciphertext noise std σ (grey levels)")
    ax.set_ylabel("Decrypted PSNR (dB)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, "fig2_error_curve")


# ------------------------------------------------------------------ Fig. 3
def fig3(test_dir, pairs, cpu):
    import numpy as np
    import torch
    from common import load_rgb, make_pairs, encode_rep, decode_rep, expand_rep, pool_rep, to_t, to_np
    from crypto_chaos import build_cipher
    from evaluate import load_net
    from PIL import Image
    dev = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    cols = ["Cover", "Stego", "|Residual| ×10", "Secret", "Ciphertext", "Extracted cipher.", "Decrypted", "Wrong key"]
    allp = make_pairs(test_dir, None)
    rows, names = [], []
    for run, cname, label in [("runs/abl_nodisc", "cpce", "Proposed"), ("runs/coco_isgi_lin", "ieps", "ISGI")]:
        net, targs = load_net(run, "final.pt", dev)
        rep, s = targs["rep"], int(targs.get("cscale", 1))
        cip = build_cipher(cname)
        with torch.no_grad():
            for i in pairs:
                cp, sp = allp[i]
                cover, secret = load_rgb(cp), load_rgb(sp)
                H, W = secret.shape[:2]
                small = np.asarray(Image.fromarray(secret).resize((W // s, H // s), Image.BICUBIC)) if s > 1 else secret
                up = (lambda a: np.asarray(Image.fromarray(a).resize((W, H), Image.BICUBIC))) if s > 1 else (lambda a: a)
                c = cip.encrypt(small, nonce=i)
                st, _ = net.hide(to_t(cover, dev) / 255, expand_rep(encode_rep(to_t(c, dev), rep), s))
                st8 = np.clip(np.rint(to_np(st) * 255), 0, 255).astype(np.uint8)
                rh = net.reveal(to_t(st8, dev) / 255)
                ch = to_np(decode_rep(pool_rep(rh, s), rep)).astype(np.float64)
                res = np.clip(np.abs(st8.astype(int) - cover.astype(int)) * 10, 0, 255).astype(np.uint8)
                rows.append([cover, st8, res, secret, up(c), up(np.clip(np.rint(ch) % 256, 0, 255).astype(np.uint8)),
                             up(cip.decrypt(ch, nonce=i)), up(cip.wrong_key().decrypt(ch, nonce=i))])
                names.append(label)
    fig, axs = plt.subplots(len(rows), 8, figsize=(7.2, 0.95 * len(rows)))
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            axs[r, c].imshow(im); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            if r == 0:
                axs[r, c].set_title(cols[c], fontsize=6)
        axs[r, 0].set_ylabel(names[r], fontsize=6)
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig3_visual.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(OUT / "fig3_visual.tif", bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    print("saved", OUT / "fig3_visual (.pdf .tif)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--json", default="results/cipher_v2/cipher_analysis.json")
    ap.add_argument("--test", default="data/coco/test")
    ap.add_argument("--pairs", default="0,1")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.fig == 1:
        fig1()
    elif a.fig == 2:
        fig2(a.json)
    else:
        fig3(a.test, [int(x) for x in a.pairs.split(",")], a.cpu)
