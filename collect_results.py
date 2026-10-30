# -*- coding: utf-8 -*-
"""
Gather all results into results/summary.md and pack everything needed for writing the paper into
results_pack.zip (json / csv logs / figures; no checkpoints). Send results_pack.zip back.

  python collect_results.py
"""
import json
import shutil
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "results"
PACK = ROOT / "results_pack"


def f(v, k=2):
    return f"{v[0]:.{k}f}±{v[1]:.{k}f}" if isinstance(v, list) else str(v)


def main():
    import argparse, fnmatch
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack_runs", default="*", help="comma list of run-name patterns to include in the zip")
    args = ap.parse_args()
    pats = args.pack_runs.split(",")
    OUT.mkdir(exist_ok=True)
    lines = ["# 实验结果汇总\n"]
    for ca in sorted(OUT.glob("*/cipher_analysis.json")):
        d = json.load(open(ca))
        lines.append("## 加密层分析\n")
        lines.append("| 方法 | 熵 | 相关H | 相关V | 相关D | χ² | NPCR_p | UACI_p | NPCR_p(新nonce) | UACI_p(新nonce) | NPCR_k | UACI_k | PNG bpp | 加密ms | 解密ms | 错误密钥PSNR |")
        lines.append("|---" * 16 + "|")
        for k, s in d.items():
            if not isinstance(s, dict) or "entropy" not in s:
                continue
            g = lambda m, n=4: f"{s[m][0]:.{n}f}" if m in s else "—"
            lines.append(f"| {k} | {g('entropy')} | {g('corr_h')} | {g('corr_v')} | {g('corr_d')} | {g('chi2', 1)} | "
                         f"{g('npcr_p', 3)} | {g('uaci_p', 3)} | {g('npcr_p_fresh', 3)} | {g('uaci_p_fresh', 3)} | "
                         f"{g('npcr_k', 3)} | {g('uaci_k', 3)} | {g('png_bpp', 2)} | {g('t_enc', 1)} | {g('t_dec', 1)} | {g('wrongkey_psnr', 2)} |")
        lines.append(f"\n单个密文像素+1误差导致的错误明文像素数: {d.get('single_error_propagation_pixels')}\n")
        ec = d.get("error_curve", {})
        if ec:
            lines.append("| σ | " + " | ".join(k for k in ec if k != "sigma") + " |")
            lines.append("|---" * len(ec) + "|")
            for i, s in enumerate(ec["sigma"]):
                lines.append(f"| {s} | " + " | ".join(f"{ec[k][i]:.2f}" for k in ec if k != "sigma") + " |")
        lines.append("")

    lines.append("## 隐藏实验（评估均在8 bit含密图上，秘密图像指标为解密后）\n")
    lines.append("| run | 评估 | 密码 | 攻击 | PSNR_c | SSIM_c | PSNR_enc | PSNR_s | SSIM_s | APD_s | 载荷bpp | 错误密钥PSNR | 原文式(9)PSNR | 重采样上限 |")
    lines.append("|---" * 14 + "|")
    for ev in sorted(ROOT.glob("runs/*/eval_*.json")):
        d = json.load(open(ev))
        for key, s in d["summary"].items():
            cn, an = key.split("|")
            lines.append(f"| {ev.parent.name} | {ev.stem[5:]} | {cn} | {an} | {f(s['psnr_c'])} | {f(s['ssim_c'], 4)} | "
                         f"{f(s['psnr_enc'])} | {f(s['psnr_s'])} | {f(s['ssim_s'], 4)} | {f(s['apd_s'])} | {f(s['payload'])} | "
                         f"{f(s['psnr_wrong'])} | {f(s['psnr_literal']) if 'psnr_literal' in s else '—'} | {f(s['psnr_resample_ceiling']) if 'psnr_resample_ceiling' in s else '—'} |")
    lines.append("\n## 训练日志（最后一次验证）\n")
    lines.append("| run | iter | val PSNR_c | val PSNR_secret/cipher | 用时s |")
    lines.append("|---|---|---|---|---|")
    for lg in sorted(ROOT.glob("runs/*/log.csv")):
        rows = [r.strip().split(",") for r in open(lg) if r.strip()]
        if len(rows) > 1:
            r = rows[-1]
            lines.append(f"| {lg.parent.name} | {r[0]} | {r[7]} | {r[8]} | {r[9]} |")
    lines.append("\n## 深度隐写分析（SRNet）\n")
    lines.append("| 样本 | 测试对数 | 准确率 | AUC | P_E |")
    lines.append("|---|---|---|---|---|")
    for sj in sorted(ROOT.glob("stego/*/steganalysis_srnet.json")):
        d = json.load(open(sj))
        lines.append(f"| {sj.parent.name} | {d['test_pairs']} | {d['test_acc']:.4f} | {d['auc']:.4f} | {d['P_E']:.4f} |")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    if PACK.exists():
        shutil.rmtree(PACK)
    PACK.mkdir()
    for p in list(ROOT.glob("results/**/*")) + list(ROOT.glob("runs/*/*")) + list(ROOT.glob("stego/*/*.json")):
        if p.parts[0] == "runs" and not any(fnmatch.fnmatch(p.parts[1], q) for q in pats):
            continue
        if p.is_file() and p.suffix in {".json", ".csv", ".md", ".png", ".jpg", ".txt", ".log"}:
            if p.suffix == ".csv" and "per_image" in p.name and p.stat().st_size > 20e6:
                continue
            dst = PACK / p
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    shutil.make_archive("results_pack", "zip", PACK)
    print("\npacked -> results_pack.zip  (把这个文件发回来即可)")


if __name__ == "__main__":
    main()
