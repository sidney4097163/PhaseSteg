# -*- coding: utf-8 -*-
r"""
Center-crop to square, resize to 256x256 (as in ISGI), save PNG, and split train/test (disjoint).

  python prepare_data.py --src D:\datasets\coco\train2017 --out data\coco --n_train 10000 --n_test 2000
  python prepare_data.py --src D:\datasets\celeba\img_align_celeba --out data\celeba --n_train 10000 --n_test 2000
"""
import argparse
import random
from pathlib import Path

from PIL import Image

from common import list_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--min_side", type=int, default=128, help="skip images smaller than this")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--crops", type=int, default=0,
                    help=">0: take this many random size x size crops per image WITHOUT resizing (high-res sets such as DIV2K)")
    a = ap.parse_args()
    files = list_images(a.src)
    print(f"found {len(files)} images")
    random.Random(a.seed).shuffle(files)
    out = Path(a.out)
    need = {"train": a.n_train, "test": a.n_test}
    done = {"train": 0, "test": 0}
    for s in need:
        (out / s).mkdir(parents=True, exist_ok=True)
    it = iter(files)
    for split in ["test", "train"]:
        while done[split] < need[split]:
            try:
                f = next(it)
            except StopIteration:
                print("not enough images");
                break
            try:
                im = Image.open(f).convert("RGB")
            except Exception:
                continue
            w, h = im.size
            if a.crops > 0:   # random crops at native resolution
                if min(w, h) < a.size:
                    continue
                rng = random.Random(hash((str(f), a.seed)) & 0xffffffff)
                for _ in range(a.crops):
                    if done[split] >= need[split]:
                        break
                    x, y = rng.randint(0, w - a.size), rng.randint(0, h - a.size)
                    im.crop((x, y, x + a.size, y + a.size)).save(out / split / f"{done[split]:05d}.png")
                    done[split] += 1
                continue
            if min(w, h) < a.min_side:
                continue
            m = min(w, h)
            im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
            im = im.resize((a.size, a.size), Image.LANCZOS)
            im.save(out / split / f"{done[split]:05d}.png")
            done[split] += 1
            if done[split] % 1000 == 0:
                print(split, done[split], flush=True)
    print("done", done)


if __name__ == "__main__":
    main()
