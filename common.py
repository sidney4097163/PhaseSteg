# -*- coding: utf-8 -*-
"""Shared utilities: data, ciphertext representations, noise layers, metrics."""
import io
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

TWO_PI = 2 * math.pi
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------------------------------------------------------- data
def list_images(folder):
    return sorted([p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT])


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


class CropDataset(torch.utils.data.Dataset):
    """Random crops (with flips) from prepared 256x256 PNGs. Returns float tensor in [0,1]."""

    def __init__(self, folder, crop=128, length=None):
        self.files = list_images(folder)
        if not self.files:
            raise RuntimeError(f"no images in {folder}")
        self.crop = crop
        self.length = length or len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        a = load_rgb(self.files[random.randrange(len(self.files))])
        H, W = a.shape[:2]
        c = self.crop
        y, x = random.randint(0, H - c), random.randint(0, W - c)
        a = a[y:y + c, x:x + c]
        if random.random() < 0.5:
            a = a[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0


def make_pairs(test_dir, n_pairs=None):
    """Deterministic cover/secret pairing on the test split: cover=i, secret=(i+N/2) mod N."""
    files = list_images(test_dir)
    N = len(files)
    n = N if n_pairs is None else min(n_pairs, N)
    return [(files[i], files[(i + N // 2) % N]) for i in range(n)]


# ---------------------------------------------------------------- representations
def encode_rep(c_uint8, rep):
    """c: tensor (B,3,H,W) with integer values 0..255 (float). -> network input."""
    if rep in ("lin", "plain"):
        return c_uint8 / 255.0
    if rep == "phase":
        th = TWO_PI * c_uint8 / 256.0
        return torch.cat([torch.cos(th), torch.sin(th)], 1)
    raise ValueError(rep)


def decode_rep(r, rep):
    """network output -> real-valued ciphertext estimate in [0,256) (phase) or [0,255] (lin)."""
    if rep in ("lin", "plain"):
        return r.clamp(0, 1) * 255.0
    C = r.shape[1] // 2
    th = torch.atan2(r[:, C:], r[:, :C])
    return torch.remainder(th, TWO_PI) * 256.0 / TWO_PI


def expand_rep(r_small, s):
    """cipher-domain redundancy: each ciphertext symbol is repeated over an s x s block."""
    return r_small if s == 1 else F.interpolate(r_small, scale_factor=s, mode="nearest")


def pool_rep(r_hat, s):
    """receiver side: average the s x s copies (vector mean on the unit circle for phase)."""
    return r_hat if s == 1 else F.avg_pool2d(r_hat, s)


def rec_loss(r_hat, r, rep):
    """Reconstruction loss. Phase loss is divided by (2*pi)^2 so that for small errors it equals the
    linear MSE in [0,1]-normalised grey levels (comparable weights between lin and phase)."""
    if rep in ("lin", "plain"):
        return F.mse_loss(r_hat, r)
    C = r.shape[1] // 2
    d = (r_hat - r) ** 2
    return (d[:, :C] + d[:, C:]).mean() / (TWO_PI ** 2)


def circ_err(a, b):
    """circular difference on Z_256 (works on tensors or numpy)."""
    d = (a - b) % 256.0
    return np.minimum(d, 256.0 - d) if isinstance(d, np.ndarray) else torch.minimum(d, 256.0 - d)


# ---------------------------------------------------------------- noise layers
def quantize_ste(x):
    return x + (torch.round(x * 255.0) / 255.0 - x).detach()


def jpeg_np(img_uint8, q):
    buf = io.BytesIO()
    Image.fromarray(img_uint8).save(buf, format="JPEG", quality=int(q))
    return np.asarray(Image.open(buf).convert("RGB"))


def jpeg_ste(x, q):
    with torch.no_grad():
        a = (x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        out = np.stack([jpeg_np(im, q) for im in a])
        y = torch.from_numpy(out).to(x.device).permute(0, 3, 1, 2).float() / 255.0
    return x + (y - x).detach()


def channel(x, quant=True, noise=False):
    """Training-time attack layer. Always clamps; optional 8-bit quantisation (STE); optional random
    distortion (identity / Gaussian / JPEG) for the robust variant."""
    x = x.clamp(0, 1)
    if noise:
        r = random.random()
        if r < 1 / 3:
            pass
        elif r < 2 / 3:
            x = x + torch.randn_like(x) * (random.uniform(0.5, 2.0) / 255.0)
        else:
            x = jpeg_ste(x, random.randint(85, 95))
        x = x.clamp(0, 1)
    if quant:
        x = quantize_ste(x)
    return x


# ---------------------------------------------------------------- metrics (numpy, uint8 HxWxC)
def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def psnr_mse(mse):
    return 100.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def apd(a, b):
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _gauss(win=11, sigma=1.5):
    g = np.exp(-((np.arange(win) - win // 2) ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


_G = torch.tensor(_gauss(), dtype=torch.float64)[None, None]


def ssim(a, b):
    """Mean SSIM over RGB channels (Wang et al. 2004, Gaussian window 11, sigma 1.5)."""
    x = torch.from_numpy(a.astype(np.float64)).permute(2, 0, 1)[:, None]
    y = torch.from_numpy(b.astype(np.float64)).permute(2, 0, 1)[:, None]
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = F.conv2d(x, _G), F.conv2d(y, _G)
    sxx = F.conv2d(x * x, _G) - mx ** 2
    syy = F.conv2d(y * y, _G) - my ** 2
    sxy = F.conv2d(x * y, _G) - mx * my
    m = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return float(m.mean())


def payload_bpp(s, e):
    """ISGI Eq.(31): (1 - mean|S-E|/255) * 8 * 3."""
    return (1 - np.mean(np.abs(s.astype(np.float64) - e.astype(np.float64))) / 255.0) * 24.0


def to_t(a, device):
    return torch.from_numpy(a).permute(2, 0, 1)[None].float().to(device)


def to_np(t):
    return t[0].detach().cpu().permute(1, 2, 0).numpy()


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
