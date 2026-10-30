# -*- coding: utf-8 -*-
"""
Networks.
  ISGINet   : re-implementation of the ISGI pre-processing + operational models (Fig.4 of the paper)
  HINN      : proposed Haar-wavelet invertible hiding network (affine coupling, dense sub-nets)
  ISGIDisc  : discriminator of ISGI (5 conv 3x3 + FC)
  SRMDisc   : proposed steganalysis-aware discriminator (fixed SRM high-pass front end + TLU)
  SRNet     : deep steganalyser used ONLY for security evaluation (Boroumand et al. 2019), colour input
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================ Haar DWT
def dwt(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1, x2 = x01[:, :, :, 0::2], x02[:, :, :, 0::2]
    x3, x4 = x01[:, :, :, 1::2], x02[:, :, :, 1::2]
    LL = x1 + x2 + x3 + x4
    HL = -x1 - x2 + x3 + x4
    LH = -x1 + x2 - x3 + x4
    HH = x1 - x2 - x3 + x4
    return torch.cat([LL, HL, LH, HH], 1)


def iwt(x):
    C = x.shape[1] // 4
    LL, HL, LH, HH = x[:, :C] / 2, x[:, C:2 * C] / 2, x[:, 2 * C:3 * C] / 2, x[:, 3 * C:] / 2
    a = LL - HL - LH + HH     # (0,0)
    b = LL - HL + LH - HH     # (1,0)
    c = LL + HL - LH - HH     # (0,1)
    d = LL + HL + LH + HH     # (1,1)
    top = torch.stack([a, c], -1).flatten(-2)          # row 0::2 : cols interleave a,c
    bot = torch.stack([b, d], -1).flatten(-2)          # row 1::2
    return torch.stack([top, bot], -2).flatten(-3, -2)


def lowpass(x):
    return dwt(x)[:, :x.shape[1]]


# ================================================================ ISGI network
def cbr(i, o, k=3, s=1, p=1, act="relu"):
    a = nn.ReLU(True) if act == "relu" else nn.LeakyReLU(0.2, True)
    return nn.Sequential(nn.Conv2d(i, o, k, s, p), nn.BatchNorm2d(o), a)


class Prep(nn.Module):
    """3 conv layers, 3x3, stride 2, padding 1 (ISGI Sec.4.2)."""

    def __init__(self, cin, chs):
        super().__init__()
        self.net = nn.Sequential(cbr(cin, chs[0], 3, 2, 1), cbr(chs[0], chs[1], 3, 2, 1), cbr(chs[1], chs[2], 3, 2, 1))

    def forward(self, x):
        return self.net(x)


class UpRes(nn.Module):
    """One residual up-stage of the ISGI operational model:
    main: ConvT 3x3 s2  -> conv(size-preserving) ; skip: ConvT 4x4 s2 ; added, BN + LeakyReLU."""

    def __init__(self, cin, cout):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, 3, 2, 1, output_padding=1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True),
            nn.Conv2d(cout, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True))
        self.skip = nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True))
        self.post = nn.Sequential(nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True))

    def forward(self, x):
        return self.post(self.main(x) + self.skip(x))


class Operational(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.Sequential(UpRes(cin, 64), UpRes(64, 32), UpRes(32, 16))
        self.out = nn.ConvTranspose2d(16, cout, 1, 1, 0)

    def forward(self, x):
        return self.out(self.up(x))


class ISGINet(nn.Module):
    """hide(cover, secret_rep) -> stego ; reveal(stego) -> secret_rep_hat"""

    def __init__(self, c_secret=3, out_kind="sigmoid"):
        super().__init__()
        self.prep_c = Prep(3, (16, 32, 64))
        self.prep_s = Prep(c_secret, (16, 32, 64))
        self.op_h = Operational(128, 3)
        self.prep_r = Prep(3, (32, 64, 128))
        self.op_r = Operational(128, c_secret)
        self.out_kind = out_kind

    def hide(self, cover, secret):
        f = torch.cat([self.prep_c(cover), self.prep_s(secret)], 1)
        return torch.sigmoid(self.op_h(f)), None

    def reveal(self, stego):
        y = self.op_r(self.prep_r(stego))
        return torch.sigmoid(y) if self.out_kind == "sigmoid" else torch.tanh(y)


# ================================================================ proposed HINN
class Dense(nn.Module):
    def __init__(self, cin, cout, g=32):
        super().__init__()
        self.c1 = nn.Conv2d(cin, g, 3, 1, 1)
        self.c2 = nn.Conv2d(cin + g, g, 3, 1, 1)
        self.c3 = nn.Conv2d(cin + 2 * g, g, 3, 1, 1)
        self.c4 = nn.Conv2d(cin + 3 * g, g, 3, 1, 1)
        self.c5 = nn.Conv2d(cin + 4 * g, cout, 3, 1, 1)
        nn.init.zeros_(self.c5.weight)
        nn.init.zeros_(self.c5.bias)
        self.a = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.a(self.c1(x))
        x2 = self.a(self.c2(torch.cat([x, x1], 1)))
        x3 = self.a(self.c3(torch.cat([x, x1, x2], 1)))
        x4 = self.a(self.c4(torch.cat([x, x1, x2, x3], 1)))
        return self.c5(torch.cat([x, x1, x2, x3, x4], 1))


class InvBlock(nn.Module):
    def __init__(self, c1, c2, g=32, clamp=2.0):
        super().__init__()
        self.phi = Dense(c2, c1, g)
        self.rho = Dense(c1, c2, g)
        self.eta = Dense(c1, c2, g)
        self.clamp = clamp

    def _s(self, y1):
        return self.clamp * (torch.sigmoid(self.rho(y1)) * 2 - 1)

    def forward(self, x1, x2):
        y1 = x1 + self.phi(x2)
        y2 = x2 * torch.exp(self._s(y1)) + self.eta(y1)
        return y1, y2

    def reverse(self, y1, y2):
        x2 = (y2 - self.eta(y1)) * torch.exp(-self._s(y1))
        x1 = y1 - self.phi(x2)
        return x1, x2


class HINN(nn.Module):
    """Haar-wavelet invertible hiding network.
    Forward  : (DWT(cover), DWT(secret)) -> K coupling blocks -> (y1, r);  stego = IWT(y1), r is discarded.
    Reverse  : (DWT(stego'), z=0)         -> K blocks reversed -> secret_hat = IWT(x2)."""

    def __init__(self, c_secret=6, blocks=8, g=32, out_kind="tanh"):
        super().__init__()
        self.cs = c_secret
        self.blocks = nn.ModuleList([InvBlock(12, 4 * c_secret, g) for _ in range(blocks)])
        self.out_kind = out_kind

    def hide(self, cover, secret):
        x1, x2 = dwt(cover), dwt(secret)
        for b in self.blocks:
            x1, x2 = b(x1, x2)
        return iwt(x1), x2

    def reveal(self, stego):
        y1 = dwt(stego)
        y2 = torch.zeros(y1.shape[0], 4 * self.cs, y1.shape[2], y1.shape[3], device=stego.device, dtype=stego.dtype)
        for b in reversed(self.blocks):
            y1, y2 = b.reverse(y1, y2)
        s = iwt(y2)
        return s.clamp(0, 1) if self.out_kind == "sigmoid" else s.clamp(-1.2, 1.2)


# ================================================================ discriminators
def _srm_bank():
    k = []
    # first order (h, v)
    k.append([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
    k.append([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
    # second order (h, v, d)
    k.append([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
    k.append([[0, 1, 0], [0, -2, 0], [0, 1, 0]])
    k.append([[1, 0, 0], [0, -2, 0], [0, 0, 1]])
    # SQUARE 3x3 / 4
    k.append([[-1 / 4, 2 / 4, -1 / 4], [2 / 4, -1, 2 / 4], [-1 / 4, 2 / 4, -1 / 4]])
    k3 = [torch.tensor(kk, dtype=torch.float32) for kk in k]
    k3 = [F.pad(t, (1, 1, 1, 1)) for t in k3]
    kv = torch.tensor([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0
    k3.append(kv)
    # edge 3x3
    e = torch.tensor([[-1, 2, -1], [2, -4, 2], [0, 0, 0]], dtype=torch.float32) / 4.0
    k3.append(F.pad(e, (1, 1, 1, 1)))
    return torch.stack(k3)  # (8,5,5)


class SRMDisc(nn.Module):
    """Steganalysis-aware discriminator: fixed SRM residual bank (8 filters x 3 colours, groups=3) + TLU(3)
    followed by a small CNN. Input in [0,1]."""

    def __init__(self):
        super().__init__()
        bank = _srm_bank()
        w = bank.unsqueeze(1).repeat(3, 1, 1, 1)          # (24,1,5,5), groups=3 -> 8 per colour
        self.register_buffer("w", w)

        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))
        self.body = nn.Sequential(blk(24, 32, 1), blk(32, 32, 2), blk(32, 64, 2), blk(64, 128, 2), blk(128, 256, 2))
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        r = F.conv2d(x * 255.0, self.w, padding=2, groups=3).clamp(-3, 3)
        f = self.body(r)
        return self.fc(f.mean((2, 3))).squeeze(1)


class ISGIDisc(nn.Module):
    """ISGI discriminator (Fig.4): conv3x3 s2 with 3,8,16,32,1 filters + fully connected."""

    def __init__(self):
        super().__init__()
        chs = [3, 3, 8, 16, 32, 1]
        layers = []
        for i in range(5):
            layers += [nn.Conv2d(chs[i], chs[i + 1], 3, 2, 1), nn.BatchNorm2d(chs[i + 1]), nn.LeakyReLU(0.2, True)]
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(8)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc(self.pool(self.body(x)).flatten(1)).squeeze(1)


# ================================================================ SRNet (evaluation only)
class _T1(nn.Sequential):
    def __init__(s, i, o):
        super().__init__(nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class _T2(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.b = nn.Sequential(_T1(c, c), nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))

    def forward(s, x):
        return x + s.b(x)


class _T3(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.b = nn.Sequential(_T1(i, o), nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.AvgPool2d(3, 2, 1))
        s.sc = nn.Sequential(nn.Conv2d(i, o, 1, 2, bias=False), nn.BatchNorm2d(o))

    def forward(s, x):
        return s.b(x) + s.sc(x)


class SRNet(nn.Module):
    def __init__(self, cin=3):
        super().__init__()
        self.f = nn.Sequential(_T1(cin, 64), _T1(64, 16), *[_T2(16) for _ in range(5)],
                               _T3(16, 16), _T3(16, 64), _T3(64, 128), _T3(128, 256),
                               _T1(256, 512), nn.Conv2d(512, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512))
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.fc(self.f(x).mean((2, 3)))


# ================================================================ Baluja-style baseline
class Branch(nn.Module):
    """Parallel 3x3 and 5x5 convolutions (50 filters each) + ReLU, concatenated (Baluja 2017 style)."""

    def __init__(self, cin, f=50):
        super().__init__()
        self.a = nn.Conv2d(cin, f, 3, 1, 1)
        self.b = nn.Conv2d(cin, f, 5, 1, 2)

    def forward(self, x):
        return F.relu(torch.cat([self.a(x), self.b(x)], 1))


class BalujaNet(nn.Module):
    """Prep (2 layers) + hide (5 layers) + reveal (5 layers), full resolution, no down-sampling.
    Used with an XOR stream cipher it reproduces the encrypt-then-hide pipeline of Duan et al. (2020)/DL-Steg."""

    def __init__(self, c_secret=3, out_kind="sigmoid", f=50):
        super().__init__()
        w = 2 * f
        self.prep = nn.Sequential(Branch(c_secret, f), Branch(w, f))
        self.hid = nn.Sequential(Branch(w + 3, f), *[Branch(w, f) for _ in range(4)])
        self.hid_out = nn.Conv2d(w, 3, 1)
        self.rev = nn.Sequential(Branch(3, f), *[Branch(w, f) for _ in range(4)])
        self.rev_out = nn.Conv2d(w, c_secret, 1)
        self.out_kind = out_kind

    def hide(self, cover, secret):
        h = self.hid(torch.cat([cover, self.prep(secret)], 1))
        return torch.sigmoid(self.hid_out(h)), None

    def reveal(self, stego):
        y = self.rev_out(self.rev(stego))
        return torch.sigmoid(y) if self.out_kind == "sigmoid" else torch.tanh(y)


# ================================================================ factory
def build(arch, rep, blocks=8):
    cs = 6 if rep == "phase" else 3
    kind = "tanh" if rep == "phase" else "sigmoid"
    if arch == "isgi":
        return ISGINet(cs, kind)
    if arch == "hinn":
        return HINN(cs, blocks=blocks, out_kind=kind)
    if arch == "baluja":
        return BalujaNet(cs, kind)
    raise ValueError(arch)


def build_disc(name):
    return {"srm": SRMDisc, "isgi": ISGIDisc}[name]() if name != "none" else None


if __name__ == "__main__":
    x = torch.rand(2, 3, 64, 64)
    print("IWT(DWT(x)) max err:", (iwt(dwt(x)) - x).abs().max().item())
    for arch in ["isgi", "hinn", "baluja"]:
        for rep in ["lin", "phase"]:
            m = build(arch, rep)
            s = torch.rand(2, 6 if rep == "phase" else 3, 64, 64)
            st, _ = m.hide(x, s)
            r = m.reveal(st)
            n = sum(p.numel() for p in m.parameters()) / 1e6
            print(f"{arch:5s} {rep:5s} stego {tuple(st.shape)} rev {tuple(r.shape)} params {n:.2f}M")
    blk = InvBlock(12, 24)
    for p in blk.parameters():
        nn.init.normal_(p, std=0.02)
    a, b = torch.randn(1, 12, 8, 8), torch.randn(1, 24, 8, 8)
    y1, y2 = blk(a, b)
    a2, b2 = blk.reverse(y1, y2)
    print("InvBlock invertibility err:", max((a - a2).abs().max().item(), (b - b2).abs().max().item()))
    for D in [SRMDisc(), ISGIDisc()]:
        print(type(D).__name__, D(x).shape, sum(p.numel() for p in D.parameters()))
    print("SRNet", SRNet()(torch.rand(2, 3, 256, 256)).shape)
