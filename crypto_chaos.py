# -*- coding: utf-8 -*-
"""
Chaotic image ciphers used in the experiments.

  IEPS  : faithful re-implementation of the cipher in ISGI (Hashemi & Majidi, MTAP 2026):
          PWLCM row/column permutation + 16x16 block / 8x8 sub-block substitution with
          SHA-512 chaining on the *previous cipher sub-block* (ciphertext feedback).
          decrypt()          -> mathematically correct inverse
          decrypt_literal()  -> Eq.(9)(10) exactly as printed in the paper (H - SB'), which is not an inverse

  CPCE  : proposed Circular-Phase Chaotic Encryption.
          nonce-based multi-lane PWLCM keystream, global pixel permutation + modular addition,
          NO ciphertext feedback (error stays local), guard-band plaintext mapping so that
          modular wrap-around of the recovered ciphertext can be resolved unambiguously.
          Decryption accepts a real-valued (network-recovered) ciphertext.

All images are uint8 arrays of shape (H, W, C).
"""
import hashlib
import numpy as np

# ---------------------------------------------------------------- PWLCM
_EPS = 1e-12


def pwlcm_step(x, p):
    """One PWLCM iteration, vectorised (x, p arrays or floats)."""
    y = np.where(x >= 0.5, 1.0 - x, x)
    out = np.where(y < p, y / p, (y - p) / (0.5 - p))
    bad = (out <= _EPS) | (out >= 1.0 - _EPS)          # escape fixed points 0 / 1 caused by finite precision
    if np.any(bad):
        out = np.where(bad, (x * 0.7310585786300049 + 0.1234567) % 1.0, out)
    return out


def pwlcm_scalar(x0, p, n, burn=200):
    """Single-orbit PWLCM sequence (used by IEPS, as in the original paper)."""
    x = float(x0)
    out = np.empty(n, dtype=np.float64)
    for _ in range(burn):
        y = 1.0 - x if x >= 0.5 else x
        x = y / p if y < p else (y - p) / (0.5 - p)
        if x <= _EPS or x >= 1 - _EPS:
            x = (x * 0.7310585786300049 + 0.1234567) % 1.0
    for i in range(n):
        y = 1.0 - x if x >= 0.5 else x
        x = y / p if y < p else (y - p) / (0.5 - p)
        if x <= _EPS or x >= 1 - _EPS:
            x = (x * 0.7310585786300049 + 0.1234567) % 1.0
        out[i] = x
    return out


def _lane_params(seed: bytes, lanes: int):
    x0 = np.empty(lanes)
    p = np.empty(lanes)
    for i in range(lanes):
        h = hashlib.sha512(seed + i.to_bytes(4, "big")).digest()
        x0[i] = (int.from_bytes(h[0:7], "big") + 0.5) / 2.0 ** 56
        p[i] = 0.05 + 0.40 * (int.from_bytes(h[7:14], "big") / 2.0 ** 56)
    return x0, p


def pwlcm_multilane(seed: bytes, n: int, lanes: int = 256, burn: int = 200):
    """L parallel PWLCM orbits whose (x0, p) are derived from SHA-512(seed || lane).
    Vectorised over lanes -> fast enough for per-image (per-nonce) keystreams."""
    x, p = _lane_params(seed, lanes)
    for _ in range(burn):
        x = pwlcm_step(x, p)
    steps = -(-n // lanes)
    out = np.empty((steps, lanes))
    for t in range(steps):
        x = pwlcm_step(x, p)
        out[t] = x
    return out.reshape(-1)[:n]


def to_bytes_stream(x):
    return (np.floor(x * 1e14) % 256).astype(np.uint8)


# ---------------------------------------------------------------- IEPS (ISGI)
class IEPS:
    """Re-implementation of IEPS (Sec. 4.1, Algorithms 1-4 of ISGI)."""

    def __init__(self, subkey1=(0.3456789012345, 0.2718281828), subkey2=(0.6789012345678, 0.3141592653),
                 subkey3=b"ISGI-IEPS-subkey3", rounds=2):
        self.k1, self.k2, self.k3, self.rounds = subkey1, subkey2, subkey3, rounds
        self._cache = {}
        self.hkey = np.frombuffer(hashlib.sha512(self.k3).digest(), dtype=np.uint8).reshape(8, 8)

    # -- permutation (Algorithm 1)
    def _perm_keys(self, H, W):
        if (H, W) not in self._cache:
            K = pwlcm_scalar(self.k1[0], self.k1[1], H * W).reshape(H, W)   # M sequences of length N
            O = pwlcm_scalar(self.k2[0], self.k2[1], W * H).reshape(W, H)   # N sequences of length M
            k = np.argsort(-K, axis=1, kind="stable")                       # "arguments of reverse-ordered"
            o = np.argsort(-O, axis=1, kind="stable")
            self._cache[(H, W)] = (k, o)
        return self._cache[(H, W)]

    def permute(self, img, inverse=False):
        H, W = img.shape[:2]
        k, o = self._perm_keys(H, W)
        rows = np.arange(H)[:, None]
        cols = np.arange(W)[None, :]
        out = np.empty_like(img)
        if not inverse:
            out[rows, k] = img                       # row i: pixel j -> position k[i, j]
            img2 = out
            out = np.empty_like(img)
            out[o.T, cols] = img2                    # column j: pixel i -> position o[j, i]
        else:
            tmp = img[o.T, cols]                     # undo column permutation
            out = tmp[rows, k]                       # undo row permutation
        return out

    # -- substitution (Algorithm 2)
    @staticmethod
    def _to_sub(img):
        H, W, C = img.shape
        b = img.reshape(H // 16, 16, W // 16, 16, C).transpose(0, 2, 4, 1, 3)      # (bh, bw, C, 16, 16)
        return [b[..., 0:8, 0:8], b[..., 0:8, 8:16], b[..., 8:16, 0:8], b[..., 8:16, 8:16]]

    @staticmethod
    def _from_sub(subs, shape):
        H, W, C = shape
        bh, bw = H // 16, W // 16
        b = np.empty((bh, bw, C, 16, 16), dtype=np.uint8)
        b[..., 0:8, 0:8], b[..., 0:8, 8:16], b[..., 8:16, 0:8], b[..., 8:16, 8:16] = subs
        return b.transpose(0, 3, 1, 4, 2).reshape(H, W, C)

    @staticmethod
    def _hash_blocks(sb):
        """SHA-512 of every 8x8 sub-block -> (bh, bw, C, 8, 8) uint8."""
        flat = np.ascontiguousarray(sb).reshape(-1, 64)
        out = np.empty_like(flat)
        for i in range(flat.shape[0]):
            out[i] = np.frombuffer(hashlib.sha512(flat[i].tobytes()).digest(), dtype=np.uint8)
        return out.reshape(sb.shape)

    def substitute(self, img):
        sb = [s.astype(np.int32) for s in self._to_sub(img)]
        enc = [None] * 4
        enc[0] = ((sb[0] + self.hkey) % 256).astype(np.uint8)
        for m in range(1, 4):
            enc[m] = ((sb[m] + self._hash_blocks(enc[m - 1])) % 256).astype(np.uint8)
        return self._from_sub(enc, img.shape)

    def inv_substitute(self, img, literal=False):
        sb = [s.astype(np.int32) for s in self._to_sub(img)]
        hs = [np.broadcast_to(self.hkey, sb[0].shape).astype(np.int32)] + \
             [self._hash_blocks(self._to_sub(img)[m - 1]).astype(np.int32) for m in range(1, 4)]
        if literal:   # Eq.(9)(10) as printed: SB = (H - SB') mod 256
            dec = [((hs[m] - sb[m]) % 256).astype(np.uint8) for m in range(4)]
        else:         # correct inverse: SB = (SB' - H) mod 256
            dec = [((sb[m] - hs[m]) % 256).astype(np.uint8) for m in range(4)]
        return self._from_sub(dec, img.shape)

    # -- Algorithms 3 / 4
    def encrypt(self, img, nonce=None):
        x = img.copy()
        for _ in range(self.rounds):
            x = self.permute(x)
            x = self.substitute(x)
        return x

    def decrypt(self, c, nonce=None, literal=False):
        """c may be real-valued (network output); IEPS needs bytes, so it is rounded first."""
        x = np.clip(np.rint(c), 0, 255).astype(np.uint8)
        for _ in range(self.rounds):
            x = self.inv_substitute(x, literal=literal)
            x = self.permute(x, inverse=True)
        return x

    def decrypt_literal(self, c, nonce=None):
        return self.decrypt(c, literal=True)

    def wrong_key(self):
        """Same cipher with subkey1.x0 changed in its last mantissa bit (1-bit key change)."""
        x0 = np.nextafter(self.k1[0], 1.0)
        return IEPS((x0, self.k1[1]), self.k2, self.k3, self.rounds)


# ---------------------------------------------------------------- CPCE (proposed)
class CPCE:
    """Circular-Phase Chaotic Encryption.

    enc:  v = G(p)                      guard-band mapping  [0,255] -> [g, 255-g]
          c[j] = (v[pi(j)] + k[j]) mod 256   pi, k from multi-lane PWLCM seeded by SHA-512(K || nonce)
    dec:  d[j] = (c_hat[j] - k[j]) mod 256, v[pi(j)] = d[j]
          v   <- circular unwrap around 127.5, clip to [g, 255-g],  p = G^{-1}(v)

    Error property: with circular distance |.|_o on Z_256, |D(c+e) - p|_o = |e|_o pixel-wise
    (error is only relocated, never amplified); with guard g, any |e| <= g is resolved exactly.
    """

    def __init__(self, key: bytes = b"PhaseSteg-master-key-256bit-demo", guard: int = 16, lanes: int = 256):
        self.key, self.g, self.lanes = key, int(guard), lanes

    def _streams(self, nonce, n):
        if isinstance(nonce, str):
            nonce = nonce.encode()
        if isinstance(nonce, int):
            nonce = nonce.to_bytes(16, "big")
        seed = hashlib.sha512(self.key + b"|" + nonce + b"|" + n.to_bytes(8, "big")).digest()
        perm = np.argsort(pwlcm_multilane(seed + b"P", n, self.lanes), kind="stable")
        ks = to_bytes_stream(pwlcm_multilane(seed + b"K", n, self.lanes)).astype(np.float64)
        return perm, ks

    def _G(self, p):
        g = self.g
        return np.rint(g + p.astype(np.float64) * (255 - 2 * g) / 255.0) if g > 0 else p.astype(np.float64)

    def _Ginv(self, v):
        g = self.g
        w = ((v - 127.5 + 128.0) % 256.0) - 128.0 + 127.5          # circular unwrap: [-0.5, 255.5)
        if g > 0:
            w = np.clip(w, g, 255 - g)
            return (w - g) * 255.0 / (255 - 2 * g)
        return np.clip(w, 0, 255)

    def encrypt(self, img, nonce):
        shp = img.shape
        perm, ks = self._streams(nonce, img.size)
        v = self._G(img).reshape(-1)
        c = (v[perm] + ks) % 256
        return c.reshape(shp).astype(np.uint8)

    def decrypt(self, c_hat, nonce):
        """c_hat: real-valued recovered ciphertext (any real numbers, interpreted mod 256)."""
        shp = c_hat.shape
        perm, ks = self._streams(nonce, c_hat.size)
        d = (np.asarray(c_hat, dtype=np.float64).reshape(-1) - ks) % 256.0
        v = np.empty_like(d)
        v[perm] = d
        p = self._Ginv(v)
        return np.clip(np.rint(p), 0, 255).astype(np.uint8).reshape(shp)

    def wrong_key(self):
        kb = bytearray(self.key)
        kb[-1] ^= 0x01
        return CPCE(bytes(kb), self.g, self.lanes)


class ShakeStream:
    """Standard-primitive control for CPCE: identical structure (nonce, global permutation, keystream,
    no ciphertext feedback) but the permutation and keystream come from SHAKE-256 instead of PWLCM.
      mode="add": modular addition + guard band  (CPCE structure with a standard XOF)
      mode="xor": bitwise XOR, as in DL-Steg / ECC-XOR schemes (no guard band; recovered ciphertext is
                  rounded to a byte before XOR, so a +-1 error can flip several bits)."""

    def __init__(self, key: bytes = b"PhaseSteg-master-key-256bit-demo", mode="add", guard=16):
        self.key, self.mode, self.g = key, mode, (guard if mode == "add" else 0)
        self._cp = CPCE(key, guard=self.g)          # reuse guard mapping / unwrap

    def _streams(self, nonce, n):
        if isinstance(nonce, int):
            nonce = nonce.to_bytes(16, "big")
        elif isinstance(nonce, str):
            nonce = nonce.encode()
        x = hashlib.shake_256(self.key + b"|" + nonce + b"|" + n.to_bytes(8, "big")).digest(n * 9)
        r = np.frombuffer(x[:n * 8], dtype=">u8")
        perm = np.argsort(r, kind="stable")
        ks = np.frombuffer(x[n * 8:], dtype=np.uint8)
        return perm, ks

    def encrypt(self, img, nonce):
        shp = img.shape
        perm, ks = self._streams(nonce, img.size)
        if self.mode == "xor":
            return (img.reshape(-1)[perm] ^ ks).reshape(shp)
        v = self._cp._G(img).reshape(-1)
        return ((v[perm] + ks) % 256).reshape(shp).astype(np.uint8)

    def decrypt(self, c_hat, nonce):
        shp = np.shape(c_hat)
        n = int(np.prod(shp))
        perm, ks = self._streams(nonce, n)
        c = np.asarray(c_hat, dtype=np.float64).reshape(-1)
        out = np.empty(n)
        if self.mode == "xor":
            cb = (np.rint(c).astype(np.int64) % 256).astype(np.uint8)
            out[perm] = cb ^ ks
            return out.reshape(shp).astype(np.uint8)
        d = (c - ks) % 256.0
        out[perm] = d
        p = self._cp._Ginv(out)
        return np.clip(np.rint(p), 0, 255).astype(np.uint8).reshape(shp)

    def wrong_key(self):
        kb = bytearray(self.key)
        kb[-1] ^= 0x01
        return ShakeStream(bytes(kb), self.mode, self.g if self.mode == "add" else 16)


class NoCipher:
    def encrypt(self, img, nonce=None):
        return img.copy()

    def decrypt(self, c_hat, nonce=None):
        return np.clip(np.rint(c_hat), 0, 255).astype(np.uint8)

    def wrong_key(self):
        return self


def build_cipher(name, rounds=2, guard=16):
    if name == "none":
        return NoCipher()
    if name == "ieps":
        return IEPS(rounds=rounds)
    if name == "cpce":
        return CPCE(guard=guard)
    if name == "shake":
        return ShakeStream(mode="add", guard=guard)
    if name == "xor":
        return ShakeStream(mode="xor")
    if name.startswith("cpce") and name[4:].isdigit():   # cpce0, cpce8, cpce32 ... (guard-band width)
        return CPCE(guard=int(name[4:]))
    raise ValueError(name)


# ---------------------------------------------------------------- self test
if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:256, 0:256]
    img = np.stack([(xx + yy) % 256, (xx * 2) % 256, (255 - yy)], -1).astype(np.uint8)
    img = np.clip(img.astype(int) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)

    ie = IEPS()
    t = time.time(); c = ie.encrypt(img); te = time.time() - t
    t = time.time(); d = ie.decrypt(c); td = time.time() - t
    print(f"IEPS round-trip exact: {np.array_equal(d, img)}   enc {te*1e3:.0f} ms  dec {td*1e3:.0f} ms")
    print("IEPS literal Eq.(9)(10) round-trip exact:", np.array_equal(ie.decrypt_literal(c), img))
    noisy = c.astype(int); noisy[100, 100, 0] = (noisy[100, 100, 0] + 1) % 256
    dd = ie.decrypt(noisy)
    print("IEPS: 1-level error on ONE cipher pixel -> wrong plaintext pixels:", int((dd != img).sum()))

    cp = CPCE(guard=16)
    t = time.time(); c2 = cp.encrypt(img, nonce=1); te = time.time() - t
    t = time.time(); d2 = cp.decrypt(c2.astype(float), nonce=1); td = time.time() - t
    print(f"CPCE round-trip max err: {np.abs(d2.astype(int) - img).max()} (guard quantisation)   enc {te*1e3:.0f} ms  dec {td*1e3:.0f} ms")
    e = rng.integers(-16, 17, c2.shape)
    d3 = cp.decrypt(c2.astype(float) + e, nonce=1)
    v_ref = cp.decrypt(c2.astype(float), nonce=1).astype(float)
    print("CPCE |error| <= 16 levels -> max plaintext err:",
          np.abs(d3.astype(float) - v_ref).max(), "(<= 16*255/223 + rounding ~ 19.3 expected)")
    cp0 = CPCE(guard=0)
    c0 = cp0.encrypt(img, 1)
    print("CPCE(g=0) exact:", np.array_equal(cp0.decrypt(c0.astype(float), 1), img))
    print("CPCE nonce 1 vs 2 differ fraction:", (cp.encrypt(img, 1) != cp.encrypt(img, 2)).mean())
