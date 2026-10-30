# PhaseSteg

Code for the paper 

PhaseSteg encrypts a secret image with **CPCE** (circular-phase chaotic encryption: nonce-driven multi-lane PWLCM keystream, global permutation, modular addition, guard band, no ciphertext feedback), represents the ciphertext as phases on the unit circle, hides it with 2×2 cipher-domain redundancy in a Haar-wavelet invertible network (HINN), and decrypts the real-valued extracted ciphertext without error amplification.

The repository also contains a re-implementation of **ISGI** (Hashemi & Majidi, MTAP 2026) and its cipher **IEPS**, a Baluja-style baseline, a SHAKE-256 control cipher, and all evaluation scripts.

## Requirements

Python 3.10, PyTorch ≥ 2.0 (CUDA recommended), plus

```
pip install -r requirements.txt
```

All experiments in the paper were run on a single NVIDIA RTX A2000 (12 GB) under Windows 10.

## Data

| Dataset | Use | Command |
|---|---|---|
| COCO train2017 | train + test | `python prepare_data.py --src <coco>/train2017 --out data/coco --n_train 10000 --n_test 2000` |
| CelebA (img_align_celeba) | train + test | `python prepare_data.py --src <celeba>/img_align_celeba --out data/celeba --n_train 10000 --n_test 2000` |
| DIV2K valid HR | test only | `python prepare_data.py --src <div2k>/DIV2K_valid_HR --out data/div2k --n_train 0 --n_test 400 --crops 4` |

Images are centre-cropped and resized to 256×256 (DIV2K: random 256×256 crops at native resolution).

## Quick start

```bash
python crypto_chaos.py          # self-test of IEPS / CPCE
python models.py                # self-test of the networks

# proposed model (HINN, phase representation, 2x2 redundancy, no discriminator)
python train.py --name ours --data data/coco/train --val data/coco/test \
                --arch hinn --rep phase --cscale 2 --disc none

# evaluate on 1000 pairs with CPCE / CPCE(g=0) / IEPS; stego is rounded to 8 bits, metrics after decryption
python evaluate.py --run runs/ours --test data/coco/test --n_pairs 1000 --ciphers cpce,cpce0,ieps
```

## Reproducing the paper

`run_all.bat` (Windows; the commands are plain Python calls and port directly to a shell script) runs everything in four stages:

| Stage | `STAGE=` | Content | Paper |
|---|---|---|---|
| A | (empty) | ISGI re-implementation, HINN phase s=1/2, linear s=2, ISGI s=2 | Table 2 |
| B | `B` | references without encryption, ablations, CelebA, robustness, SRNet | Tables 3, 5, 7, 8 |
| C | `C` | discriminator-free versions of all comparisons (final configuration) | Tables 2, 3, 5, 6, 7 |
| D | `D` | extra seeds, SHAKE-256 control, Baluja baseline, DIV2K | Tables 1, 2, 3, 4 |

`python collect_results.py` writes `results/summary.md` with every number used in the paper.
`python make_figures.py --fig 1|2|3` produces Figs. 1–3 (PDF/EPS/SVG/TIFF).

## Files

| File | Content |
|---|---|
| `crypto_chaos.py` | PWLCM, IEPS (incl. Eq. (9) of ISGI as printed), CPCE, SHAKE-256 controls |
| `models.py` | ISGI networks, HINN, Baluja-style network, ISGI / SRM discriminators, SRNet |
| `common.py` | data loading, phase representation, quantization (STE), noise layers, metrics |
| `train.py` / `evaluate.py` | training / evaluation under the 8-bit, decrypted-domain protocol |
| `cipher_analysis.py` | entropy, correlation, NPCR/UACI, error-propagation and error-tolerance curves |
| `steganalysis.py` | SRNet detection with paired 60/10/30 split |
| `prepare_data.py`, `collect_results.py`, `make_figures.py` | data, result tables, figures |

## Notes

* The network is trained on uniform random bytes in place of ciphertext (`--train_cipher random`, default); the real cipher is applied at test time. `--train_cipher cpce` trains on real CPCE ciphertexts.
* The key in `crypto_chaos.py` is a demo key. For real use, supply your own 256-bit key and **never reuse a nonce** with the same key.
* CPCE can be instantiated with any secure keystream generator; `ShakeStream(mode="add")` gives identical results.

## Citation

If you use this code, please cite the paper (citation will be added after publication).

## License

MIT
