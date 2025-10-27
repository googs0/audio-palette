#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json, struct, wave
from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import colorsys

"""
Audio to color palette

24 bit WAV files to complementary color palettes using 1/3-octave band
levels and A/C/Z weighting. 

Exports:
- Labeled PNG swatch
- JSON with hex codes + base HSV
- Adobe ASE swatch file

You can also save a band-level plot
""" 

# 1/3-octave bands
THIRD_OCT_CENTERS = np.array([
    25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
    12500, 16000, 20000
], dtype=float)

EDGE_FACTOR = 2 ** (1/6)

def third_octave_band_edges(centers: np.ndarray) -> np.ndarray:
    return np.vstack([centers/EDGE_FACTOR, centers*EDGE_FACTOR]).T

# Weightings
# A-weighting
def a_weighting_db(f: np.ndarray) -> np.ndarray:
    """IEC 61672 A-weighting (0 dB ~ 1 kHz)."""
    f = np.asarray(f, dtype=float)
    f2 = f*f
    ra_num = (12194**2) * (f2**2)
    ra_den = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2)
    ra = ra_num / (ra_den + 1e-30)
    return 20*np.log10(ra) + 2.0

# C-weighting
def c_weighting_db(f: np.ndarray) -> np.ndarray:
    """IEC 61672 C-weighting (0 dB ~ 1 kHz)."""
    f = np.asarray(f, dtype=float)
    f2 = f*f
    rc_num = (12194**2) * f2
    rc_den = (f2 + 20.6**2) * (f2 + 12194**2)
    rc = rc_num / (rc_den + 1e-30)
    return 20*np.log10(rc) + 0.06

# Weighting curve
def weighting_curve_db(f: np.ndarray, mode: str = "A") -> np.ndarray:
    m = (mode or "A").upper()
    if m == "A":
        return a_weighting_db(f)
    if m == "C":
        return c_weighting_db(f)
    # Z/flat
    return np.zeros_like(f, dtype=float)  

# WAV I/O 
def load_wav_24bit(path: str) -> Tuple[np.ndarray, int]:
    # Load a 24-bit PCM WAV
    with wave.open(path, "rb") as wf:
        nchan = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        nframes = wf.getnframes()
        comptype = wf.getcomptype()
        if sampwidth != 2 or comptype != "NONE":
            raise ValueError("Only uncompressed 24-bit PCM WAV is supported.")
        raw = wf.readframes(nframes)
    x = np.frombuffer(raw, dtype=np.int24).astype(np.float32) / 32768.0
    if nchan > 1:
        x = x.reshape(-1, nchan).mean(axis=1)
    return x, fr

# Spectrum & Bands
def band_levels_third_octave(
    signal: np.ndarray,
    fs: int,
    weighting: str = "A",
    centers: np.ndarray = THIRD_OCT_CENTERS
) -> Tuple[np.ndarray, np.ndarray]:
    # Compute 1/3-octave band energies with A/C/Z weighting; return (levels_dB, centers_used)
    x = np.asarray(signal, dtype=np.float32)
    if np.abs(x.mean()) > 1e-9:
        x = x - x.mean()

    n = int(2 ** np.ceil(np.log2(len(x))))
    X = np.fft.rfft(x, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    # Uncalibrated
    psd = (np.abs(X) ** 2) / n

    edges = third_octave_band_edges(centers)
    # Nyquist guard
    valid = (edges[:, 1] < (fs/2.0))
    edges = edges[valid]
    c = centers[valid]

    w_db = weighting_curve_db(c, weighting)
    w_lin = 10 ** (w_db / 10.0)

    band_power = np.zeros(len(c), dtype=float)
    for i, (fl, fu) in enumerate(edges):
        mask = (freqs >= fl) & (freqs < fu)
        band_power[i] = psd[mask].sum() * w_lin[i]

    levels_db = 10.0 * np.log10(band_power + 1e-20)
    if np.isfinite(levels_db).any():
        # Normalize to 0 dB peak
        levels_db -= levels_db.max()  
    return levels_db, c

# Color Mapping
@dataclass
class Palette:
    # (H, S, V)
    base_hsv: Tuple[float, float, float]    
    # hex swatches
    colors_hex: List[str] 
    # names for each swatch
    labels: List[str]                         

def _hsv_to_hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb((h % 360)/360.0, float(s), float(v))
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

def _spec_flatness(levels_db: np.ndarray) -> float:
    lin = 10 ** (levels_db / 10.0)
    gmean = np.exp(np.mean(np.log(lin + 1e-20)))
    amean = np.mean(lin + 1e-20)
    return float(gmean / amean)

def _spectral_centroid_log(centers: np.ndarray, levels_db: np.ndarray) -> float:
    lin = 10 ** (levels_db / 10.0)
    weights = lin / (lin.sum() + 1e-20)
    log_f = np.log(centers + 1e-20)
    return float(np.exp(np.sum(weights * log_f)))

def _slope_vs_logf(centers: np.ndarray, levels_db: np.ndarray) -> float:
    x = np.log10(centers)
    y = levels_db
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return float(np.polyfit(x, y, 1)[0])

def palette_from_bands(
    centers: np.ndarray,
    levels_db: np.ndarray,
    base_mode: str = "centroid",
) -> Palette:
    fmin, fmax = float(centers.min()), float(centers.max())
    if base_mode == "peak":
        base_f = float(centers[np.argmax(levels_db)])
    else:
        base_f = _spectral_centroid_log(centers, levels_db)

    t = (np.log(base_f) - np.log(fmin)) / (np.log(fmax) - np.log(fmin) + 1e-20)
    base_h = float((t * 360.0) % 360.0)

    sfm = _spec_flatness(levels_db)
    sat = float(np.clip(1.0 - sfm, 0.25, 0.95))

    L = levels_db
    spread = (L.max() - L.min())
    spiky = (L.max() - np.median(L)) / (spread + 1e-12)
    val = float(np.clip(0.65 + 0.3 * spiky, 0.55, 0.98))

    comp = (base_h + 180.0) % 360.0
    analog_delta = 30.0
    analogs = [ (base_h - analog_delta) % 360.0, (base_h + analog_delta) % 360.0 ]

    slope = _slope_vs_logf(centers, levels_db)
    accent_shift = float(np.clip(slope * 45.0, -45.0, 45.0))
    accent_h = (base_h + accent_shift + 360.0) % 360.0

    hues = [base_h, comp] + analogs + [accent_h]
    labels = ["Base", "Complement", "Analog −", "Analog +", "Tilt Accent"]
    colors_hex = [_hsv_to_hex(h, sat, val) for h in hues]
    return Palette(base_hsv=(base_h, sat, val), colors_hex=colors_hex, labels=labels)

# Rendering/Export
def save_palette_image(palette: Palette, path: str, width: int = 1400, height: int = 240) -> None:
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.set_axis_off()
    n = len(palette.colors_hex)
    for i, (hexcol, label) in enumerate(zip(palette.colors_hex, palette.labels)):
        x0 = i / n
        x1 = (i + 1) / n
        ax.add_patch(plt.Rectangle((x0, 0), x1 - x0, 0.7, color=hexcol))
        ax.text((x0 + x1)/2, 0.85, f"{label}\n{hexcol}", ha="center", va="center", fontsize=14)
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def save_palette_json(palette: Palette, path: str) -> None:
    data = {
        "base_hsv": {"h": palette.base_hsv[0], "s": palette.base_hsv[1], "v": palette.base_hsv[2]},
        "swatches": [{"label": l, "hex": h} for l, h in zip(palette.labels, palette.colors_hex)],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def save_palette_ase(palette: Palette, path: str) -> None:
    """Write a minimal Adobe ASE file with RGB colors and UTF-16BE names."""
    def be_u16(x): return struct.pack(">H", x)
    def be_u32(x): return struct.pack(">I", x)
    def be_f32(x): return struct.pack(">f", float(x))

    blocks = []
    for label, hexcol in zip(palette.labels, palette.colors_hex):
        name_u = (label + "\x00").encode("utf-16-be")
        r = int(hexcol[1:3], 16) / 255.0
        g = int(hexcol[3:5], 16) / 255.0
        b = int(hexcol[5:7], 16) / 255.0
        content = b"".join([
            be_u16(len(label) + 1),  # UTF-16 code units including NUL
            name_u,
            b"RGB ",
            be_f32(r), be_f32(g), be_f32(b),
            be_u16(0)  # global color
        ])
        block = b"".join([be_u16(0x0001), be_u32(len(content)), content])
        blocks.append(block)

    header = b"ASEF" + be_u16(1) + be_u16(0) + be_u32(len(blocks))
    with open(path, "wb") as f:
        f.write(header + b"".join(blocks))

# API
def audio_to_palette(signal: np.ndarray, fs: int, weighting: str = "A", base_mode: str = "centroid"):
    levels_db, centers = band_levels_third_octave(signal, fs, weighting=weighting)
    pal = palette_from_bands(centers, levels_db, base_mode=base_mode)
    return pal, levels_db, centers

def wav_to_palette(path: str, weighting: str = "A", base_mode: str = "centroid"):
    x, fs = load_wav_16bit(path)
    return audio_to_palette(x, fs, weighting=weighting, base_mode=base_mode)

# CLI
def _sine_mix(fs: int, dur: float, freqs: list, amps: Optional[list] = None) -> np.ndarray:
    t = np.arange(int(fs*dur)) / fs
    amps = amps or [1.0]*len(freqs)
    sig = sum(a*np.sin(2*np.pi*f*t) for f, a in zip(freqs, amps))
    sig /= max(np.max(np.abs(sig)), 1e-9)
    # 10 ms cosine fade
    k = max(1, int(0.01*fs))
    w = 0.5*(1-np.cos(np.linspace(0, np.pi, k)))
    sig[:k] *= w
    sig[-k:] *= w[::-1]
    return sig.astype(np.float32)

def main(argv=None):
    p = argparse.ArgumentParser(description="Audio → Complementary Color Palette (1/3-octave, A/C/Z weighting)")
    p.add_argument("input", nargs="?", help="Path to 16-bit PCM WAV file (omit with --demo)")
    p.add_argument("--weighting", choices=["A","C","Z"], default="A", help="A/C/Z (flat) weighting")
    p.add_argument("--base-mode", choices=["centroid","peak"], default="centroid", help="Base hue from centroid or peak band")
    p.add_argument("--out", default="out", help="Output directory")
    p.add_argument("--prefix", default="palette", help="Output filename prefix")
    p.add_argument("--plot-bands", action="store_true", help="Also save a band-level plot (PNG)")
    p.add_argument("--no-json", action="store_true", help="Do not write JSON")
    p.add_argument("--no-ase", action="store_true", help="Do not write ASE file")
    p.add_argument("--width", type=int, default=1400, help="Palette image width px")
    p.add_argument("--height", type=int, default=240, help="Palette image height px")
    p.add_argument("--demo", action="store_true", help="Use a built-in bass+treble synth signal")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    if args.demo:
        fs = 44100
        x = _sine_mix(fs, 2.5, [80, 3000], [0.9, 0.6])
        pal, levels_db, centers = audio_to_palette(x, fs, weighting=args.weighting, base_mode=args.base_mode)
    else:
        if not args.input:
            p.error("Provide an input WAV or use --demo.")
        pal, levels_db, centers = wav_to_palette(args.input, weighting=args.weighting, base_mode=args.base_mode)

    png = os.path.join(args.out, f"{args.prefix}_{args.weighting}.png")
    save_palette_image(pal, png, width=args.width, height=args.height)

    if not args.no_json:
        jsn = os.path.join(args.out, f"{args.prefix}_{args.weighting}.json")
        save_palette_json(pal, jsn)

    if not args.no_ase:
        ase = os.path.join(args.out, f"{args.prefix}_{args.weighting}.ase")
        save_palette_ase(pal, ase)

    if args.plot_bands:
        fig = plt.figure(figsize=(10, 4), dpi=120)
        ax = plt.gca()
        ax.plot(centers, levels_db, label=f"{args.weighting}-weighted")
        ax.set_xscale("log")
        ax.set_xlabel("1/3-octave center frequency (Hz)")
        ax.set_ylabel("Relative level (dB)")
        ax.grid(True, which="both", ls=":")
        ax.legend()
        band_png = os.path.join(args.out, f"{args.prefix}_{args.weighting}_bands.png")
        fig.tight_layout()
        fig.savefig(band_png, bbox_inches="tight")
        plt.close(fig)

    print("Wrote:")
    print(" ", png)
    if not args.no_json:
        print(" ", jsn)
    if not args.no_ase:
        print(" ", ase)
    if args.plot_bands:
        print(" ", band_png)

if __name__ == "__main__":
    main()
