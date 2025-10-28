#!/usr/bin/env python3

from __future__ import annotations
import argparse
import colorsys
import json
import os
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

"""
Audio to color palette

Create complementary color palettes from audio using 1/3-octave band analysis
with A, C, or Z (flat) weighting. Works with most audio files.

Requirements:
  - numpy
  - matplotlib
  - librosa (recommended) or soundfile
"""

# CONSTANTS
EPS = 1e-20

# ISO/IEC nominal 1/3-octave centers (25 Hz - 20 kHz)
THIRD_OCT_CENTER_HZ = np.array(
    [
        25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
        800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
        10000, 12500, 16000, 20000,
    ],
    dtype=float,
)
THIRD_OCT_EDGE_FACTOR = 2 ** (1 / 6)

# Palette mapping design constants
HUE_ANALOG_SPREAD_DEG = 30.0
TILT_MAX_SHIFT_DEG = 45.0
MIN_SAT = 0.25
MAX_SAT = 0.95
MIN_VAL = 0.55
MAX_VAL = 0.98


# PATH / OUTPUT HELPERS 
def resolve_out_dir(p: str) -> str:
    """Expand ~, make absolute, create dir if missing, and return path."""
    out = os.path.abspath(os.path.expanduser(p))
    os.makedirs(out, exist_ok=True)
    return out


# AUDIO I/O 
def _to_float32_mono(y: np.ndarray, mono: bool = True) -> np.ndarray:
    # Ensure float32 mono in [-1, 1]
    y = np.asarray(y)
    if y.ndim > 1 and mono:
        y = y.mean(axis=1)
    if y.dtype != np.float32:
        y = y.astype(np.float32, copy=False)
    return y

def load_audio(
    path: str,
    mono: bool = True,
    # "auto" | "librosa" | "soundfile"
    prefer: str = "auto", 
) -> Tuple[np.ndarray, int]:
    # Load audio --> (float32 mono signal, sample_rate)
    # Tries librosa first, then soundfile
    err = None
    if prefer in ("auto", "librosa"):
        try:
            import librosa  # type: ignore
            y, sr = librosa.load(path, sr=None, mono=mono)
            return _to_float32_mono(y, mono=True), int(sr)
        except Exception as e:
            err = e
    if prefer in ("auto", "soundfile", "librosa"):
        try:
            import soundfile as sf  # type: ignore
            y, sr = sf.read(path, dtype="float32", always_2d=False)
            return _to_float32_mono(y, mono=True), int(sr)
        except Exception as e:
            err = e
    raise RuntimeError(
        "Audio load failed. Install one of:\n"
        "  pip install librosa\n"
        "  or\n"
        "  pip install soundfile"
    ) from err

# 1/3-OCTAVE BAND UTILITIES
def third_octave_band_edges(centers_hz: np.ndarray) -> np.ndarray:
    # Return [N x 2] array of lower/upper edges for each 1/3-octave band
    return np.vstack(
        [centers_hz / THIRD_OCT_EDGE_FACTOR, centers_hz * THIRD_OCT_EDGE_FACTOR]
    ).T

# WEIGHTINGS
def a_weighting_db(freq_hz: np.ndarray) -> np.ndarray:
    """IEC 61672 A-weighting (≈0 dB at 1 kHz)."""
    f2 = np.asarray(freq_hz, float) ** 2
    num = (12194.0**2) * (f2**2)
    den = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2)
    ra = num / (den + EPS)
    return 20.0 * np.log10(ra + EPS) + 2.0

def c_weighting_db(freq_hz: np.ndarray) -> np.ndarray:
    """IEC 61672 C-weighting (≈0 dB at 1 kHz)."""
    f2 = np.asarray(freq_hz, float) ** 2
    num = (12194.0**2) * f2
    den = (f2 + 20.6**2) * (f2 + 12194.0**2)
    rc = num / (den + EPS)
    return 20.0 * np.log10(rc + EPS) + 0.06

def get_weighting_db(freq_hz: np.ndarray, mode: str = "A") -> np.ndarray:
    """Return weighting curve in dB for the given mode ('A'|'C'|'Z')."""
    mode = (mode or "A").upper()
    weighting_map = {
        "A": a_weighting_db,
        "C": c_weighting_db,
        # Z = flat
        "Z": lambda f: np.zeros_like(np.asarray(f, float)),
    }
    return weighting_map.get(mode, weighting_map["A"])(freq_hz)

# SPECTRUM & BAND ANALYSIS
def compute_third_octave_levels(
    signal: np.ndarray,
    sample_rate: int,
    weighting: str = "A",
    band_centers_hz: np.ndarray = THIRD_OCT_CENTER_HZ,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (band_levels_db, band_centers_used_hz) for 1/3-octave bands with selected weighting.
    Levels are relative (normalized to 0 dB peak).
    """
    x = np.asarray(signal, np.float32)
    if x.size == 0:
        raise ValueError("Empty audio buffer.")
    mean = float(x.mean())
    if abs(mean) > 1e-9:
        x = x - mean

    # FFT
    n = int(2 ** np.ceil(np.log2(len(x))))
    spectrum = np.fft.rfft(x, n=n)
    freqs_hz = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    psd = (np.abs(spectrum) ** 2) / max(n, 1)

    # Band edges within Nyquist
    edges = third_octave_band_edges(band_centers_hz)
    valid = edges[:, 1] < (sample_rate / 2.0)
    edges = edges[valid]
    centers_used_hz = band_centers_hz[valid]

    # Vectorized bin to band assignment by upper-edge search
    upper_edges = edges[:, 1]
    band_idx = np.searchsorted(upper_edges, freqs_hz, side="left")
    in_range = (band_idx < len(edges)) & (freqs_hz >= edges[np.clip(band_idx, 0, len(edges)-1), 0])

    # Sum PSD per band with bincount
    power_per_band = np.bincount(
        band_idx[in_range],
        weights=psd[in_range],
        minlength=len(edges)
    )

    # Apply weighting per band center
    w_db = get_weighting_db(centers_used_hz, weighting)
    w_lin = 10.0 ** (w_db / 10.0)
    band_power = power_per_band * w_lin

    band_levels_db = 10.0 * np.log10(band_power + EPS)
    band_levels_db -= np.max(band_levels_db)  # normalize to 0 dB peak
    return band_levels_db, centers_used_hz

# COLOR MAPPING 
@dataclass
class Palette:
    # (H°, S, V)
    base_hsv: Tuple[float, float, float]
    swatches_hex: List[str]
    swatch_labels: List[str]

def _hsv_to_hex(h_deg: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb((h_deg % 360.0) / 360.0, float(s), float(v))
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

def _spectral_flatness(levels_db: np.ndarray) -> float:
    lin = 10.0 ** (levels_db / 10.0)
    gmean = np.exp(np.mean(np.log(lin + EPS)))
    amean = np.mean(lin + EPS)
    return float(gmean / amean)

def _log_spectral_centroid_hz(centers_hz: np.ndarray, levels_db: np.ndarray) -> float:
    lin = 10.0 ** (levels_db / 10.0)
    w = lin / (lin.sum() + EPS)
    return float(np.exp(np.sum(w * np.log(centers_hz + EPS))))

def _slope_vs_log_frequency(centers_hz: np.ndarray, levels_db: np.ndarray) -> float:
    x = np.log10(centers_hz)
    y = levels_db
    x = (x - x.mean()) / (x.std() + EPS)
    y = (y - y.mean()) / (y.std() + EPS)
    return float(np.polyfit(x, y, 1)[0])

def build_palette_from_bands(
    band_centers_hz: np.ndarray,
    band_levels_db: np.ndarray,
    # "centroid" | "peak"
    base_selector: str = "centroid",
) -> Palette:
    fmin, fmax = float(band_centers_hz.min()), float(band_centers_hz.max())
    if base_selector == "peak":
        base_f_hz = float(band_centers_hz[np.argmax(band_levels_db)])
    else:
        base_f_hz = _log_spectral_centroid_hz(band_centers_hz, band_levels_db)

    # Map log frequency to hue (0 - 360)
    t = (np.log(base_f_hz) - np.log(fmin)) / (np.log(fmax) - np.log(fmin) + EPS)
    base_h_deg = float((t * 360.0) % 360.0)

    # Saturation from spectral flatness (noisy to less saturated)
    sat = float(np.clip(1.0 - _spectral_flatness(band_levels_db), MIN_SAT, MAX_SAT))

    # Value from spiky-ness
    spread = float(band_levels_db.max() - band_levels_db.min())
    spiky = (float(band_levels_db.max()) - float(np.median(band_levels_db))) / (spread + EPS)
    val = float(np.clip(0.65 + 0.3 * spiky, MIN_VAL, MAX_VAL))

    # Complement + analogs + tilt accent
    complement_h = (base_h_deg + 180.0) % 360.0
    analogs = [
        (base_h_deg - HUE_ANALOG_SPREAD_DEG) % 360.0,
        (base_h_deg + HUE_ANALOG_SPREAD_DEG) % 360.0,
    ]
    tilt = _slope_vs_log_frequency(band_centers_hz, band_levels_db)
    accent_shift = float(np.clip(tilt * TILT_MAX_SHIFT_DEG, -TILT_MAX_SHIFT_DEG, TILT_MAX_SHIFT_DEG))
    accent_h = (base_h_deg + accent_shift + 360.0) % 360.0

    hues = [base_h_deg, complement_h] + analogs + [accent_h]
    labels = ["Base", "Complement", "Analog −", "Analog +", "Tilt Accent"]
    swatches_hex = [_hsv_to_hex(h, sat, val) for h in hues]
    return Palette(base_hsv=(base_h_deg, sat, val), swatches_hex=swatches_hex, swatch_labels=labels)

# EXPORTS
def export_palette_png(
    palette: Palette, path: str, width_px: int = 1400, height_px: int = 240
) -> None:
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax.set_axis_off()
    n = len(palette.swatches_hex)
    for i, (hexcol, label) in enumerate(zip(palette.swatches_hex, palette.swatch_labels)):
        x0, x1 = i / n, (i + 1) / n
        ax.add_patch(plt.Rectangle((x0, 0), x1 - x0, 0.7, color=hexcol))
        ax.text((x0 + x1) / 2, 0.85, f"{label}\n{hexcol}", ha="center", va="center", fontsize=14)
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def export_palette_json(palette: Palette, path: str) -> None:
    payload = {
        "base_hsv": {"h": palette.base_hsv[0], "s": palette.base_hsv[1], "v": palette.base_hsv[2]},
        "swatches": [{"label": l, "hex": h} for l, h in zip(palette.swatch_labels, palette.swatches_hex)],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def export_palette_ase(palette: Palette, path: str) -> None:
    # Write Adobe ASE file with RGB colors and UTF-16BE names
    def be_u16(x): return struct.pack(">H", x)
    def be_u32(x): return struct.pack(">I", x)
    def be_f32(x): return struct.pack(">f", float(x))

    blocks = []
    for label, hexcol in zip(palette.swatch_labels, palette.swatches_hex):
        name_u = (label + "\x00").encode("utf-16-be")
        r = int(hexcol[1:3], 16) / 255.0
        g = int(hexcol[3:5], 16) / 255.0
        b = int(hexcol[5:7], 16) / 255.0
        content = b"".join([be_u16(len(label) + 1), name_u, b"RGB ", be_f32(r), be_f32(g), be_f32(b), be_u16(0)])
        blocks.append(b"".join([be_u16(0x0001), be_u32(len(content)), content]))

    header = b"ASEF" + be_u16(1) + be_u16(0) + be_u32(len(blocks))
    with open(path, "wb") as f:
        f.write(header + b"".join(blocks))

# API
def palette_from_signal(
    signal: np.ndarray, sample_rate: int, weighting: str = "A", base_selector: str = "centroid"
):
    band_levels_db, band_centers_hz = compute_third_octave_levels(signal, sample_rate, weighting)
    palette = build_palette_from_bands(band_centers_hz, band_levels_db, base_selector)
    return palette, band_levels_db, band_centers_hz

def palette_from_file(
    path: str,
    weighting: str = "A",
    base_selector: str = "centroid",
    loader: str = "auto",
):
    y, sr = load_audio(path, mono=True, prefer=loader)
    return palette_from_signal(y, sr, weighting=weighting, base_selector=base_selector)

# CLI
def synth_tone_mix(fs: int, dur: float, freqs: List[float], amps: Optional[List[float]] = None) -> np.ndarray:
    """Simple sine mix generator with 10 ms cosine fades."""
    t = np.arange(int(fs * dur)) / fs
    amps = amps or [1.0] * len(freqs)
    sig = sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps))
    sig /= max(np.max(np.abs(sig)), 1e-9)
    k = max(1, int(0.01 * fs))  # 10 ms fade
    w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    sig[:k] *= w
    sig[-k:] *= w[::-1]
    return sig.astype(np.float32)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate complementary color palettes from audio.")
    parser.add_argument("input", nargs="?", help="Audio file path (WAV/MP3/etc). Omit with --demo.")
    parser.add_argument("--weighting", choices=["A", "C", "Z"], default="A", help="A, C, or flat (Z).")
    parser.add_argument("--base-mode", choices=["centroid", "peak"], default="centroid", help="Base hue from centroid or peak band.")
    parser.add_argument("--loader", choices=["auto", "librosa", "soundfile"], default="auto", help="Audio loader preference.")
    parser.add_argument("--out", default="out", help="Output directory.")
    parser.add_argument("--prefix", default="palette", help="Output filename prefix.")
    parser.add_argument("--plot-bands", action="store_true", help="Also save a 1/3-octave band plot (PNG).")
    parser.add_argument("--no-json", action="store_true", help="Skip JSON export.")
    parser.add_argument("--no-ase", action="store_true", help="Skip ASE export.")
    parser.add_argument("--width", type=int, default=1400, help="Palette image width (px).")
    parser.add_argument("--height", type=int, default=240, help="Palette image height (px).")
    parser.add_argument("--demo", action="store_true", help="Use a built-in bass+treble signal.")
    args = parser.parse_args(argv)

    # Normalize output directory and optionally auto-derive prefix from input
    out_dir = resolve_out_dir(args.out)
    if not args.demo and args.prefix == "palette" and args.input:
        args.prefix = os.path.splitext(os.path.basename(args.input))[0]

    if args.demo:
        fs = 44100
        sig = synth_tone_mix(fs, 2.5, [80, 3000], [0.9, 0.6])
        palette, band_levels_db, band_centers_hz = palette_from_signal(sig, fs, args.weighting, args.base_mode)
    else:
        if not args.input:
            parser.error("Provide an input file or use --demo.")
        palette, band_levels_db, band_centers_hz = palette_from_file(
            args.input, weighting=args.weighting, base_selector=args.base_mode, loader=args.loader
        )

    outputs = []

    png_path = os.path.join(out_dir, f"{args.prefix}_{args.weighting}.png")
    export_palette_png(palette, png_path, width_px=args.width, height_px=args.height)
    outputs.append(png_path)

    if not args.no_json:
        json_path = os.path.join(out_dir, f"{args.prefix}_{args.weighting}.json")
        export_palette_json(palette, json_path)
        outputs.append(json_path)

    if not args.no_ase:
        ase_path = os.path.join(out_dir, f"{args.prefix}_{args.weighting}.ase")
        export_palette_ase(palette, ase_path)
        outputs.append(ase_path)

    if args.plot_bands:
        fig = plt.figure(figsize=(10, 4), dpi=120)
        ax = plt.gca()
        ax.plot(band_centers_hz, band_levels_db, label=f"{args.weighting}-weighted")
        ax.set_xscale("log")
        ax.set_xlabel("1/3-octave center frequency (Hz)")
        ax.set_ylabel("Relative level (dB)")
        ax.grid(True, which="both", ls=":")
        ax.legend()
        band_png = os.path.join(out_dir, f"{args.prefix}_{args.weighting}_bands.png")
        fig.tight_layout()
        fig.savefig(band_png, bbox_inches="tight")
        plt.close(fig)
        outputs.append(band_png)

    print("Wrote:")
    for p in outputs:
        print(" ", p)

if __name__ == "__main__":
    main()
