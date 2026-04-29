from pathlib import Path
import re
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

# Reference: https://www.baeldung.com/cs/rgb-color-light-frequency

# Input folder produced by decompose.py
FREQ_CSV_DIR = Path(
    "/Users/lxl/Desktop/Imaging Instrumentation/Segmentation/segmented/frequency_stacks/csv"
)

# Outputs
OUT_RGB_CSV = FREQ_CSV_DIR.parent / "rgb_stitched.csv"
OUT_RGB_IMAGE = FREQ_CSV_DIR.parent / "rgb_stitched.png"
SHOW_IMAGE = False
INTENSITY_MAX = 255.0
GAMMA = 0.8
GAIN_EPS = 1e-12
BALANCE_CYCLES = 3


def parse_wavelength_from_filename(path):
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)nm$", path.stem)
    if not match:
        raise ValueError(f"Could not parse wavelength from filename: {path.name}")
    return float(match.group(1))


def wavelength_to_rgb(wl):
    """
    Piecewise conversion from wavelength (nm) to RGB in [0, 1].
    """
    if 645 <= wl <= 780:
        r, g, b = 1.0, 0.0, 0.0
    elif 580 <= wl < 645:
        r = 1.0
        g = -(wl - 645.0) / (645.0 - 580.0)
        b = 0.0
    elif 510 <= wl < 580:
        r = (wl - 510.0) / (580.0 - 510.0)
        g = 1.0
        b = 0.0
    elif 490 <= wl < 510:
        r = 0.0
        g = 1.0
        b = -(wl - 510.0) / (510.0 - 490.0)
    elif 440 <= wl < 490:
        r = 0.0
        g = (wl - 440.0) / (490.0 - 440.0)
        b = 1.0
    elif 380 <= wl < 440:
        r = -(wl - 440.0) / (440.0 - 380.0)
        g = 0.0
        b = 1.0
    else:
        r, g, b = 0.0, 0.0, 0.0

    # Intensity falloff near human vision limits.
    if 380 <= wl < 420:
        factor = 0.3 + 0.7 * (wl - 380.0) / (420.0 - 380.0)
    elif 420 <= wl < 701:
        factor = 1.0
    elif 701 <= wl < 781:
        factor = 0.3 + 0.7 * (780.0 - wl) / (780.0 - 700.0)
    else:
        factor = 0.0

    # Gamma/intensity-adjusted RGB then back to [0,1] for downstream blending.
    r8 = 0.0 if r == 0.0 else round(INTENSITY_MAX * ((r * factor) ** GAMMA))
    g8 = 0.0 if g == 0.0 else round(INTENSITY_MAX * ((g * factor) ** GAMMA))
    b8 = 0.0 if b == 0.0 else round(INTENSITY_MAX * ((b * factor) ** GAMMA))

    return np.array([r8, g8, b8], dtype=np.float64) / INTENSITY_MAX


def load_frequency_csv(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    image_idx = data["image_idx"].astype(np.int32)
    pixel_idx = data["pixel_idx"].astype(np.int32)
    intensity = data["intensity_milli"].astype(np.float64) / 1000.0
    return image_idx, pixel_idx, intensity


def normalize_rgb(rgb, valid_mask=None):
    """
    Shared-scale normalization to [0,255] uint8.
    Uses one common high-percentile scale (no per-channel offset) to preserve RGB ratios.
    """
    if valid_mask is not None and np.any(valid_mask):
        vals = rgb[valid_mask]
    else:
        vals = rgb.reshape(-1, 3)

    hi = np.percentile(vals, 99.5)
    if hi <= 0:
        return np.zeros_like(rgb, dtype=np.uint8)
    out = np.clip(rgb / hi, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def intensity_to_value_map(weight_sum, valid_mask):
    """
    Convert per-pixel spectral power to HSV V target in [0,1] while preserving
    relative intensity ordering (with robust clipping for outliers).
    """
    if valid_mask is not None and np.any(valid_mask):
        vals = weight_sum[valid_mask]
    else:
        vals = weight_sum.reshape(-1)

    lo = float(np.percentile(vals, 1.0))
    hi = float(np.percentile(vals, 99.5))
    if hi <= lo:
        return np.zeros_like(weight_sum, dtype=np.float64)

    v_map = np.clip((weight_sum - lo) / (hi - lo), 0.0, 1.0)
    return v_map


def enforce_value_channel(rgb, value_map):
    """
    Convert RGB->HSV, replace V by value_map, then back to RGB.
    """
    rgb01 = np.clip(rgb, 0.0, 1.0)
    hsv = mcolors.rgb_to_hsv(rgb01)
    hsv[..., 2] = value_map
    return mcolors.hsv_to_rgb(hsv)


def main():
    csv_paths = sorted(FREQ_CSV_DIR.glob("*nm.csv"), key=parse_wavelength_from_filename)
    if not csv_paths:
        raise FileNotFoundError(f"No wavelength CSV files found in: {FREQ_CSV_DIR}")

    # Infer output shape from first file.
    i0, p0, _ = load_frequency_csv(csv_paths[0])
    num_images = int(np.max(i0)) + 1
    width = int(np.max(p0)) + 1

    rgb_accum = np.zeros((num_images, width, 3), dtype=np.float64)
    weight_sum = np.zeros((num_images, width), dtype=np.float64)

    for csv_path in csv_paths:
        wl = parse_wavelength_from_filename(csv_path)
        rgb = wavelength_to_rgb(wl)

        image_idx, pixel_idx, intensity = load_frequency_csv(csv_path)

        # Weighted accumulation by intensity per pixel.
        rgb_accum[image_idx, pixel_idx, :] += intensity[:, None] * rgb[None, :]
        weight_sum[image_idx, pixel_idx] += intensity

    # Weighted-average spectral color (chromaticity prior).
    denom = np.where(weight_sum > 1e-12, weight_sum, 1.0)
    rgb_color = rgb_accum / denom[:, :, None]

    valid = weight_sum > 1e-12
    value_target = intensity_to_value_map(weight_sum, valid)

    # Iterative coupling:
    # 1) enforce intensity via HSV V
    # 2) apply global RGB mean balancing
    rgb_work = np.clip(rgb_color, 0.0, 1.0)
    gains = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    ch_mean = np.mean(rgb_work[valid], axis=0) if np.any(valid) else np.mean(rgb_work.reshape(-1, 3), axis=0)
    for _ in range(BALANCE_CYCLES):
        rgb_work = enforce_value_channel(rgb_work, value_target)
        ch_mean = np.mean(rgb_work[valid], axis=0) if np.any(valid) else np.mean(rgb_work.reshape(-1, 3), axis=0)
        target = float(np.mean(ch_mean))
        gains = target / np.maximum(ch_mean, GAIN_EPS)
        rgb_work = np.clip(rgb_work * gains[None, None, :], 0.0, 1.0)

    # Final intensity enforcement after the last color balance step.
    rgb_final = enforce_value_channel(rgb_work, value_target)
    rgb_u8 = (np.clip(rgb_final, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Save one compact stitched RGB CSV.
    img_idx = np.repeat(np.arange(num_images, dtype=np.int32), width)
    pix_idx = np.tile(np.arange(width, dtype=np.int32), num_images)
    flat = rgb_u8.reshape(-1, 3)
    out = np.column_stack([img_idx, pix_idx, flat[:, 0], flat[:, 1], flat[:, 2]])
    np.savetxt(
        OUT_RGB_CSV,
        out,
        delimiter=",",
        fmt=["%d", "%d", "%d", "%d", "%d"],
        header="image_idx,pixel_idx,r,g,b",
        comments="",
    )

    # Display + save image.
    plt.figure(figsize=(15, 15))
    plt.imshow(rgb_u8, aspect="auto")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_RGB_IMAGE, dpi=300, bbox_inches="tight", pad_inches=0)
    if SHOW_IMAGE:
        plt.show()
    plt.close()

    print(f"RGB volume shape: {rgb_u8.shape}")
    ch_mean_after = np.mean(rgb_final[valid], axis=0) if np.any(valid) else np.mean(rgb_final.reshape(-1, 3), axis=0)
    ch_u8_mean = np.mean(rgb_u8.reshape(-1, 3), axis=0)
    v_before = np.mean(value_target[valid]) if np.any(valid) else np.mean(value_target)
    v_after = np.mean(mcolors.rgb_to_hsv(np.clip(rgb_final, 0.0, 1.0))[..., 2][valid]) if np.any(valid) else np.mean(
        mcolors.rgb_to_hsv(np.clip(rgb_final, 0.0, 1.0))[..., 2]
    )
    print(f"Channel means before gain: {ch_mean}")
    print(f"Channel means after gain: {ch_mean_after}")
    print(f"Applied gains (R,G,B): {gains}")
    print(f"Channel means in saved uint8 (R,G,B): {ch_u8_mean}")
    print(f"Mean V target / final (valid pixels): {v_before:.6f} / {v_after:.6f}")
    print(f"Saved stitched RGB CSV: {OUT_RGB_CSV}")
    print(f"Saved stitched RGB image: {OUT_RGB_IMAGE}")


if __name__ == "__main__":
    main()
