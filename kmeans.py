import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ===== User-editable config =====
INPUT_PATH = "segmented/frequency_stacks/csv"
SELECTED_FREQUENCIES = [440, 570, 620]
USE_SELECTED_FREQUENCIES = True
K_CLUSTERS = 4
VISUALIZE_NORMALIZED_FREQUENCIES = False
NORMALIZED_FIGURE_DIR = "kmeans_normalized_frequency_figures"
USE_SPATIAL_FEATURES = True
SPATIAL_WEIGHT_X = 0.4
SPATIAL_WEIGHT_Y = 0.4
# ===============================


def wavelength_from_name(path: Path) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)nm", path.stem)
    return float(match.group(1))


def load_frequency_csvs(csv_dir: Path, selected_frequencies=None):
    csv_paths = sorted(csv_dir.glob("*.csv"), key=wavelength_from_name)
    if selected_frequencies:
        selected_set = {float(x) for x in selected_frequencies}
        csv_paths = [p for p in csv_paths if float(wavelength_from_name(p)) in selected_set]
        if not csv_paths:
            raise ValueError("No CSV files matched SELECTED_FREQUENCIES.")

    wavelengths = np.array([wavelength_from_name(p) for p in csv_paths], dtype=float)
    first = np.genfromtxt(csv_paths[0], delimiter=",", names=True, dtype=np.int64)

    max_img = int(np.max(first["image_idx"]))
    max_px = int(np.max(first["pixel_idx"]))
    num_images = max_img + 1
    xvalues = max_px + 1
    num_freq = len(csv_paths)

    cube = np.zeros((num_images, num_freq, xvalues), dtype=np.float32)

    for fi, csv_path in enumerate(csv_paths):
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.int64)
        img_idx = np.asarray(data["image_idx"], dtype=np.int64)
        px_idx = np.asarray(data["pixel_idx"], dtype=np.int64)
        intensity_milli = np.asarray(data["intensity_milli"], dtype=np.float32)

        cube[img_idx, fi, px_idx] = intensity_milli / 1000.0

    return cube, wavelengths


def cube_to_2d(cube: np.ndarray) -> np.ndarray:
    num_images, num_freq, xvalues = cube.shape
    matrix_2d = np.transpose(cube, (0, 2, 1)).reshape(num_images * xvalues, num_freq)
    return matrix_2d


def build_xy_features(num_images: int, xvalues: int) -> np.ndarray:
    y_idx = np.repeat(np.arange(num_images, dtype=np.float32), xvalues)
    x_idx = np.tile(np.arange(xvalues, dtype=np.float32), num_images)
    x_norm = x_idx / max(1.0, float(xvalues - 1))
    y_norm = y_idx / max(1.0, float(num_images - 1))
    return np.column_stack((x_norm, y_norm))


def equalize_xvalue_means_per_frequency(cube: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    For each frequency slice (num_images, xvalues), scale each x-value column
    so every column has the same mean intensity (target = slice-wide mean).
    """
    normalized = cube.copy()
    num_images, num_freq, xvalues = normalized.shape

    for fi in range(num_freq):
        freq_slice = normalized[:, fi, :]  # (num_images, xvalues)
        col_medians = np.median(freq_slice, axis=0)  # (xvalues,)
        target_mean = float(np.median(col_medians))
        scale = target_mean / np.maximum(col_medians, eps)
        normalized[:, fi, :] = freq_slice * scale[np.newaxis, :]

    return normalized


def save_normalized_frequency_figures(cube: np.ndarray, wavelengths: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    num_images, num_freq, xvalues = cube.shape

    for fi in range(num_freq):
        freq_img = cube[:, fi, :]
        wl = float(wavelengths[fi])
        out_path = out_dir / f"normalized_{wl:.2f}nm.png"

        plt.figure(figsize=(15, 15))
        plt.imshow(freq_img, cmap="gray", aspect="auto")
        plt.axis("off")
        plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()


def main():
    csv_dir = Path(INPUT_PATH)

    selected = SELECTED_FREQUENCIES if USE_SELECTED_FREQUENCIES else None
    cube, wavelengths = load_frequency_csvs(csv_dir, selected_frequencies=selected)
    cube = equalize_xvalue_means_per_frequency(cube)
    if VISUALIZE_NORMALIZED_FREQUENCIES:
        save_normalized_frequency_figures(cube, wavelengths, Path(NORMALIZED_FIGURE_DIR))
        print(f"Saved normalized frequency figures to: {Path(NORMALIZED_FIGURE_DIR).resolve()}")

    matrix_2d = cube_to_2d(cube)
    print(f"3D matrix shape (num_images, frequency, xvalues): {cube.shape}")
    print(f"2D matrix shape: {matrix_2d.shape}")
    print(f"Frequencies used ({len(wavelengths)}): {wavelengths}")

    num_images, num_freq, xvalues = cube.shape
    spectral_features = StandardScaler().fit_transform(matrix_2d).astype(np.float32)
    if USE_SPATIAL_FEATURES:
        xy = build_xy_features(num_images, xvalues)
        xy_scaled = StandardScaler().fit_transform(xy).astype(np.float32)
        xy_scaled[:, 0] *= SPATIAL_WEIGHT_X
        xy_scaled[:, 1] *= SPATIAL_WEIGHT_Y
        pixel_values = np.hstack((spectral_features, xy_scaled)).astype(np.float32)
    else:
        pixel_values = spectral_features

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    k = K_CLUSTERS
    retval, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    labels_1d = labels.reshape(-1)
    segmented_image = labels_1d.reshape(num_images, xvalues)

    plt.figure(figsize=(15, 15))
    plt.imshow(segmented_image, cmap="tab10", aspect="auto", interpolation="nearest")
    plt.axis("off")
    plt.savefig("segmented_kmeans.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    main()
