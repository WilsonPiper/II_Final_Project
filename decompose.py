import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


SPECTRA_ROOT = Path("/Users/lxl/Desktop/Imaging Instrumentation/Segmentation/segmented/spectra_results")
CSV_NAME = "multiple_spectra.csv"
OUTPUT_DIR = SPECTRA_ROOT.parent / "frequency_stacks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUTPUT_DIR = OUTPUT_DIR / "csv"
CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_index(name):
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


def extract_suffix_index(header_name):
    match = re.search(r"(\d+)$", header_name)
    return int(match.group(1)) if match else -1


def read_csv_matrix(csv_path):
    """
    Returns:
    - wavelengths: (num_frequencies,)
    - values:      (num_frequencies, width)
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    headers = list(data.dtype.names)

    wl_cols = sorted(
        [h for h in headers if h.startswith("wl_col_")],
        key=extract_suffix_index,
    )
    val_cols = sorted(
        [h for h in headers if h.startswith("val_col_")],
        key=extract_suffix_index,
    )

    if wl_cols:
        wavelengths = np.asarray(data[wl_cols[0]], dtype=float)
    else:
        wavelengths = np.asarray(data["wavelength_nm"], dtype=float)
    values = np.column_stack([np.asarray(data[h], dtype=float) for h in val_cols])
    return wavelengths, values


def load_cube(spectra_root):
    csv_paths = sorted(
        spectra_root.glob(f"*/{CSV_NAME}"),
        key=lambda p: extract_index(p.parent.name),
    )
    if not csv_paths:
        raise FileNotFoundError(f"No '{CSV_NAME}' files found under: {spectra_root}")

    ref_wl, ref_vals = read_csv_matrix(csv_paths[0])
    min_freq = ref_vals.shape[0]
    min_width = ref_vals.shape[1]
    matrices = [ref_vals]

    for csv_path in csv_paths[1:]:
        _, vals = read_csv_matrix(csv_path)
        matrices.append(vals)
        min_freq = min(min_freq, vals.shape[0])
        min_width = min(min_width, vals.shape[1])

    cube = np.stack([m[:min_freq, :min_width] for m in matrices], axis=0)
    return cube, ref_wl[:min_freq]


def save_frequency_images(cube, wavelengths, output_dir):
    # cube shape: (num_images, num_frequencies, width)
    num_images, num_freq, width = cube.shape

    for fi in range(num_freq):
        matrix = cube[:, fi, :]  # (num_images, width)
        matrix = matrix.reshape(num_images, width)
        m_min = float(np.min(matrix))
        m_max = float(np.max(matrix))
        if m_max > m_min:
            norm = (matrix - m_min) / (m_max - m_min)
        else:
            norm = np.zeros_like(matrix)

        wl = wavelengths[fi]
        out_path = output_dir / f"{wl:.2f}nm.png"
        plt.figure(figsize=(15, 15))
        plt.imshow(norm, cmap="gray", aspect="auto")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()


def save_frequency_csvs(cube, wavelengths, output_dir):
    """
    Save one compact CSV per wavelength with columns:
    image_idx, pixel_idx, intensity_milli
    where intensity_milli = round(intensity * 1000), i.e. 3 decimal precision.
    """
    num_images, num_freq, width = cube.shape

    image_idx = np.repeat(np.arange(num_images, dtype=np.int32), width)
    pixel_idx = np.tile(np.arange(width, dtype=np.int32), num_images)

    for fi in range(num_freq):
        matrix = cube[:, fi, :].reshape(num_images, width)
        intensity_milli = np.rint(matrix.reshape(-1) * 1000.0).astype(np.int32)

        out = np.column_stack((image_idx, pixel_idx, intensity_milli))
        wl = wavelengths[fi]
        out_path = output_dir / f"{wl:.2f}nm.csv"
        np.savetxt(
            out_path,
            out,
            delimiter=",",
            fmt=["%d", "%d", "%d"],
            header="image_idx,pixel_idx,intensity_milli",
            comments="",
        )


def main():
    cube, wavelengths = load_cube(SPECTRA_ROOT)
    save_frequency_images(cube, wavelengths, OUTPUT_DIR)
    save_frequency_csvs(cube, wavelengths, CSV_OUTPUT_DIR)

    print(f"Cube shape: {cube.shape}")
    print(f"Saved {cube.shape[1]} frequency images to: {OUTPUT_DIR}")
    print(f"Saved {cube.shape[1]} frequency CSVs to: {CSV_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
