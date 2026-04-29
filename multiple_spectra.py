from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def compute_spectra(img_bgr, num_spectra=20, half_width=2):
    height, width = img_bgr.shape[:2]
    if num_spectra is None:
        cols = np.arange(width, dtype=int)
    else:
        cols = np.linspace(0, width - 1, num_spectra).astype(int)

    all_spectra = []
    for col in cols:
        c1 = max(0, col - half_width)
        c2 = min(width, col + half_width + 1)
        vertical_slice = img_bgr[:, c1:c2, :]

        hsv_img = cv2.cvtColor(vertical_slice, cv2.COLOR_BGR2HSV)
        h, _, v = cv2.split(hsv_img)

        h_mean = np.nanmean(h, axis=1)
        v_mean = np.nanmean(v, axis=1)

        hue_deg = h_mean.astype(float) * 2
        hue_deg[hue_deg > 290] = np.nan

        wavelength = 700 - (hue_deg / 290.0) * (611.28 - 392.58)

        # Data source: https://colorspect.com/hue-wavl-convert.htm?rgb2_x=00AA00&col1_x=FF0000&hue_f=0.00&wavl_f=611.28
        idx = np.arange(len(v_mean), dtype=float)
        v_flat = v_mean.astype(float)
        wl_flat = wavelength.astype(float)

        p50 = np.percentile(v_flat, 50)
        mask = (v_flat >= p50) & (~np.isnan(wl_flat))
        if np.sum(mask) < 2:
            continue

        a, b = np.polyfit(idx[mask], wl_flat[mask], 1)
        wl_linear_fit = a * idx + b

        order = np.argsort(wl_linear_fit)
        x = wl_linear_fit[order]
        y = v_flat[order]

        if len(y) >= 41:
            y_smooth = savgol_filter(y, window_length=41, polyorder=3)
        else:
            y_smooth = y

        all_spectra.append((col, x, y_smooth))

    return all_spectra, height


def get_image_wavelength_range(all_spectra):
    """Return (min_wavelength, max_wavelength) from one image's spectra."""
    if not all_spectra:
        return None, None
    wl_min = min(float(np.nanmin(x)) for _, x, _ in all_spectra)
    wl_max = max(float(np.nanmax(x)) for _, x, _ in all_spectra)
    return wl_min, wl_max


def save_plot(all_spectra, plot_path):
    plt.figure(figsize=(12, 6))

    cmap = plt.cm.viridis
    length = len(all_spectra)
    if length == 0:
        plt.title("No valid spectra found")
    else:
        for i, (col, x, y_smooth) in enumerate(all_spectra):
            color = cmap(i / max(1, length - 1))
            plt.plot(x, y_smooth, linewidth=2, label=f"x={col}", color=color)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Value")
    plt.title("Multiple spectra from left to right")
    if length > 0:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def save_spectra_csv(all_spectra, csv_path, height, wavelength_step_nm=None, wavelength_grid=None):
    if not all_spectra:
        np.savetxt(csv_path, np.empty((0, 1)), delimiter=",", header="wavelength_nm", comments="")
        return

    # Optional wavelength-grid export (uniform frequency axis).
    if wavelength_grid is not None:
        grid = np.asarray(wavelength_grid, dtype=float)
        data_cols = [grid]
        headers = ["wavelength_nm"]
        for col, x, y in all_spectra:
            y_interp = np.interp(grid, x, y, left=0.0, right=0.0)
            data_cols.append(y_interp)
            headers.append(f"val_col_{col}")

        arr = np.column_stack(data_cols)
        header_line = ",".join(headers)
        np.savetxt(csv_path, arr, delimiter=",", header=header_line, comments="")
        return

    if wavelength_step_nm is not None and wavelength_step_nm > 0:
        wl_min = max(400.0, min(np.nanmin(x) for _, x, _ in all_spectra))
        wl_max = min(700.0, max(np.nanmax(x) for _, x, _ in all_spectra))
        grid = np.arange(wl_min, wl_max + wavelength_step_nm / 2.0, wavelength_step_nm)

        data_cols = [grid]
        headers = ["wavelength_nm"]
        for col, x, y in all_spectra:
            y_interp = np.interp(grid, x, y, left=0.0, right=0.0)
            data_cols.append(y_interp)
            headers.append(f"val_col_{col}")

        arr = np.column_stack(data_cols)
        header_line = ",".join(headers)
        np.savetxt(csv_path, arr, delimiter=",", header=header_line, comments="")
        return

    data_cols = [np.arange(height)]
    headers = ["pixel_row"]
    for col, x, y in all_spectra:
        data_cols.append(x)
        data_cols.append(y)
        headers.append(f"wl_col_{col}")
        headers.append(f"val_col_{col}")

    arr = np.column_stack(data_cols)
    header_line = ",".join(headers)
    np.savetxt(csv_path, arr, delimiter=",", header=header_line, comments="")


def process_image(
    image_path,
    results_root,
    num_spectra=20,
    half_width=2,
    vertical=False,
    plot_enabled=True,
    wavelength_step_nm=None,
    wavelength_grid=None,
    precomputed=None,
):
    image_out_dir = results_root / image_path.stem
    image_out_dir.mkdir(parents=True, exist_ok=True)

    if precomputed is None:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            return
        if vertical:
            # Rotate 90 degrees clockwise before spectra extraction.
            img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        all_spectra, height = compute_spectra(img_bgr, num_spectra=num_spectra, half_width=half_width)
    else:
        all_spectra = precomputed["all_spectra"]
        height = precomputed["height"]

    if plot_enabled:
        save_plot(all_spectra, image_out_dir / "multiple_spectra.png")
    save_spectra_csv(
        all_spectra,
        image_out_dir / "multiple_spectra.csv",
        height,
        wavelength_step_nm=wavelength_step_nm,
        wavelength_grid=wavelength_grid,
    )

    print(f"Processed {image_path.name}")


def process_folder(
    input_dir,
    num_spectra=20,
    half_width=2,
    vertical=False,
    plot_enabled=True,
    wavelength_step_nm=None,
):
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Invalid input directory: {input_dir}")

    image_paths = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS
    )
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    results_root = input_dir / "spectra_results"
    results_root.mkdir(parents=True, exist_ok=True)

    # Pass 1: compute spectra + wavelength range per image.
    precomputed_by_path = {}
    mins = []
    maxs = []

    for image_path in image_paths:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        if vertical:
            img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)

        all_spectra, height = compute_spectra(img_bgr, num_spectra=num_spectra, half_width=half_width)
        wl_min, wl_max = get_image_wavelength_range(all_spectra)
        if wl_min is None or wl_max is None:
            print(f"Skipping image with no valid spectra: {image_path.name}")
            continue

        precomputed_by_path[image_path] = {"all_spectra": all_spectra, "height": height}
        mins.append(wl_min)
        maxs.append(wl_max)

    if not precomputed_by_path:
        print("No valid spectra found in folder.")
        return

    global_min = min(mins)
    global_max = max(maxs)
    if global_min >= global_max:
        raise ValueError(
            f"Invalid global wavelength range across images: [{global_min:.2f}, {global_max:.2f}]"
        )

    # Build one uniform wavelength grid shared by all images (full span, not overlap).
    if wavelength_step_nm is not None and wavelength_step_nm > 0:
        common_grid = np.arange(
            global_min,
            global_max + wavelength_step_nm / 2.0,
            wavelength_step_nm,
        )
    else:
        first_path = next(iter(precomputed_by_path))
        first_all = precomputed_by_path[first_path]["all_spectra"]
        ref = np.sort(np.unique(np.concatenate([x for _, x, _ in first_all])))
        common_grid = ref[(ref >= global_min) & (ref <= global_max)]

    if common_grid.size < 2:
        raise ValueError("Common wavelength grid is too small after overlap filtering.")

    print(
        f"Shared wavelength grid: {common_grid[0]:.2f} to {common_grid[-1]:.2f} nm "
        f"({common_grid.size} points)"
    )

    # Pass 2: save all images using the same wavelength grid.
    for image_path in image_paths:
        precomputed = precomputed_by_path.get(image_path)
        if precomputed is None:
            continue
        process_image(
            image_path,
            results_root,
            num_spectra=num_spectra,
            half_width=half_width,
            vertical=vertical,
            plot_enabled=plot_enabled,
            wavelength_step_nm=wavelength_step_nm,
            wavelength_grid=common_grid,
            precomputed=precomputed,
        )


if __name__ == "__main__":
    # Parameters
    INPUT_PATH = "segmented"
    NUM_SPECTRA = None
    HALF_WIDTH = 0
    VERTICAL = True
    PLOT_ENABLED = False
    WAVELENGTH_STEP_NM = 5.0

    input_path = Path(INPUT_PATH)
    if input_path.is_dir():
        process_folder(
            input_path,
            num_spectra=NUM_SPECTRA,
            half_width=HALF_WIDTH,
            vertical=VERTICAL,
            plot_enabled=PLOT_ENABLED,
            wavelength_step_nm=WAVELENGTH_STEP_NM,
        )
    else:
        results_root = input_path.parent / "spectra_results"
        results_root.mkdir(parents=True, exist_ok=True)
        process_image(
            input_path,
            results_root,
            num_spectra=NUM_SPECTRA,
            half_width=HALF_WIDTH,
            vertical=VERTICAL,
            plot_enabled=PLOT_ENABLED,
            wavelength_step_nm=WAVELENGTH_STEP_NM,
        )
