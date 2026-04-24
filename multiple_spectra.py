import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

img_bgr = cv2.imread('IMG_6387.JPG')
H, W = img_bgr.shape[:2]


num_spectra = 20
cols = np.linspace(0, W-1, num_spectra).astype(int)

half_width = 2

all_spectra = []

for col in cols:
    c1 = max(0, col - half_width)
    c2 = min(W, col + half_width + 1)

    vertical_slice = img_bgr[:, c1:c2, :]

    hsv_img = cv2.cvtColor(vertical_slice, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    # average across the small column window
    h_mean = np.nanmean(h, axis=1)
    v_mean = np.nanmean(v, axis=1)

    hue_deg = h_mean.astype(float) * 2
    hue_deg[hue_deg > 270] = np.nan

    wavelength = 700 - (hue_deg / 360) * (700 - 400)
    idx = np.arange(len(v_mean))
    v_flat = v_mean.astype(float)
    wl_flat = wavelength.astype(float)

    # fit wavelength vs pixel index using bright pixels
    p50 = np.percentile(v_flat, 50)
    mask = (v_flat >= p50) & (~np.isnan(wl_flat))
    if np.sum(mask) < 2:
        print(f"WARNING: Insufficient Mask for col = {col}")
        continue

    a, b = np.polyfit(idx[mask], wl_flat[mask], 1)
    wl_linear_fit = a * idx + b

    order = np.argsort(wl_linear_fit)
    x = wl_linear_fit[order]
    y = v_flat[order]

    # smooth
    if len(y) >= 41:
        y_smooth = savgol_filter(y, window_length=41, polyorder=3)
    else:
        y_smooth = y

    all_spectra.append((col, x, y_smooth))

# plot all spectra
plt.figure(figsize=(12, 6))


cmap = plt.cm.viridis
length = len(all_spectra)
for i, (col, x, y_smooth) in enumerate(all_spectra):
    color = cmap(i / (length - 1))
    plt.plot(x, y_smooth, linewidth=2, label=f"x={col}", color = color)


plt.xlabel("Wavelength (nm)")
plt.ylabel("Value")
plt.title("Multiple spectra from left to right")
plt.legend(fontsize=8)
plt.savefig("multiple_spectra.png", dpi=300)
plt.show()