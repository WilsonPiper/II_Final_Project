import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
There are several assumptions made in this code:
1. The spectrum distribution is [400, 700] nm, which is the visible light range.
2. The pixel is linearly correlated with wavelength, which may not be the case. 
'''

img_rgb = cv2.imread('IMG_6388.JPG')

h, w = img_rgb.shape[:2]

# Right now it just takes column in the middle
col = w // 2
vertical_slice = img_rgb[:, col:col+1, :]

hsv_img = cv2.cvtColor(vertical_slice, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_img)

# Hue to wavelength convertion
hue_deg = h.astype(float) * 2 # OpenCV hue is in [0, 179], so we multiply by 2 to get degrees in [0, 360]
hue_deg[hue_deg > 270] = np.nan # so there are some outliers formed in this range, since purple is not monochromatic
# Therefore I decided to discard it
wavelength = 700 - (hue_deg / 360) * (700 - 400)

'''
# index to value (intensity) plot
plt.figure(figsize=(12, 6))
# plt.plot(wavelength.flatten(), s.flatten(), 'o', markersize=2)
# plt.plot(wavelength.flatten(), v.flatten(), 'o', markersize=2)
idx = np.arange(v.size)
plt.plot(idx, v.flatten(), 'o', markersize=2)
plt.show()

# index to wavelength plot
plt.figure(figsize=(12, 6))
plt.plot(idx, wavelength.flatten(), 'o', markersize=2)
plt.show()
'''

#  Linear fit 
idx = np.arange(v.size)
wl_flat = wavelength.flatten().astype(float)
v_flat = v.flatten().astype(float)
p50 = np.percentile(v_flat, 50)
mask = (v_flat >= p50) & (~np.isnan(wl_flat))
idx_fit = idx[mask]
wl_fit = wavelength.flatten()[mask]
a, b = np.polyfit(idx_fit, wl_fit, 1)
wl_linear_fit = a * idx + b

plt.figure(figsize=(12, 6))
plt.plot(idx, wavelength.flatten(), 'o', markersize=2, alpha=0.3, label='All points')
plt.plot(idx_fit, wl_fit, 'o', markersize=3, label='Points used for fit')
plt.plot(idx, wl_linear_fit, '-', linewidth=2, label='Linear Fit')
plt.xlabel("Pixel Index")
plt.ylabel("Wavelength (nm)")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(wl_linear_fit, v_flat, 'o', markersize=2, alpha=0.3)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Value")
plt.show()