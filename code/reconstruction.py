# imports
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from natsort import natsorted

# get image stack
folder_path = '/Users/sareena/PycharmProjects/Spectrometer_Analysis/stickman_new_camera_cropped' # Path to your .npy files

# rotate the images (later perhaps, if needed)

# calibration: get the number of spectral collumns
sample_file = os.listdir(folder_path)[0]
sample_data = np.load(os.path.join(folder_path, sample_file))
# collapse rgb (third column) to intensity
sample_intensity = np.mean(sample_data, axis=2)
num_rows_spatial, num_cols_spectral = sample_intensity.shape

# find column indices where each of the colors peak (done in a separate file)
# hopefully each image should have the same number of wavelengths... define the red, blue and green peaks
blue_col_idx = 211
green_col_idx = 550
red_col_idx = 827
filter_width = 40 # determines how wide the eventual gaussian distribution will be

# for wavelength groups (red area 50 nm, green area 50 nm, blue area 50 nm): extract that
# region from each image in the original data and account for center; fit to gaussian distribution to count for variation
# in spectrum (it's nost just a color block)
cols = np.arange(num_cols_spectral) # create the indices
b_weight = np.exp(-((cols - blue_col_idx)**2) / (2 * filter_width**2))
g_weight = np.exp(-((cols - green_col_idx)**2) / (2 * filter_width**2))
r_weight = np.exp(-((cols - red_col_idx)**2) / (2 * filter_width**1.9))

# process the whole image stack, and define the number of steps; use that to define the
# final image that will be generated from the original image
files = natsorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])


#files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
num_steps = len(files)
print(num_steps)

# height (num images), width (spatial data), color (rgb -- will be calculated)
final_image = np.zeros((num_rows_spatial, num_steps, 3), dtype=np.float32)

# dot product column intensity (spectral data) against image intensity to get final reconstructed image
print("Processing stack...")
for i, filename in enumerate(files):
    raw_frame = np.load(os.path.join(folder_path, filename))
    #raw_frame = np.transpose(raw_frame, (1, 0, 2))

    # average away RGB data
    frame_intensity = np.mean(raw_frame, axis=2)

    # Background noise subtraction
    frame_intensity = np.where(frame_intensity < 10, 0, frame_intensity - 10)

    # Use explicit multiplication and sum to get the spatial line
    # Resulting lines should have shape (1550,)
    r_line = np.sum(frame_intensity *  20 * r_weight, axis=1)
    g_line = np.sum(frame_intensity * 20 * g_weight, axis=1)
    b_line = np.sum(frame_intensity * 20 * b_weight, axis=1)


    # Explicitly assign to the i-th column of the final image
    # Note: the [:, i, 0] syntax means: "All rows, column i, channel 0"
    final_image[:, i, 0] = r_line
    final_image[:,  i,  1] = g_line
    final_image[:, i,  2] = b_line

col_means = np.mean(final_image, axis=0)

# To avoid dividing by zero if a column is pitch black:
col_means[col_means == 0] = 1

# Divide each column by its own mean
# This makes the average brightness of every vertical slice identical
final_image = final_image / col_means

# Optional: Scale it back to a reasonable range before the final max-normalization
final_image = final_image * np.median(col_means)

# normalize and plot the results
final_image = (final_image) / (final_image.max())


final_image_resized = cv2.resize(final_image, (num_rows_spatial, num_rows_spatial), interpolation=cv2.INTER_CUBIC)
final_image_rotated = cv2.rotate(final_image_resized, cv2.ROTATE_90_CLOCKWISE)

# 4. PLOTTING EVERYTHING TOGETHER
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot A: Your "Filters" (To verify color mapping)
ax1.plot(cols, r_weight, 'r', label='Red Filter')
ax1.plot(cols, g_weight, 'g', label='Green Filter')
ax1.plot(cols, b_weight, 'b', label='Blue Filter')
ax1.set_title("Spectral Filter Calibration")
ax1.set_xlabel("Column Index (Wavelength)")
ax1.legend()

# Plot B: The Reconstructed Smiley Face
ax2.imshow(final_image_rotated)
ax2.set_title(f"Reconstructed Stickman Object ({num_steps} slices stretched)")
ax2.axis('off')

plt.tight_layout()
plt.show()