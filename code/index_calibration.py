import numpy as np
import matplotlib.pyplot as plt
import os

# --- SETTINGS ---
folder_path = '/Users/sareena/PycharmProjects/Spectrometer_Analysis/smiley_color_new_camera_cropped'  # Point to your folder
files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])

if not files:
    print("No .npy files found! Check your folder path.")
else:
    # Load the first file
    sample_data = np.load(os.path.join(folder_path, files[0]))

    # average rgb channels away
    intensity = np.mean(sample_data, axis=-1)

    # average spatial rows to get spectral rows
    spectral_profile = np.mean(intensity, axis=0)
    cols = np.arange(len(spectral_profile))

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})

    # Top Plot: Heatmap (What the sensor actually sees)
    img = ax1.imshow(intensity, aspect='auto', cmap='gray')
    ax1.set_title("Raw Spectral Frame (Spatial Rows vs Spectral Columns)")
    ax1.set_ylabel("Spatial Axis (Slit Height)")
    ax1.set_xlabel("Column Index (Spectral Axis)")
    fig.colorbar(img, ax=ax1, orientation='vertical', label='Intensity')

    # Bottom Plot: The "Peak Finder"
    ax2.plot(cols, spectral_profile, color='black', linewidth=1.5)
    ax2.set_title("Spectral Intensity Profile")
    ax2.set_xlabel("Column Index")
    ax2.set_ylabel("Brightness")
    ax2.grid(alpha=0.3)

    # ADD INSTRUCTIONS
    print("\n--- HOW TO CALIBRATE ---")
    print("1. Look at the bottom graph.")
    print("2. Find the 'humps' or peaks in the graph.")
    print("3. Hover your mouse over the Blue, Green, and Red peaks.")
    print("4. Note the 'x=' value in the bottom right of the window.")
    print("5. Use those numbers for blue_col_idx, green_col_idx, and red_col_idx.")

    #save_path = '/Users/sareena/PycharmProjects/Spectrometer_Analysis/images_for_report'
    plt.tight_layout()
    #plt.savefig("smiley_linstage_old_camera_calibration.png")
    plt.show()