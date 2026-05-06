Final project - Imaging Instrumentation
==============

**Author:** *Sareena*, *Wilson*, and *Ruthika*, advised under *Dr. Web Stayman*

# Repository Structure
- [x] segmented: contain sample images to process (which are captured by our self-made spectrometer)
- [x] code: contains code files (detailed explanation is provided in the following sections)

# A few flags to check before running

1. Check if the image is vertical or not on the original image. If vertical, set the rotation flag in multiple_spectra to be True. 
2. Also, if rotation is needed, check if the image has blue on the left side or not. The inputted image needs to have blue on the left side.
3. If no rotation is needed, the inputted image needs to have blue on the TOP side. 
4. If those flags are not configured correctly, the code is not going to work correctly

# Pre-processing

The following pre-processing is required before analyzing using the code.
- Crop the image to determine FOV. Ideally cropped to contain spectrum only
- Rotate the image to make sure image is upright. The code could deal with horizontal or vertical spectrum by manually setting flags. But starting from a 45 degree rotated spectrum, for example, will be problematic.

# Running the code

Note: This repo’s core scripts are configured by editing constants in each file.

## 1) `multiple_spectra.py`

Extracts spectra from one image or a folder of images and writes `multiple_spectra.csv` (and optional plots).

Run:
```bash
python3 multiple_spectra.py
```

Main constants to set:
- `INPUT_PATH`: image file or image folder
- `NUM_SPECTRA`: number of sampled vertical slices (`None` = all columns)
- `HALF_WIDTH`: half-width of each sampled slice in pixels
- `VERTICAL`: rotate image 90 deg CCW before extraction
- `PLOT_ENABLED`: save `multiple_spectra.png`
- `WAVELENGTH_STEP_NM`: wavelength resampling step (for shared grid output)
- `LINEAR_WAVELENGTH_SCALE`: `True` for adding linear 400-700nm axis, `False` for hue-based mapping

## 2) `decompose.py`

Builds a frequency cube from all `multiple_spectra.csv` files and exports per-wavelength image/CSV stacks.

Run:
```bash
python3 decompose.py
```

Main constants to set:
- `SPECTRA_ROOT`: folder containing per-image spectra folders
- `CSV_NAME`: spectra filename to load


## 3) `kmeans.py`

Runs K-means segmentation on decomposed frequency CSV stacks.

Run:
```bash
python3 kmeans.py
```

Main constants to set:
- `INPUT_PATH`: folder of per-wavelength CSVs
- `SELECTED_FREQUENCIES`: wavelengths to use when filtering
- `USE_SELECTED_FREQUENCIES`: enable/disable wavelength filtering
- `K_CLUSTERS`: number of K-means clusters
- `USE_SPATIAL_FEATURES`: include normalized x/y position in clustering
- `SPATIAL_WEIGHT_X`, `SPATIAL_WEIGHT_Y`: weights for spatial features
- `VISUALIZE_NORMALIZED_FREQUENCIES`: optionally save normalized frequency figures
- `NORMALIZED_FIGURE_DIR`: output directory for normalized frequency images

## 4) `index_calibration.py`

Helps find spectral column indices for blue/green/red peaks by plotting:
- a grayscale intensity heatmap of one sample frame
- a 1D spectral profile for manual peak picking

Run:
```bash
python3 index_calibration.py
```

Main constant to set:
- `folder_path`: path to the folder containing `.npy` files
- Run the script, inspect the bottom plot, and record the x-index of blue/green/red peaks.
- Use those values in `reconstruction.py` as `blue_col_idx`, `green_col_idx`, and `red_col_idx`.

## 5) `reconstruction.py`

Reconstructs an RGB image from a stack of `.npy` spectral frames using Gaussian spectral weights for R/G/B.

Run:
```bash
python3 reconstruction.py
```

Main constants to set:
- `folder_path`: path to input `.npy` stack
- `blue_col_idx`, `green_col_idx`, `red_col_idx`: calibration peak indices (from `index_calibration.py`)
- `filter_width`: Gaussian width controlling spectral band spread

Output:
- Displays filter curves and reconstructed image 
