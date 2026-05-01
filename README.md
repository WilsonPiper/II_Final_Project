# Final project - Imaging Instrumentation

# HEIC to jpg convertion:
The following code may be utilized

`
for f in *.HEIC; do
  sips -s format jpeg "$f" --out "${f%.HEIC}.jpg"
done
`
# A few flags to change/check before running

1. Check if the image is vertical or not on the original image. If vertical, set the rotation flag in multiple_spectra to be True. 
2. Also, if rotation is needed, check if the image has blue on the left side or not. The inputted image needs to have blue on the left side.
3. If no rotation is needed, the inputted image needs to have blue on the TOP side. 
4. If those flags are not configured correctly, the code is not going to work correctly, unfortunately. 