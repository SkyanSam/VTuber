import numpy as np
import cv2
nrows = 600
ncols = 800
image_in = np.random.randint(0, 255, size=(nrows, ncols, 3))
scale_factor = 1.5

r = np.arange(nrows, dtype=float) * scale_factor
c = np.arange(ncols, dtype=float) * scale_factor

rr, cc = np.meshgrid(r, c, indexing='ij')

# Nearest Neighbor Interpolation
# np.floor if scale_factor >= 1. np.ceil otherwise
rr = np.floor(rr).astype(int).clip(0, nrows-1)
cc = np.floor(cc).astype(int).clip(0, ncols-1)

image_out = image_in
image_out[rr, cc, :] = image_in
print(image_in.shape)
print(image_out.shape)
cv2.imshow("aaa", image_in)
