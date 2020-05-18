import numpy as np
from PIL import Image
import tools as t
import mkl_fft

block = t.block

# Initialize quantization matrix
Q = np.zeros((block, block))
Q[:, 0] = np.ones((1, block))
Q[0, :] = np.ones((1, block))

file = t.naturefile  # ct.cityfile

img = Image.open(file + t.extension)
width, height = img.size

img = t.rgb2ycbcr(np.array(img))

# Ensure Image can be split in blocks by 8 and split it
height = height - height % block
width = width - width % block
hmax = height // block
wmax = width // block

CompIMG = np.zeros((height, width, 3))

# Process every component
for kk in range(3):
    I = img[:height, :width, kk]
    MM = np.array(I, dtype=np.float32)
    CC = np.zeros((height, width))
    for i in range(hmax):
        for j in range(wmax):
            # Splint int 8x8 blocks
            M = MM[block * i: block * (i + 1), block * j: block * (j + 1)]
            # Execute fft
            D = np.real(mkl_fft.fft(M)).reshape((-1, 8))
            # Compress dividing by quantization matrix (most coef are zeros)
            C = np.round(D * Q)
            CC[block * i: block * (i + 1), block * j: block * (j + 1)] = C.copy()
    for i in range(hmax):
        for j in range(wmax):
            # Restore image with ifft
            CompIMG[block * i: block * (i + 1), block * j: block * (j + 1), kk] = np.real(
                mkl_fft.ifft(CC[block * i: block * (i + 1), block * j: block * (j + 1)])).reshape(-1, 8)
# Convert image back to RGB
t.saveImage(CompIMG, file + "_compressedFFT")
t.showResults(file, file + "_compressedFFT")
