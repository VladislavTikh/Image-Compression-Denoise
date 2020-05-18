import numpy as np
from PIL import Image
from skimage.util import random_noise
import tools as t

block = t.block
T = t.dctMatrixInit()
# Quantization matrix with discarded last 3 elements
Q = np.zeros((block, block))
Q[0:5, 0:5] = np.ones((5, 5))

file = t.cityfile
img = Image.open(file + t.extension)
width, height = img.size

img = t.rgb2ycbcr(np.array(img))

# Ensure Image can be split in blocks by 8 and split it
height = height - height % block
width = width - width % block
hmax = height // block
wmax = width // block

CompIMG = np.zeros((height, width, 3))
# Add some salt and pepper noise to the image
img = random_noise(img, mode='s&p', amount=0.1)
img = np.array(255 * img, dtype='uint8')
t.saveImage(img, file + "_Noised")

# Process every component
for kk in range(3):
    I = img[:height, :width, kk]
    MM = np.array(I, dtype=np.float32)
    CC = np.zeros((height, width))
    for i in range(hmax):
        for j in range(wmax):
            # Split into 8x8 blocks
            M = MM[block * i: block * (i + 1), block * j: block * (j + 1)]
            # Execute DCT
            D = np.matmul(np.matmul(T, M), T.T)
            C = np.round(D * Q)
            # Build blocks together
            CC[block * i: block * (i + 1), block * j: block * (j + 1)] = C.copy()
    for i in range(hmax):
        for j in range(wmax):
            # Inverse DCT T'*CC*T
            CompIMG[block * i: block * (i + 1), block * j: block * (j + 1), kk] = np.matmul(
                np.matmul(T.T, CC[block * i: block * (i + 1), block * j: block * (j + 1)]), T)
# Convert Image back to RGB
t.saveImage(CompIMG, file + "_Denoised")
t.showResults(file + "_Noised", file + "_Denoised", "Noised Image")
