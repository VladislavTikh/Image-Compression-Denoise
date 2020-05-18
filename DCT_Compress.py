import numpy as np
from PIL import Image
import tools as t


# Matrix to execute DCT transform
T = t.dctMatrixInit()
block = t.block
file = t.naturefile #ct.cityfile (option)
img = Image.open(file + t.extension)
width, height = img.size
# Convert Image to ycbcr
img = t.rgb2ycbcr(np.array(img))

# Ensure Image can be split in blocks by 8 and split it
height = height - height % block
width = width - width % block
hmax = height // block
wmax = width // block

# Compressed image
CompIMG = np.zeros((height, width, 3))

Q = np.array(t.Q)

# Process every component
for kk in range(3):
    I = img[:height, :width, kk]
    MM = np.array(I, dtype=np.float32)
    CC = np.zeros((height, width))
    # Adjust quantization matrix to specific component
    QQ = Q * t.Qual[kk]
    for i in range(hmax):
        for j in range(wmax):
            # Split data into 8x8 blocks
            M = MM[block * i: block * (i + 1), block * j: block * (j + 1)].copy()
            # DCT in matrix way T*M*T'
            D = np.matmul(np.matmul(T, M), T.T)
            # Compress dividing by quantization matrix (most coef are zeros)
            C = np.round(D / QQ)
            # Build 8x8 blocks together
            CC[block * i: block * (i + 1), block * j: block * (j + 1)] = C.copy()
    RR = np.zeros((height, width))
    # Decompress image back
    for i in range(hmax):
        for j in range(wmax):
            # Get coefficients back
            R = CC[block * i: block * (i + 1), block * j: block * (j + 1)] * QQ
            # Build blocks together
            RR[block * i: block * (i + 1), block * j: block * (j + 1)] = R.copy()
    for i in range(hmax):
        for j in range(wmax):
            # Inverse DCT T'*RR*T
            CompIMG[block * i: block * (i + 1), block * j: block * (j + 1), kk] = np.matmul(
                np.matmul(T.T, RR[block * i: block * (i + 1), block * j: block * (j + 1)]), T)
# Convert Image back to RGB mode
t.saveImage(CompIMG, file + "_compressedDCT")
t.showResults(file, file + "_compressedDCT")
