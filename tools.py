import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

extension = ".jpg"
naturefile = "Nature"
cityfile = "Morning_City"

# JPEG Quantization Matrix
Q = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
]

# Default block size for DCT
block = 8

# Change coefficients to achieve best image quality
# brightness
Yq = 1
# blue chrominance
Cbq = 10
# red chrominance
Crq = 10

# q- component related to compression degree
Qual = [Yq, Cbq, Crq]


# Convert rgb to ycbcr
def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)


# Convert back
def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


def dctMatrixInit():
    T = np.zeros((block, block))
    T[0] = np.ones((1, block)) / (block ** 0.5)
    for i in range(1, block):
        for j in range(block):
            T[i][j] = (1 / 2) * np.cos((2 * j + 1) * i * np.pi / 16)
    return T


def saveImage(im, filename):
    im = ycbcr2rgb(im)
    result_image = Image.fromarray(im)
    result_image.save(filename + extension)


def getCompressionRatio(file1, file2):
    return os.path.getsize(file1 + extension) / os.path.getsize(file2 + extension)


def showResults(fileName1, fileName2, text="Original Image"):
    ratio = getCompressionRatio(fileName1, fileName2)
    imgOriginal = mpimg.imread(fileName1 + extension)
    imgCompressed = mpimg.imread(fileName2 + extension)
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(imgOriginal)
    ax1.set_title(text)
    ax2.imshow(imgCompressed)
    ax2.set_title('Compressed Image')
    plt.title("Compress ratio : " + str(ratio))
    plt.show()
