"""Visualization helpers for standalone CC-SGCL."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

colors = np.array([
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
    [176, 48, 96], [46, 139, 87], [160, 32, 240], [255, 127, 80], [127, 255, 212],
    [218, 112, 214], [160, 82, 45], [127, 255, 0], [216, 191, 216], [128, 0, 0], [0, 128, 0],
    [0, 0, 128],
])


def imgDraw(label, imgName, path='./pictures', show=True):
    row, col = label.shape
    numClass = int(label.max())
    Y_RGB = np.zeros((row, col, 3)).astype('uint8')
    Y_RGB[np.where(label == 0)] = [0, 0, 0]
    for i in range(1, numClass + 1):
        try:
            Y_RGB[np.where(label == i)] = colors[i - 1]
        except Exception:
            Y_RGB[np.where(label == i)] = np.random.randint(0, 256, size=3)
    plt.axis('off')
    if show:
        plt.imshow(Y_RGB)
    os.makedirs(path, exist_ok=True)
    plt.imsave(path + '/' + str(imgName) + '.png', Y_RGB)
    return Y_RGB
