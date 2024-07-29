import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import sys

fpath = "polarDataset/20220621_142942"
aolpdir = os.path.join(fpath, "AoLP_undistorted")
dolpdir = os.path.join(fpath, "DoLP_undistorted")

filename = ["0001365"]

aolpIMG = cv2.imread(os.path.join(aolpdir, filename[0]+".jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
dolpIMG = cv2.imread(os.path.join(dolpdir, filename[0]+".jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)

## normalize
dolpIMG = dolpIMG / 255.0

P0 = np.sin(2 * aolpIMG)
P1 = np.cos(2 * aolpIMG)
P2 = 2 * dolpIMG - 1

## range check
# print("max dolp : ", np.max(P2), "\nmin : ", np.min(P2))
# print("max aolp : ", np.max(P1), "\nmin : ", np.min(P1))
# sys.exit()

p_image = cv2.merge([P0, P1, P2])

print(np.shape(p_image))
plt.imshow(p_image)
plt.show()