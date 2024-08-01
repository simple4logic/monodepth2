import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import sys

def make_Pchannel(self, single_filename, data_path):
    line = single_filename.split() # date, number
    date = line[0]
    sub_folder = ["DoLP_undistorted", "AoLP_undistorted"]
    frame_index = str(line[1]).zfill(7)
    dolp_name = os.path.join(data_path, date, sub_folder[0], frame_index, ".jpg")
    aolp_name = os.path.join(data_path, date, sub_folder[1], frame_index, ".jpg")
    aolpIMG = cv2.imread(aolp_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)  
    dolpIMG = cv2.imread(dolp_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)


    dolpIMG = dolpIMG / 255.0
    P0 = np.sin(2 * aolpIMG)
    P1 = np.cos(2 * aolpIMG)
    P2 = 2 * dolpIMG - 1

    p_image = cv2.merge([P0, P1, P2])
    cv2.imwrite(os.path.join(data_path, "p_channel", "date", frame_index, ".jpg"), p_image)


## normalize


## range check
# print("max dolp : ", np.max(P2), "\nmin : ", np.min(P2))
# print("max aolp : ", np.max(P1), "\nmin : ", np.min(P1))
# sys.exit()


# print(np.shape(p_image))
# plt.imshow(p_image)
# plt.show()



# fpath = os.path.join(os.path.dirname(__file__), "splits", "polar_ref", "{}_files.txt")
# train_filenames = readlines(fpath.format("train"))

train_filenames = ["20220621_142942 1365"]
total_length = len(train_filenames)


for i in range(total_length):
    make_Pchannel(i, train_filenames[i])
