import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import sys

## float(-1, 1) -> int(0, 255) 로 mapping (이미지 저장용)
def numpy_mapping(arr):
    return ((arr + 1) * 255 / 2).astype(np.uint8)

def make_Pchannel(self, single_filename, data_path):
    line = single_filename.split() # date, number
    date = line[0]
    sub_folder = ["DoLP_undistorted", "AoLP_undistorted"]
    frame_index = str(line[1]).zfill(7)
    dolp_name = os.path.join(data_path, date, sub_folder[0], frame_index + ".jpg")
    aolp_name = os.path.join(data_path, date, sub_folder[1], frame_index + ".jpg")
    aolpIMG = cv2.imread(aolp_name, cv2.IMREAD_GRAYSCALE)
    dolpIMG = cv2.imread(dolp_name, cv2.IMREAD_GRAYSCALE)

    ## Aolp를 degree domain으로 변경
    aolpIMG = np.interp(aolpIMG, (0, 255), (0, 180))
    ## Dolp를 0~1로 normalize
    dolpIMG = dolpIMG / 255.0

    ## 셋다 범위 -1 ~ 1
    P0 = np.sin(2 * aolpIMG)
    P1 = np.cos(2 * aolpIMG)
    P2 = 2 * dolpIMG - 1

    p_image = cv2.merge([P0, P1, P2])
    p_image = numpy_mapping(p_image)
    cv2.imshow("image", p_image)  
    cv2.waitKey(0)
    # cv2.imwrite((r"./test.jpg"), p_image)
    print(os.path.join(data_path, "p_channel", date, frame_index + ".jpg"))
    cv2.imwrite(os.path.join(data_path, "p_channel", date, frame_index + ".jpg"), p_image)

train_filenames = ["20220621_142942 1365"]
total_length = len(train_filenames)
file_dir = os.path.dirname(__file__)
dataset_path = os.path.join(file_dir, "polarDataset")

print(dataset_path)

for i in range(total_length):
    make_Pchannel(i, train_filenames[i], dataset_path)