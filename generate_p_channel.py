import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import sys

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

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
    # aolpIMG = np.interp(aolpIMG, (0, 255), (0, 180))
    ## Dolp를 0~1로 normalize
    aolpIMG = aolpIMG / 255.0 * np.pi   # 0 ~ pi
    dolpIMG = dolpIMG / 255.0           # 0 ~ 1

    ## 셋다 범위 -1 ~ 1
    P0 = np.sin(2 * aolpIMG)
    P1 = np.cos(2 * aolpIMG)
    P2 = 2 * dolpIMG - 1

    p_image = cv2.merge([P0, P1, P2])
    p_image = numpy_mapping(p_image)
    # cv2.imshow("image", p_image)  
    # cv2.waitKey(0)
    # cv2.imwrite((r"./test.jpg"), p_image)
    # print(os.path.join(data_path, "p_channel", date, frame_index + ".jpg"))

    makedirs(os.path.join(data_path, "p_channel", date))
    cv2.imwrite(os.path.join(data_path, "p_channel", date, frame_index + ".jpg"), p_image)


file_dir = os.path.dirname(__file__) ## 이 python 파일이 놓여있는 directory 위
dataset_path = os.path.join(file_dir, "polarimetric_imaging_dataset") ## linux
# dataset_path = os.path.join("C:\\Users\\okht1\\Downloads", "polarimetric_imaging_dataset") ## this window os

fpath = os.path.join(dataset_path, "data_splits", "monodepth", "{}_files.txt")
usage = ["train", "test", "val"]
# print(dataset_path)

for name in usage:
    filenames = ["20220621_142942 1365"]
    # filenames = readlines(fpath.format(name)) #["20220621_142942 1365"]
    total_length = len(filenames)
    for i in range(total_length):
        # print("target file name : ", train_filenames[i])
        make_Pchannel(i, filenames[i], dataset_path)
        print(i, "-th pre-processing done!", " working on ", name, " currently.")