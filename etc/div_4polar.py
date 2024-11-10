import cv2
import os
import numpy as np

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

dataset_path = os.path.join("C:\\Users\\okht1\\Downloads", "polarimetric_imaging_dataset") ## this window os
# dataset_path = os.path.join(file_dir, "polarimetric_imaging_dataset") ## linux

#input AoLP DoLP RGB img
# they are all btw 0 ~ 255
def calculate_I_values(single_filename, data_path):
    line = single_filename.split() # date, number
    date = line[0]
    frame_index = str(line[1]).zfill(7)
    sub_folder = ["DoLP_undistorted", "AoLP_undistorted", "RGB_undistorted"]
    # sub_folder = ["I_0", "I_45", "I_90", "I_135"]

    # aolp, dolp, rgb, filename
    dolp_name = os.path.join(data_path, date, sub_folder[0], frame_index + ".jpg")
    aolp_name = os.path.join(data_path, date, sub_folder[1], frame_index + ".jpg")
    rgb_name  = os.path.join(data_path, date, sub_folder[2], frame_index + ".jpg")
    aolpIMG = cv2.imread(aolp_name, cv2.IMREAD_GRAYSCALE)
    dolpIMG = cv2.imread(dolp_name, cv2.IMREAD_GRAYSCALE)
    rgbIMG  = cv2.imread(rgb_name, cv2.IMREAD_COLOR)

    # Normalize RGB image to intensity (I_total)
    # [H, W, RGB] // 마지막 차원에 대해서 평균 계산
    # I_total : 0 ~ 255
    I_total = np.mean(rgbIMG, axis=2) # I_total = (I0 + I45 I90 I135) / 2
    
    # Convert degrees to radians for calculations
    aolp = aolpIMG / 255.0 * np.pi     # 0 ~ pi
    dolp = dolpIMG / 255.0             # 0 ~ 1

    alpha = 2 * aolp
    
    # Calculate I0, I45, I90, and I135 using Aolp, Dolp, and I_total
    I0   = I_total * (1 + dolp * np.cos(alpha)) / 2
    I90  = I_total * (1 - dolp * np.cos(alpha)) / 2
    I45  = I_total * (1 + dolp * np.sin(alpha)) / 2
    I135 = I_total * (1 - dolp * np.sin(alpha)) / 2

    makedirs(os.path.join(dataset_path, "polar4", date, "I_0"  ))
    makedirs(os.path.join(dataset_path, "polar4", date, "I_45" ))
    makedirs(os.path.join(dataset_path, "polar4", date, "I_90" ))
    makedirs(os.path.join(dataset_path, "polar4", date, "I_135"))
    cv2.imwrite(os.path.join(dataset_path, "polar4", date, "I_0"  , frame_index + ".jpg"), I0  )
    cv2.imwrite(os.path.join(dataset_path, "polar4", date, "I_45" , frame_index + ".jpg"), I45 )
    cv2.imwrite(os.path.join(dataset_path, "polar4", date, "I_90" , frame_index + ".jpg"), I90 )
    cv2.imwrite(os.path.join(dataset_path, "polar4", date, "I_135", frame_index + ".jpg"), I135)

    return I0, I45, I90, I135

# DO HERE

fpath = os.path.join(dataset_path, "data_splits", "monodepth", "{}_files.txt")
usage = ["train", "test", "val"]
# print(dataset_path)

for name in usage:
    filenames = ["20220621_142942 1365"]
    # filenames = readlines(fpath.format(name)) #["20220621_142942 1365"]
    total_length = len(filenames)
    for i in range(total_length):
        # print("target file name : ", train_filenames[i])
        calculate_I_values(filenames[i], dataset_path)
        print(i, "-th pre-processing done!", " working on ", name, " currently.")

# I0, I45, I90, I135 = calculate_I_values(aolp, dolp, rgb)
