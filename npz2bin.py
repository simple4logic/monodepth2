import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import open3d as o3d


def convert_npz2img(single_filename, data_path):
    line = single_filename.split() # date, number
    date = line[0]
    sub_folder = ["XYZ", "DoLP_undistorted"]
    frame_index = str(line[1]).zfill(7)
    gt_name = os.path.join(data_path, date, sub_folder[0], frame_index + ".npz")
    img_name = os.path.join(data_path, date, sub_folder[1], frame_index + ".jpg")
    img = plt.imread(img_name).copy()
    data = np.load(gt_name) ## lidar file open
    xyz = data["xyz"]
    print("shape of XYZ",np.shape(xyz))

    cam_intr_path = os.path.join(data_path,'camera_intrinsics.txt')
    lidar2cam_path = os.path.join(data_path,'lidar_to_camera_transform.txt')

    camera_intrinsics = np.loadtxt(cam_intr_path, delimiter=' ')
    lidar2camera_transform = np.loadtxt(lidar2cam_path, delimiter=' ')

    lidar_points_3d_homogeneous = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
    camera_points_3d = lidar2camera_transform @ lidar_points_3d_homogeneous.T
    camera_points_3d_in_front = camera_points_3d[:, camera_points_3d[2, :] > 0]
    image_points_2d = camera_intrinsics @ camera_points_3d_in_front
    print(np.shape(image_points_2d))

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(image_points_2d.T)
    # o3d.visualization.draw_geometries([pcd])

    ## image_Points_2d => image plane에서 나타내어진 좌표계
    image_points_2d = image_points_2d[:2, :] / image_points_2d[2, :]
    image_points_2d = np.round(image_points_2d).astype(int)
    mask_x = np.logical_and(image_points_2d[0, :] >= 0, image_points_2d[0, :] < img.shape[1])
    mask_y = np.logical_and(image_points_2d[1, :] >= 0, image_points_2d[1, :] < img.shape[0])
    mask = np.logical_and(mask_x, mask_y)
    image_points_2d_in_frame = image_points_2d[:, mask]
    intensity = camera_points_3d_in_front[2, mask]

    min_distance = 3
    max_distance = 40
    intensity[intensity < min_distance] = min_distance
    intensity[intensity > max_distance] = max_distance
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    intensity_color = plt.cm.jet(intensity)[:,:-1]
    img[image_points_2d_in_frame[1, :], image_points_2d_in_frame[0, :]] = intensity_color * 255

    plt.show(img)


    # npz 파일에서 데이터 읽기
    # intensity = data["intensity"]
    # target_name = os.path.join(data_path, "gt_bin", date, frame_index + ".bin")


train_filenames = ["20220621_142942 1365"]
total_length = len(train_filenames)

file_dir = os.path.dirname(__file__)
dataset_path = os.path.join(file_dir, "polarimetric_imaging_dataset")

for i in range(total_length):
    convert_npz2img(train_filenames[i], dataset_path)