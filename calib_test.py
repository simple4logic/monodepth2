import os
import numpy as np
from collections import Counter
import open3d as o3d
from matplotlib import pyplot as plt


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """

    lidar2cam_path = os.path.join(calib_dir,'lidar_to_camera_transform.txt')
    camera_intr_path = os.path.join(calib_dir,'camera_intrinsics.txt')

    gt_npy = np.load(velo_filename)
    # print(gt_npy)
    xyz = gt_npy["xyz"]
    # intensity = gt_npy["intensity"]

    lidar2cam = np.loadtxt(lidar2cam_path, delimiter=' ') ## 우리가 로드하는 데이터는 이미 3 * 4
    camera_intrinsics = np.loadtxt(camera_intr_path, delimiter=' ')

    lidar_points_3d_homogeneous = np.hstack((xyz, np.ones((np.shape(xyz)[0], 1)))) ## n * 3 -> n * 4로 확장
    camera_points_3d = lidar2cam @ lidar_points_3d_homogeneous.T ## 3 * 4 X 4 * n => 3 * n
    camera_points_3d_in_front = camera_points_3d[:, camera_points_3d[2, :] > 0] ## 앞 부분만 채택
    image_points_2d = camera_intrinsics @ camera_points_3d_in_front ## => 3 * n
    ## 아래의 depth value 부분에서
    ## camera_points_3d_in_front[2, mask]를 그대로 쓰는 거랑
    ## camera instrinsics가 곱해진 image_points_2d[2, mask]를 쓰는 거랑 이미지 차이가 없음... 정상인가??
    # temp = image_points_2d[2, :]

    ## 동차 좌표계 => 2D 좌표계로 변환 (u, v, w) 꼴 -> (u/w, v/w) 꼴로 변환
    image_points_2d = image_points_2d[:2, :] / image_points_2d[2, :] 
    image_points_2d = np.round(image_points_2d).astype(int) ## 반올림 이후 int형으로 변환
    print("shape of image_points_2d : ", np.shape(image_points_2d))

    ## frame 내부만 포인트만 가져오도록 masking
    mask_x = np.logical_and(image_points_2d[0, :] >= 0, image_points_2d[0, :] < im_shape[1])
    mask_y = np.logical_and(image_points_2d[1, :] >= 0, image_points_2d[1, :] < im_shape[0])
    mask = np.logical_and(mask_x, mask_y)
    image_points_2d_in_frame = image_points_2d[:, mask]
    depth_value = camera_points_3d_in_front[2, mask] # temp[mask] 

    min_distance = 3
    max_distance = 40
    depth_value = np.clip(depth_value, min_distance, max_distance) # min, max clipping

    depth_map = np.full((im_shape[0], im_shape[1], 3), 0) ## 투영시킬 틀
    depth_value = (depth_value - min_distance) / (max_distance - min_distance) # do normalize
    depth_color = plt.cm.jet(depth_value)[:,:-1] ## depth value를 color 에 대응시킴

    depth_map[image_points_2d_in_frame[1, :], image_points_2d_in_frame[0, :]] = depth_color * 255 ## 0 ~ 1을 0 ~ 255로 확장

    plt.imshow(depth_map, cmap='jet')
    plt.colorbar(label='Depth')
    plt.title('Depth Map')
    plt.show()

    # lidar visualization test
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # o3d.visualization.draw_geometries([pcd])
    exit()

    ############################### 여기까지가 image plane으로 투영된 라이다 데이터

    # always false
    # if vel_depth:
    #     velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    image_points_2d[:, 0] = np.round(image_points_2d[:, 0]) - 1
    image_points_2d[:, 1] = np.round(image_points_2d[:, 1]) - 1
    val_inds = (image_points_2d[:, 0] >= 0) & (image_points_2d[:, 1] >= 0)
    val_inds = val_inds & (image_points_2d[:, 0] < im_shape[1]) & (image_points_2d[:, 1] < im_shape[0])
    image_points_2d = image_points_2d[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[image_points_2d[:, 1].astype(np.int), image_points_2d[:, 0].astype(np.int)] = image_points_2d[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, image_points_2d[:, 1], image_points_2d[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(image_points_2d[pts[0], 0])
        y_loc = int(image_points_2d[pts[0], 1])
        depth[y_loc, x_loc] = image_points_2d[pts, 2].min()
    depth[depth < 0] = 0

    return depth

file_dir = os.path.dirname(__file__)
calib_path = os.path.join(file_dir, "polarimetric_imaging_dataset", "calibration")
target_npz_path = os.path.join(file_dir, "polarimetric_imaging_dataset", "20220621_142942", "XYZ", "0001365.npz")

## 높이, 너비, 채널 수 형태
im_shape = [1023, 1223]
gt_data = generate_depth_map(calib_path, target_npz_path)


#############

import numpy as np
import matplotlib.pyplot as plt
img = plt.imread('20220621_133513/RGB_undistorted/0000920.jpg').copy()
lidar = np.load('20220621_133513/XYZ/0000920.npz')
camera_intrinsics = np.loadtxt('camera_intrinsics.txt', delimiter=' ')
lidar2camera_transform = np.loadtxt('lidar_to_camera_transform.txt', delimiter=' ')
lidar_points_3d_homogeneous = np.hstack((lidar['xyz'], np.ones((lidar['xyz'].shape[0], 1))))
camera_points_3d = lidar2camera_transform @ lidar_points_3d_homogeneous.T
camera_points_3d_in_front = camera_points_3d[:, camera_points_3d[2, :] > 0]
image_points_2d = camera_intrinsics @ camera_points_3d_in_front
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
plt.imshow(img)
plt.show()


