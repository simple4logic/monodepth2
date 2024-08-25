from __future__ import absolute_import, division, print_function

import os
import numpy as np
from collections import Counter


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    # calib_dir = monodepth2/polarimetric_imaging_dataset/calibration/
    lidar2cam_path = os.path.join(calib_dir,'lidar_to_camera_transform.txt')
    camera_intrinsics_path = os.path.join(calib_dir,'camera_intrinsics.txt')
    lidar2cam = np.loadtxt(lidar2cam_path, delimiter=' ') ## 우리가 로드하는 데이터는 이미 3 * 4
    camera_intrinsics = np.loadtxt(camera_intrinsics_path, delimiter=' ')

    # load velodyne data (ground truth)
    ## velo_filename 
    gt_npy = np.load(velo_filename)
    xyz = gt_npy["xyz"] # intensity = gt_npy["intensity"]

    lidar_points_3d_homogeneous = np.hstack((xyz, np.ones((np.shape(xyz)[0], 1)))) ## n * 3 -> n * 4로 확장
    camera_points_3d = lidar2cam @ lidar_points_3d_homogeneous.T ## (3 X 4) * (4 X n) => 3 X n
    camera_points_3d_in_front = camera_points_3d[:, camera_points_3d[2, :] > 0] ## 이미지 뒷부분 제거
    image_points_2d = camera_intrinsics @ camera_points_3d_in_front ## => (3 X 3) * (3 X n) => 3 X n

    ## 동차 좌표계 => 2D 좌표계로 변환 (u, v, w) 꼴 -> (u/w, v/w) 꼴로 변환
    image_points_2d = image_points_2d[:2, :] / image_points_2d[2, :] ## 2 X n 
    image_points_2d = np.round(image_points_2d).astype(int) - 1 ## 반올림 이후 int형으로 변환 후 -1(원본 코드)

    # get image shape
    im_shape = [1023, 1223] # 그냥 이미지 사이즈(1223 * 1023) 고정으로 넣음 (가능하면 read 한 이후 shape 가져오는게 좋음)

    ## check if in bounds
    ## 2d plane에 투영시킨 이후, 원본 이미지 범위만 가져오도록 mask 준비
    mask_x = np.logical_and(image_points_2d[0, :] >= 0, image_points_2d[0, :] < im_shape[1])
    mask_y = np.logical_and(image_points_2d[1, :] >= 0, image_points_2d[1, :] < im_shape[0])
    mask = np.logical_and(mask_x, mask_y)

    ## mask 적용
    image_points_2d_in_frame = image_points_2d[:, mask] # 2 X n(with mask), set of (u/w, v/w)
    depth_value = camera_points_3d_in_front[2, mask] # 1 X n(with mask)   # temp[mask]

    ## transpose -> 원래 코드에 맞게 전부 transpose
    image_points_2d_in_frame = image_points_2d_in_frame.T # n X 2
    depth_value = depth_value.T # n X 1

    ## min, max clipping
    min_distance = 3
    max_distance = 40
    depth_value = np.clip(depth_value, min_distance, max_distance) # min, max clipping

    ## 여기서는 normalize 안했는데 이걸 해야할지 말아야할지 보류!! Q
    # depth_value = (depth_value - min_distance) / (max_distance - min_distance) # do normalize

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[image_points_2d_in_frame[:, 1], image_points_2d_in_frame[:, 0]] = depth_value

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, image_points_2d_in_frame[:, 1], image_points_2d_in_frame[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]

    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(image_points_2d_in_frame[pts[0], 0])
        y_loc = int(image_points_2d_in_frame[pts[0], 1])
        depth[y_loc, x_loc] = depth_value[pts].min()
    depth[depth < 0] = 0

    return depth
