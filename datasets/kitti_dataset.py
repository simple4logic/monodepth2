# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation. 
        # width = 1216, height = 1024
        self.K = np.array([[0.94025049342, 0, 0.49906634292, 0],
                           [0, 1.12856769531, 0.50701688867, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1242, 375)
        self.full_res_shape = (1223, 1023)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        ## filenames[0] = "1111111111_1111 00"
        line = self.filenames[0].split() ## 정황상 첫번째 양식만 가져와서 gt가 존재하는지 확인하는 것 같음
        folder = line[0] ## 날짜
        frame_index = int(line[1]) ## 이미지 번호

        velo_filename = os.path.join(
            self.data_path,
            folder, ## 날짜 이름 폴더
            "XYZ",
            "{:07d}.npz".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index):
        color = self.loader(self.get_image_path(folder, frame_index))

        ## do not flip
        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = self.data_path

        ## 특정 idx image의 depth ground truth인 lidar point cloud를 로드
        ## 이는 bin 형식으로 정렬되어있음
        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    ## depth image 말하는 것 같음
    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

class polarDataset(KITTIDataset):
    """polar dataset load original depth maps(npz format) for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(polarDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index):
        f_str = "{:07d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, "p_channel", folder, f_str)
        # image_path = os.path.join(self.data_path, folder, "RGB_undistorted", f_str) # for RGB input
        return image_path

    ## TODO -> gt-depth를 만들때 호출되는 함수
    def get_depth(self, folder, frame_index):
        ## data_path = os.path.join("data","UROP","UROP_polardepth","polarimetric_imaging_dataset",)
        calib_path = os.path.join(self.data_path, "calibration")

        ## 특정 idx image의 depth ground truth인 lidar point cloud를 로드
        ## 이는 bin 형식으로 정렬되어있음
        velo_filename = os.path.join(
            self.data_path,
            folder,
            "XYZ",
            "{:07d}.npz".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename)

        ## depthmap data를 resize함
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        ## do not flip
        # if do_flip:
        #     depth_gt = np.fliplr(depth_gt)

        return depth_gt