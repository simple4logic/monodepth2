# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil

from utils import readlines
from kitti_utils import generate_depth_map


def export_gt_depths_kitti():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["eigen", "eigen_benchmark"])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    for line in lines:

        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)

        ## TODO -> 현재 이 두 split도 아닌 polar_Ref split인데 이러면 필요 없는건가?
        if opt.split == "eigen":
            calib_dir = os.path.join(opt.data_path, "calibration")
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            
            ## lidar에서 값을 읽어와서, 이미지 사이즈만큼 생긴 0위에다가 값을 넣어온다. 이미지 크기의 depth map 로딩
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        elif opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth",
                                         "groundtruth", "image_02", "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256

        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depths.npz") ## 결국에는 npz로 저장하기 위한 코드로 보인다 일단 필요 없을 듯

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))

## 즉 이 파일을 train.py와는 별개로 "따로" 실행해서 gt를 npz 형태로 추출하는 python code이다
if __name__ == "__main__":
    export_gt_depths_kitti()
