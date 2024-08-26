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


'''
원래는 쓸 kitti-dataset의 .bin 파일들을 모두 모아서,
.npz 형태로 저장해주는 파일.
이렇게 gt를 뽑는 이유는 gt를 뽑아서 trained 모델에 대해 test할 때 쓰기 위해서이다.

polar에서는 각각에 대한 npz 파일을 썼지만, 여기서는 그 npz들을 모아 하나의 총괄적인 큰 npz 하나를 조립한 것으로 보인다
'''
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
                        choices=["eigen", "eigen_benchmark", "polar_ref"])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    i = 0
    for line in lines:
        i +=1
        ## 20192_12321 0000000.jpg
        folder, frame_id = line.split() ## folder == 날짜 / frame_id == 이미지 번호
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

        elif opt.split == "polar_ref":
            calib_dir = os.path.join(opt.data_path, "calibration")

            velo_filename = os.path.join(
                opt.data_path, folder, "XYZ", "{:07d}.npz".format(int(frame_id)))

            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)

        gt_depths.append(gt_depth.astype(np.float32))
        print(i,"-th gt done.")

    ## 모든 testfile에 대한 depth-gt가 합쳐진 npz 파일
    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))

## 즉 이 파일을 train.py와는 별개로 "따로" 실행해서 gt를 npz 형태로 추출하는 python code이다
if __name__ == "__main__":
    export_gt_depths_kitti()
