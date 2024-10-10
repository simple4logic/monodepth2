# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    try:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    except (IOError, FileNotFoundError) as e:
        # IOError: 이미지 파일이 열리지 않는 경우
        # FileNotFoundError: 파일이 존재하지 않는 경우
        # print("Error occured : ",e)
        return None


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        # try:
        #     self.brightness = (0.8, 1.2)
        #     self.contrast = (0.8, 1.2)
        #     self.saturation = (0.8, 1.2)
        #     self.hue = (-0.1, 0.1)
        #     transforms.ColorJitter.get_params(
        #         self.brightness, self.contrast, self.saturation, self.hue)
        # except TypeError:
        #     self.brightness = 0.2
        #     self.contrast = 0.2
        #     self.saturation = 0.2
        #     self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    ## 하나의 index에 대해서 dictionary 제작
    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        # do_color_aug = self.is_train and random.random() > 0.5
        # do_flip = self.is_train and random.random() > 0.5

        ## filenames[0] = "1111111111_1111 00"
        line = self.filenames[index].split()
        folder = line[0] ## 날짜
        frame_index = int(line[1]) ## 이미지 번호
        
        ## left right 서로 뒤집는 부분!!
        # if len(line) == 3:
        #     frame_index = int(line[1])
        # else:
        #     frame_index = 0

        # if len(line) == 3:
        #     side = line[2]
        # else:
        #     side = None

        ## side 변수 삭제
        # side = "l"

        ## 연속된 구간인지 검사하고, 없을 경우 이 index는 학습 데이터로부터 제외시킴
        for i in self.frame_idxs:
            temp_test = self.get_color(folder, frame_index + i)
            if(temp_test == None):
                ## 한 번이라도 None이 확인되면, 그냥 None를 리턴해버려서 i 번째 데이터를 학습에 쓰지 않는다
                ## 이걸 로드하는 부분에서 None이면 그 데이터를 쓰지 않게 해야함
                return None

        ## frame_idxs = -1 0 1로 입력 받았음
        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            ## 역행렬 계산 => inversion of K
            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K) ## shape = 4*4
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K) ## shape = 4*4

        ## 색상 증강 설정 OFF
        color_aug = (lambda x: x) # 그냥 원본
        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)] # 원래 코드
            # inputs.pop(("color", i, -1), None) ## pop으로 에러 커버
            del inputs[("color_aug", i, -1)] # 원래 코드
            # inputs.pop(("color_aug", i, -1), None) ## pop으로 에러 커버

        ## 이 부분에서, kitti_dataeset에서 정의된 get_depth를 이용해 gt를 넣어준다
        ## kitti_dataset에서 가져올 수 있도록 미리 npz를 bin 파일로 변환해둘 것
        if self.load_depth:
            ## TODO -> get_depth 부분, kitti_dataset.py의 polardataset의 get_depth를 호출
            depth_gt = self.get_depth(folder, frame_index)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        # if "s" in self.frame_idxs:
        #     stereo_T = np.eye(4, dtype=np.float32)
        #     baseline_sign = -1 if do_flip else 1
        #     side_sign = -1 if side == "l" else 1
        #     stereo_T[0, 3] = side_sign * baseline_sign * 0.1

        #     inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
