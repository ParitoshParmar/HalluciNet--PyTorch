# Author: Paritosh

import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
from opts import *

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True

def image_hori_flipper():
    return


def load_image(image_path, hori_flip, transform=None, net=None):
    image = Image.open(image_path)
    if net == 'alexnet':
        input_resize = int(alexnet_size[0]), int(alexnet_size[1])
    elif net == 'c3d':
        input_resize = int(c3d_size[0]), int(c3d_size[1])
    else:
        print('Net not specified.')
    interpolator_idx = random.randint(0,3)
    # print('Interpolator: ', interpolator_idx)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(input_resize, interpolator)
    if hori_flip:
        image.transpose(Image.FLIP_LEFT_RIGHT)
    # print('image shpae', image.size())
    # image.show()
    if transform is not None:
        if net == 'alexnet':
            image = transform(image)
        elif net == 'c3d':
            image = transform(image).unsqueeze(0)
        else:
            print('Net not specified.')
    return image


class VideoDataset_2(Dataset):
    def __init__(self, mode):
        super(VideoDataset_2, self).__init__()
        self.n_frames = 16
        self.mode = mode
        if mode == 'train':
            self.set = open('/home/paritosh/pytorch_tutos/aar/ucf101_splits/ucfTrainTestlist/trainlist01.txt', 'r').read().splitlines()
        if mode == 'test':
            self.set = open('/home/paritosh/pytorch_tutos/aar/ucf101_splits/ucfTrainTestlist/testlist01_w_labels.txt', 'r').read().splitlines()

    def __getitem__(self, ix_0):
        transform_alexnet = transforms.Compose([transforms.CenterCrop(alexnet_H),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])
        transform_c3d = transforms.Compose([transforms.CenterCrop(c3d_H),
                                            transforms.ToTensor(),
                                            # transforms.Normalize(mean=[114.7748 / 255, 107.7354 / 255, 99.4750 / 255],
                                            #                      std=[38.7568578 / 255, 37.88248729 / 255, 40.02898126 / 255])
                                            transforms.Normalize(mean=[114.7748 / 255, 107.7354 / 255, 99.4750 / 255],
                                                                 std=[1,1,1])
                                            # transforms.Normalize(mean=[110.63666788 / 255, 103.16065604 / 255, 96.29023126 / 255],
                                            #                      std=[1, 1, 1])
                                            # transforms.Normalize(mean=[255, 255, 255],
                                            #                      std=[38.7568578 / 255, 37.88248729 / 255, 40.02898126 / 255])
        ])

        # first appearance-video pair
        hori_flip = random.randint(0, 1)

        image_alexnet_0 = torch.zeros(self.n_frames, alexnet_C, alexnet_H, alexnet_W)
        images_c3d_0 = torch.zeros(self.n_frames, c3d_C, c3d_H, c3d_W)

        clip_0 = self.set[ix_0].split(' ')
        # print('CLip: ', clip_0)
        clip_dir_0 = '/home/paritosh/Important/UCF-101/' + clip_0[0].split('.')[0]
        # print('CLip_dir: ', clip_dir_0)

        label_0 = int(clip_0[1])
        if self.mode == 'train':
            label_0 = label_0 - 1
        # print('labels: ', label_0)
        image_list_0 = sorted((glob.glob(os.path.join(clip_dir_0, '*.jpg'))))
        # print('IMg lst: ', image_list_0)

        # if len(image_list_0) > 32:
        #     start_frame_0 = random.randint(16, len(image_list_0) - 16)
        # else:
        #     start_frame_0 = 0
        start_frame_0 = int((len(image_list_0))/2)

        # alexnet image
        image_number_0 = start_frame_0
        image_alexnet_0 = load_image(image_list_0[image_number_0], hori_flip, transform_alexnet, 'alexnet')
        # c3d_clip
        if start_frame_0 + self.n_frames >= len(image_list_0):
            start_frame_0 = len(image_list_0) - (self.n_frames + 1)
        for i in np.arange(start_frame_0, start_frame_0 + self.n_frames):
            images_c3d_0[i - start_frame_0] = load_image(image_list_0[i], hori_flip, transform_c3d, 'c3d')
        # images_c3d_0 = images_c3d_0.transpose(2,3).contiguous()
        data = {}
        data['alexnet_image_0'] = image_alexnet_0;
        data['video_0'] = images_c3d_0;
        data['label_0'] = label_0
        # if self.mode == 'train':
        #     data['alexnet_image_1'] = image_alexnet_1; data['video_1'] = images_c3d_1; data['label_1'] = label_1

        return data

    def __len__(self):
        return len(self.set)