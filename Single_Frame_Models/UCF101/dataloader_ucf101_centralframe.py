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
    if net == 'model_2dcnn':
        input_resize = int(model_2dcnn_size[0]), int(model_2dcnn_size[1])
    elif net == 'model_3dcnn':
        input_resize = int(model_3dcnn_size[0]), int(model_3dcnn_size[1])
    else:
        print('Net not specified.')
    interpolator_idx = random.randint(0,3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(input_resize, interpolator)
    if hori_flip:
        image.transpose(Image.FLIP_LEFT_RIGHT)
    # image.show()
    if transform is not None:
        if net == 'model_2dcnn':
            image = transform(image)
        elif net == 'model_3dcnn':
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
            self.set = open('.../ucf101_splits/ucfTrainTestlist/trainlist01.txt', 'r').read().splitlines()
        if mode == 'test':
            self.set = open('.../ucf101_splits/ucfTrainTestlist/testlist01_w_labels.txt', 'r').read().splitlines()

    
    def __getitem__(self, ix_0):
        transform_model_2dcnn = transforms.Compose([transforms.CenterCrop(model_2dcnn_H),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])
        transform_model_3dcnn = transforms.Compose([transforms.CenterCrop(model_3dcnn_H),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[114.7748 / 255, 107.7354 / 255, 99.4750 / 255],
                                                                 std=[1,1,1])
        ])

        hori_flip = random.randint(0, 1)

        image_model_2dcnn_0 = torch.zeros(self.n_frames, model_2dcnn_C, model_2dcnn_H, model_2dcnn_W)
        images_model_3dcnn_0 = torch.zeros(self.n_frames, model_3dcnn_C, model_3dcnn_H, model_3dcnn_W)

        clip_0 = self.set[ix_0].split(' ')

        # set directory containing UCF-101 dataset (in form of extracted frames)
        clip_dir_0 = '.../UCF-101/' + clip_0[0].split('.')[0]

        label_0 = int(clip_0[1])
        if self.mode == 'train':
            label_0 = label_0 - 1

        image_list_0 = sorted((glob.glob(os.path.join(clip_dir_0, '*.jpg'))))

        start_frame_0 = int((len(image_list_0))/2)

        # model_2dcnn image
        image_number_0 = start_frame_0
        image_model_2dcnn_0 = load_image(image_list_0[image_number_0], hori_flip, transform_model_2dcnn, 'model_2dcnn')
        # model_3dcnn_clip
        if start_frame_0 + self.n_frames >= len(image_list_0):
            start_frame_0 = len(image_list_0) - (self.n_frames + 1)
        for i in np.arange(start_frame_0, start_frame_0 + self.n_frames):
            images_model_3dcnn_0[i - start_frame_0] = load_image(image_list_0[i], hori_flip, transform_model_3dcnn, 'model_3dcnn')

        data = {}
        data['2dcnn_image'] = image_model_2dcnn_0;
        # multiplying input with 255 to make it correclty scaled for ResNeXt3D-101 by KenshoHara
        data['video'] = images_model_3dcnn_0 * float(255.0000);
        data['label'] = label_0

        return data


    def __len__(self):
        return len(self.set)