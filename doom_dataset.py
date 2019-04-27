from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os import listdir
from os.path import isfile, join
import os
import re


class DoomDataset(Dataset):

    def __init__(self, root_dir, classes=-1, train_rnn=False, transform=None):
        """
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = [f for f in listdir(self.root_dir)]
        self.classes = classes
        self.train_rnn = train_rnn

    def __len__(self):
        # length = 0
        # for f in self.filenames:
        #     frame_range_str = f[f.find('[') + 1 : f.find(']')]
        #     frame_range = [int(x) for x in frame_range_str.split(',')]
        #     frame_range_high = frame_range[1]
        #     if frame_range_high > length:
        #         length = frame_range_high
        #
        # return length + 1
        return len(self.filenames)

    def __getitem__(self, idx):
        # for f in self.filenames:
        #     frame_range_str = f[f.find('[') + 1: f.find(']')]
        #     frame_range = [int(x) for x in frame_range_str.split(',')]
        #     if frame_range[0] <= idx <= frame_range[1]:
        #         data = np.load(os.path.join(self.root_dir, f))
        #         image = data['obs.npy'][idx - frame_range[0]].astype(np.float) / 255.0
        #         return np.moveaxis(image, -1, 0)
        #
        # raise IndexError('Index out of range.')
        # data = np.load(os.path.join(self.root_dir, self.filenames[idx]))
        # image = (data / 255.0).astype(np.float32)
        # return np.moveaxis(image, -1, 0)
        data = np.load(os.path.join(self.root_dir, self.filenames[idx]))
        images = (data['obs'] / 255.0).astype(np.float32)
        images = np.moveaxis(images, -1, 1)
        if not self.train_rnn:
            return images
        else:
            obs = images[:-1]
            next_obs = images[1:]
            acs = data['action']
            one_hot_acs = np.eye(2)[acs].astype(np.float32)
            return {'obs': obs, 'next_obs': next_obs, 'acs': one_hot_acs}
