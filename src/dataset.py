import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from os import listdir
from os.path import isfile, join
from utils_functions import scale_array_to_range, set_seed
import ntpath
import configs 


class BrainTissueDataset(Dataset):
    def __init__(self, data_dir_path, img_idx_start, img_idx_stop, axis, transform = None, is_eval = False):
        if img_idx_start > img_idx_stop:
            raise Exception("invalid img range")
        if axis not in ('X', 'Y', 'Z'):
            raise Exception('Axis values can be only: X, Y, Z')

        self.data_dir_path = data_dir_path
        self.img_idx_start = img_idx_start
        self.img_idx_stop = img_idx_stop
        self.axis = axis
        self.transfrom = transform
        self.is_eval = is_eval

        self.X_img_paths = []
        self.Y_mask_paths = []

        for idx in range(img_idx_start, img_idx_stop+1):
            volumne_dir = os.path.join(self.data_dir_path, f"sub-feta{str(idx).zfill(configs.SLICE_DECIMATE_IDENTIFIER)}", axis)
            slice_img_paths = [os.path.join(volumne_dir, f) for f in listdir(volumne_dir) if isfile(join(volumne_dir, f))]
            self.X_img_paths += slice_img_paths
            
            mask_dir = os.path.join(self.data_dir_path, f"sub-feta{str(idx).zfill(configs.SLICE_DECIMATE_IDENTIFIER)}seg", axis)
            slice_mask_paths = [os.path.join(mask_dir, f) for f in listdir(mask_dir) if isfile(join(mask_dir, f))]
            self.Y_mask_paths += slice_mask_paths

    def __len__(self):
        return len(self.X_img_paths)

    def __getitem__(self, index):
        x = Image.open(self.X_img_paths[index]) # pil image in range [0, 255]
        y = Image.open(self.Y_mask_paths[index])  # pil image in range [0, 255]

        seed = np.random.randint(2147483647) 
        if self.transfrom is not None:
            set_seed(seed)
            x = self.transfrom(x)
            set_seed(seed)
            y = self.transfrom(y)
        
        x = transforms.ToTensor()(x) # tranasforms pil image in range [0, 255] to tensor in range [0, 1.0]
        y = torch.from_numpy(np.round(scale_array_to_range(np.array(y), 0, 255, 0, 7)).astype('uint8')) # y is in range [0, 255] so first scale it to range {0, 1, ..., 7}, round in order to get proper mask values, cast to uint8, and then transform to tensor
        y = y.type(torch.LongTensor)

        if self.is_eval:
            return x, y, ntpath.basename(self.X_img_paths[index])
        return x, y

