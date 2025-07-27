import os
import numpy as np
from torch.utils.data import Dataset
import h5py
import cv2
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils import data

INTERPOLATION = TF.InterpolationMode.BILINEAR 
LOW, HIGH = -1000, 2800

class MAPairedDataset(Dataset):
    def __init__(self, data_root: str, imgsize: int = 0):
        super().__init__()
        self.in_folder = data_root
        # self.namelist = [os.path.join(fold, name) for fold in os.listdir(data_root) for name in os.listdir(os.path.join(data_root, fold) ) if name.endswith(".h5")]
        self.namelist = get_all_files(self.in_folder)
        self.imgsize = imgsize
        
    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        h5_file = os.path.join(self.in_folder, self.namelist[idx])
        with h5py.File(h5_file, mode='r') as f:
            data_gt = np.asarray(f['gt_CT']) # f['gt_CT'] with shape (H, W)
            data_ma = np.asarray(f['ma_CT']) # f['ma_CT'] with shape (1, H, W)
            data_metalmask = np.asarray(f['metal_mask']) # f['metal_mask'] with shape (1, H, W)
            data_ma = data_ma[0]
            data_metalmask = data_metalmask[0]


        def preprocess(img):
            img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
            
            minmaxval = get_minmax()
            img = normalize(img, minmaxval)
            img = adjust_scale(img)
            img = (img * 255).round()
            img = np.array([img, img, img]).transpose([1,2,0])
            img = np.uint8(img)
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(img)
            return img
        
        data_gt = preprocess(data_gt)
        data_ma = preprocess(data_ma)

        name = self.namelist[idx].replace('/', '-')
        name = name.replace('.h5', '.jpg')
        data = {
            "data_gt": data_gt,
            "data_ma": data_ma,
            "name" : name,
        }
        return data


class MAClinicalDataset(Dataset):
    def __init__(self, data_root: str, imgsize: int = 0):
        super().__init__()
        self.in_folder = data_root
        self.namelist = get_all_files(self.in_folder)
        self.imgsize = imgsize
        
    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        h5_file = os.path.join(self.in_folder, self.namelist[idx])
        with h5py.File(h5_file, mode='r') as f:
            data_ma = np.asarray(f['ma_CT']) # f['ma_CT'] with shape (1, H, W)
            data_metalmask = np.asarray(f['metal_mask']) # f['metal_mask'] with shape (1, H, W)
            data_ma = data_ma[0]
            data_metalmask = data_metalmask[0]


        def preprocess(img):
            img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
            
            minmaxval = get_minmax()
            img = normalize(img, minmaxval)
            img = adjust_scale(img)
            img = (img * 255).round()
            img = np.array([img, img, img]).transpose([1,2,0])
            img = np.uint8(img)
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(img)
            return img
        
        data_ma = preprocess(data_ma)
        data_metalmask = preprocess(data_metalmask)

        name = self.namelist[idx].replace('/', '-')
        name = name.replace('.h5', '.jpg')
        data = {
            "data_metalmask": data_metalmask,
            "data_ma": data_ma,
            "name" : name,
        }
        return data


def adjust_ct_window(imgdata: np.ndarray, minval, maxval, inplace=True):
	if inplace:
		data = imgdata
	else:
		data = imgdata.copy()
	data[data < minval] = minval
	data[data > maxval] = maxval

	return data

def adjust_scale(data):
    # [-1, 1] to [0, 1]
    return data * 0.5 + 0.5

def get_all_files(folder: str):
    # name does not contain the root folder
    filelist = []
    for root, _, names in os.walk(folder, followlinks=True):
        folder_prefix = root[len(folder)+1:]
        for name in names:
            if name.endswith(".h5"):
                if len(folder_prefix) > 0:
                    filename = folder_prefix + '/' + name
                else:
                    filename = name
                filelist.append(filename)
    return filelist

def get_minmax():
    return 0.0, 0.73

def normalize(data, minmax):
    minval, maxval = minmax
    data = np.clip(data, minval, maxval)
    data = (data - minval) / (maxval - minval)
    data = data * 2.0 - 1.0
    return data

def denormalize(data, minmax):
    minval, maxval = minmax
    data = data * 0.5 + 0.5
    data = data * (maxval - minval) + minval
    return data

