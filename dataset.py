import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from scipy.io import loadmat


class Dataset(data.Dataset):
    def __init__(self, iseval, dataratio=1.0):

        self.eval = iseval

        # mnist..
        data = loadmat('data/mnist/mnist32_train.mat')
        self.datalist_src = [{
                                'image': data['X'][ij],
                                'label': int(data['y'][0][ij])
        } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]

        # svhn.
        # number 0 maps to label 10, fix that here
        data = loadmat('data/svhn/train_32x32.mat')
        self.datalist_target = [{
                                'image': data['X'][..., ij],
                                'label': int(data['y'][ij][0]) if int(data['y'][ij][0]) < 10 else 0
        } for ij in range(data['y'].shape[0]) if np.random.rand() <= dataratio]

        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.source_larger = len(self.datalist_src) > len(self.datalist_target)
        self.n_smallerdataset = len(self.datalist_target) if self.source_larger else len(self.datalist_src)

    def __len__(self):
        return np.maximum(len(self.datalist_src), len(self.datalist_target))

    def shuffledata(self):
        self.datalist_src = [self.datalist_src[ij] for ij in torch.randperm(len(self.datalist_src))]
        self.datalist_target = [self.datalist_target[ij] for ij in torch.randperm(len(self.datalist_target))]

    def __getitem__(self, index):

        index_src = index if self.source_larger else index % self.n_smallerdataset
        index_target = index if not self.source_larger else index % self.n_smallerdataset

        image_source = self.datalist_src[index_src]['image']
        image_source = self.totensor(image_source)
        image_source = self.normalize(image_source)

        image_target = self.datalist_target[index_target]['image']
        image_target = self.totensor(image_target)
        image_target = self.normalize(image_target)

        return image_source, self.datalist_src[index_src]['label'], image_target, self.datalist_target[index_target]['label']


class Dataset_eval(data.Dataset):
    def __init__(self):

        # svhn.
        # number 0 maps to label 10, fix that here
        data = loadmat('data/svhn/test_32x32.mat')
        self.datalist_target = [{
                                'image': data['X'][..., ij],
                                'label': int(data['y'][ij][0]) if int(data['y'][ij][0]) < 10 else 0
        } for ij in range(data['y'].shape[0])]

        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.datalist_target)

    def __getitem__(self, index):

        image_target = self.datalist_target[index]['image']
        image_target = self.totensor(image_target)
        image_target = self.normalize(image_target)

        return image_target, self.datalist_target[index]['label']


def GenerateIterator(args, iseval=False):
    params = {
        'pin_memory': True,
        'batch_size': args.batch_size if not iseval else args.batch_size_eval,
        'shuffle': True,
        'num_workers': args.workers,
        'drop_last': True,
    }

    return data.DataLoader(Dataset(iseval), **params)


def GenerateIterator_eval(args):
    params = {
        'pin_memory': True,
        'batch_size': args.batch_size_eval,
        'num_workers': args.workers,
    }

    return data.DataLoader(Dataset_eval(), **params)
