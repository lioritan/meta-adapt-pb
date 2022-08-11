from __future__ import absolute_import, division, print_function

import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import os
import numpy as np
from learn2learn.data.task_dataset import TaskDataset  # object for train/test set, contains metadataset & transforms
from learn2learn.data.meta_dataset import MetaDataset
from learn2learn.data.transforms import *  # nays, kshots, loaddata, remaplabels, consequtivelabels
import learn2learn as l2l

# -------------------------------------------------------------------------------------------
#  Task generator class
# -------------------------------------------------------------------------------------------
MNIST_CLASSES = 10
MNIST_SHAPE = (1, 28, 28)


def load_mnist(data_path):
    # Data transformations list:
    transform = [transforms.ToTensor()]

    # Normalize values:
    # Note: original values  in the range [0,1]

    # MNIST_MEAN = (0.1307,)  # (0.5,)
    # MNIST_STD = (0.3081,)  # (0.5,)
    # transform += transforms.Normalize(MNIST_MEAN, MNIST_STD)

    transform += [transforms.Normalize((0.5,), (0.5,))]  # transform to [-1,1]

    root_path = os.path.join(data_path, 'MNIST')

    # Train set:
    train_dataset = datasets.MNIST(root_path, train=True, download=True,
                                   transform=transforms.Compose(transform))

    # Test set:
    test_dataset = datasets.MNIST(root_path, train=False, download=True,
                                  transform=transforms.Compose(transform))

    return train_dataset, test_dataset


def permute_pixels(x, inds_permute):
    ''' Permute pixels of a tensor image'''
    im_ = x[0]
    im_H = im_.shape[1]
    im_W = im_.shape[2]
    input_size = im_H * im_W
    new_x = im_.view(input_size)  # flatten image
    new_x = new_x[inds_permute]
    new_x = new_x.view(1, im_H, im_W)

    return new_x, x[1]


def create_pixel_permute_trans():
    input_shape = MNIST_SHAPE
    input_size = input_shape[0] * input_shape[1] * input_shape[2]
    inds_permute = torch.randperm(input_size)
    transform_func = lambda x: permute_pixels(x, inds_permute)
    return transform_func


def create_limited_pixel_permute_trans(n_pixels_to_change):
    input_shape = MNIST_SHAPE
    input_size = input_shape[0] * input_shape[1] * input_shape[2]
    inds_permute = torch.LongTensor(np.arange(0, input_size))

    for i_shuffle in range(n_pixels_to_change):
        i1 = np.random.randint(0, input_size)
        i2 = np.random.randint(0, input_size)
        temp = inds_permute[i1]
        inds_permute[i1] = inds_permute[i2]
        inds_permute[i2] = temp

    transform_func = lambda x: permute_pixels(x, inds_permute)
    return transform_func


def create_label_permute_trans(n_classes):
    inds_permute = torch.randperm(n_classes)
    transform_func = lambda target: (target[0], inds_permute[target[1]].item())
    return transform_func


class ShufflePixels(TaskTransform):
    def __init__(self, dataset, n_pixels=-1):
        super(ShufflePixels, self).__init__(dataset)
        self.dataset = dataset
        self.n_pixels = n_pixels

    def __call__(self, task_description):
        if task_description is None:
            task_description = self.new_task()
        if self.n_pixels == 0:
            return task_description

        if self.n_pixels == -1:
            shuffle_func = create_pixel_permute_trans()
        else:
            shuffle_func = create_limited_pixel_permute_trans(self.n_pixels)
        for data_description in task_description:
            data_description.transforms.append(shuffle_func)
        return task_description


class PermuteLabels(TaskTransform):
    def __init__(self, dataset, ways, permute=True):
        super(PermuteLabels, self).__init__(dataset)
        self.dataset = dataset
        self.permute = permute
        self.ways = ways

    def __call__(self, task_description):
        if task_description is None:
            task_description = self.new_task()
        if not self.permute:
            return task_description

        label_permutation = create_label_permute_trans(self.ways)
        for data_description in task_description:
            data_description.transforms.append(label_permutation)
        return task_description


class MnistDataset(object):
    def __init__(self, data_path, train_ways, train_shots, test_ways, test_shots, shuffle_pixels=False,
                 permute_labels=False,
                 n_pixels_to_change_train=1, n_pixels_to_change_test=1):
        train_dataset, test_dataset = load_mnist(data_path)
        pixels_to_change_train = n_pixels_to_change_train if shuffle_pixels else 0
        pixels_to_change_test = n_pixels_to_change_test if shuffle_pixels else 0
        meta_train = MetaDataset(train_dataset)
        meta_test = MetaDataset(test_dataset)
        self.train = TaskDataset(meta_train, task_transforms=[
            l2l.data.transforms.FusedNWaysKShots(meta_train,
                                                 n=train_ways,
                                                 k=train_shots),
            l2l.data.transforms.LoadData(meta_train),
            ShufflePixels(meta_train, pixels_to_change_train),
            l2l.data.transforms.RemapLabels(meta_train),
            l2l.data.transforms.ConsecutiveLabels(meta_train),
            PermuteLabels(meta_train, train_ways, permute_labels),
        ], num_tasks=-1)

        self.test = TaskDataset(meta_test, task_transforms=[
            l2l.data.transforms.FusedNWaysKShots(meta_test,
                                                 n=test_ways,
                                                 k=test_shots),
            l2l.data.transforms.LoadData(meta_test),
            ShufflePixels(meta_test, pixels_to_change_test),
            l2l.data.transforms.RemapLabels(meta_test),
            l2l.data.transforms.ConsecutiveLabels(meta_test),
            PermuteLabels(meta_test, test_ways, permute_labels),
        ], num_tasks=-1)
