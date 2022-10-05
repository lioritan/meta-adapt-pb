from dataset_and_model.base_dataset_loader import BaseDatasetLoader
import learn2learn as l2l
from learn2learn.data.task_dataset import TaskDataset  # object for train/test set, contains metadataset & transforms
from learn2learn.data.meta_dataset import MetaDataset
from learn2learn.data.transforms import *  # nways, kshots, loaddata, remaplabels, consequtivelabels
import models.stochastic_models
from torchvision import datasets, transforms
import os
import torch
import numpy as np

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


class MnistLoader(BaseDatasetLoader):
    def __init__(self, n_ways, test_mult=2, data_path='~/data',
                 permute_labels=False,
                 n_pixels_to_change_train=0,
                 n_pixels_to_change_test=0):
        self.n_ways = n_ways
        self.test_mult = test_mult
        self.data_path = data_path
        self.permute_labels = permute_labels
        self.n_pixels_to_change_train = n_pixels_to_change_train
        self.n_pixels_to_change_test = n_pixels_to_change_test
        self.mnist_data = load_mnist(self.data_path)
        trn_size = len(self.mnist_data[0])
        frac = 10
        trn, val = torch.utils.data.random_split(self.mnist_data[0], [(frac-1)*trn_size//frac, trn_size//frac],
                                                 generator=torch.Generator().manual_seed(42))
        self.mnist_data = (trn, val, self.mnist_data[1])


    def get_deterministic_model(self):
        return l2l.vision.models.OmniglotCNN(self.n_ways)

    def get_name(self):
        return "mnist"

    def get_stochastic_model(self):
        log_var_init = {"mean": -10, "std": 0.1}
        model = models.stochastic_models.get_model("omniglot", log_var_init, input_shape=MNIST_SHAPE,
                                                   output_dim=self.n_ways)
        return model

    def get_taskset(self, n_ways, n_shots, meta_dataset, pixels):
        return TaskDataset(meta_dataset, task_transforms=[
            l2l.data.transforms.FusedNWaysKShots(meta_dataset,
                                                 n=n_ways,
                                                 k=n_shots),
            l2l.data.transforms.LoadData(meta_dataset),
            ShufflePixels(meta_dataset, pixels),
            l2l.data.transforms.RemapLabels(meta_dataset),
            l2l.data.transforms.ConsecutiveLabels(meta_dataset),
            PermuteLabels(meta_dataset, n_ways, self.permute_labels),
        ], num_tasks=-1)

    def train_taskset(self, n_ways, n_shots):
        meta_train = MetaDataset(self.mnist_data[0])
        return self.get_taskset(n_ways, n_shots, meta_train, self.n_pixels_to_change_train)

    def validation_taskset(self, n_ways, n_shots):
        meta_val = MetaDataset(self.mnist_data[1])
        return self.get_taskset(n_ways, n_shots, meta_val, self.n_pixels_to_change_train)

    def test_taskset(self, n_ways, n_shots):
        meta_test = MetaDataset(self.mnist_data[2])
        return self.get_taskset(n_ways, self.test_mult * n_shots, meta_test, self.n_pixels_to_change_test)
