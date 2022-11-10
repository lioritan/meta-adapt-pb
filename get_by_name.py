import subprocess
import traceback

import torch
import torch.nn as nn
import numpy as np

from dataset_and_model.mini_imagenet_dataset_loader import MiniImagenetLoader
from dataset_and_model.mnist_dataset_loader import MnistLoader
from dataset_and_model.onmiglot_dataset_loader import OmniglotLoader
from dataset_and_model.tiered_imagenet_dataset_loader import TieredImagenetLoader
from meta_learning.bayesian_vi import BayesianVI
from meta_learning.maml import MamlMetaLearner
from meta_learning.meta_adaptation import MetaAdaptation
from meta_learning.train_on_test import TrainOnTestLearner
from meta_learning.vampire.vampire_wrapper import VampireMetaLearner


def get_dataset_by_name(dataset_name, args):
    if dataset_name == "mini-imagenet":
        return MiniImagenetLoader(args.n_ways, args.test_set_mult)
    elif dataset_name == "mnist":
        return MnistLoader(args.n_ways, args.test_set_mult, permute_labels=args.mnist_permute_labels,
                           n_pixels_to_change_train=args.mnist_pixels_to_permute_train,
                           n_pixels_to_change_test=args.mnist_pixels_to_permute_test)
    elif dataset_name == "omniglot":
        return OmniglotLoader(args.n_ways, args.test_set_mult)
    elif dataset_name == "tiered-imagenet":
        return TieredImagenetLoader(args.n_ways, args.test_set_mult)


def get_free_gpu():
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        # smi query lists all gpu devices and the memory usage, grep filters shared bar and such and only lists lines until the used memory one
        smi_result = subprocess.check_output("nvidia-smi -q -d Memory | grep -A4 GPU", shell=True)
        gpu_info = smi_result.decode("utf-8").split("\n")
        gpu_info = [int(line.split(":")[1].replace("MiB", "").strip()) for line in gpu_info if "Used" in line]
        sufficiently_free_gpus = [used_mbs if used_mbs < (2**12) else 2**20 for used_mbs in gpu_info if used_mbs < (2**12)]
        print(sufficiently_free_gpus)
        least_used_gpu = np.argmin(sufficiently_free_gpus) if min(sufficiently_free_gpus) < 2**12 else 0
        print(least_used_gpu)
        return f'cuda:{least_used_gpu}'
    except subprocess.CalledProcessError as e:
        traceback.print_exc()
        return 'cuda:0'


def get_algorithm_by_name(algorithm_name, args, dataset):
    loss = nn.CrossEntropyLoss(reduction='mean')
    device = torch.device(get_free_gpu())
    if algorithm_name == "train-on-test":
        return TrainOnTestLearner(args.per_task_lr, args.test_set_mult, loss,
                                  device, args.seed, args.n_ways, dataset.get_deterministic_model(),
                                  args.optimizer_weight_decay)
    elif algorithm_name == "maml":
        return MamlMetaLearner(args.per_task_lr, args.meta_lr, args.train_adapt_steps, args.test_adapt_steps,
                               args.meta_batch_size,
                               dataset.get_deterministic_model(), loss, device, args.seed, args.n_ways,
                               args.test_set_mult, args.optimizer_weight_decay, args.optimizer_lr_decay_epochs,
                               args.optimizer_lr_schedule_type, args.early_stop)
    elif algorithm_name == "bayesian-vi":
        return BayesianVI(args.per_task_lr, args.meta_lr, args.train_adapt_steps, args.test_adapt_steps,
                          args.meta_batch_size,
                          loss, device, args.seed, args.n_ways, dataset.get_stochastic_model(),
                          lambda x: dataset.get_stochastic_model(),
                          args.test_set_mult)
    elif algorithm_name == "meta-adaptation":
        return MetaAdaptation(args.per_task_lr, args.meta_lr, args.train_adapt_steps, args.test_adapt_steps,
                              args.meta_batch_size,
                              loss, device, args.seed, args.n_ways, dataset.get_stochastic_model(),
                              lambda x: dataset.get_stochastic_model(),
                              args.test_set_mult, args.meta_adaptation_is_adaptive)
    elif algorithm_name == "vampire":
        kl_weight = args.vampire_kl_weight
        num_models = args.vampire_num_models
        data_loader = get_dataset_by_name(args.dataset, args).train_taskset(args.n_ways, args.n_shots)
        return VampireMetaLearner(args.per_task_lr, args.meta_lr, kl_weight, loss, args.train_adapt_steps,
                                  args.test_adapt_steps, args.meta_batch_size, device, args.seed, args.n_ways,
                                  args.n_shots, num_models, data_loader=data_loader, dataset_name=args.dataset)
    elif algorithm_name == "bmaml":
        num_models = args.bmaml_num_particles
        data_loader = get_dataset_by_name(args.dataset, args).train_taskset(args.n_ways, args.n_shots)
        return VampireMetaLearner(args.per_task_lr, args.meta_lr, loss, args.train_adapt_steps,
                                  args.test_adapt_steps, args.meta_batch_size, device, args.seed, args.n_ways,
                                  args.n_shots, num_models, data_loader=data_loader, dataset_name=args.dataset)
