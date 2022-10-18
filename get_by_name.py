import torch
import torch.nn as nn

from dataset_and_model.mini_imagenet_dataset_loader import MiniImagenetLoader
from dataset_and_model.mnist_dataset_loader import MnistLoader
from dataset_and_model.onmiglot_dataset_loader import OmniglotLoader
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
        pass  # TODO


def get_algorithm_by_name(algorithm_name, args, dataset):
    loss = nn.CrossEntropyLoss(reduction='mean')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if algorithm_name == "train-on-test":
        return TrainOnTestLearner(args.per_task_lr, args.test_set_mult, loss,
                                  device, args.seed, args.n_ways, dataset.get_deterministic_model())
    elif algorithm_name == "maml":
        return MamlMetaLearner(args.per_task_lr, args.meta_lr, args.train_adapt_steps, args.test_adapt_steps,
                               args.meta_batch_size,
                               dataset.get_deterministic_model(), loss, device, args.seed, args.n_ways,
                               args.test_set_mult)
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
