import os

import torch

from meta_learning.base_meta_learner import BaseMetaLearner
from meta_learning.vampire.Vampire2 import Vampire2
from meta_learning.vampire._utils import train_val_split


class VampireMetaLearner(BaseMetaLearner):
    def __init__(self, per_task_lr, meta_lr, kl_weight, f_loss, train_adapt_steps, test_adapt_steps,
                 meta_batch_size,
                 device, seed, n_ways, k_shots, num_models, num_models_test,
                 step_epochs, scheduler,
                 data_loader=None, dataset_name=None):
        self.train_adapt_steps = train_adapt_steps
        config = {
            'resume_epoch': 0,
            'logdir': "artifacts/vampire" if not dataset_name else f"artifacts/{dataset_name}/vampire/{seed}",
            'minibatch_print': -1,  # !
            'num_episodes_per_epoch': meta_batch_size,  # number of meta updates per epoch
            'train_val_split_function': train_val_split,
            'k_shot': k_shots,
            'device': device,
            'minibatch': meta_batch_size,

            'num_episodes': test_adapt_steps,  # validation param

            'train_flag': True,
            'num_inner_updates': train_adapt_steps,
            'num_models': num_models,  # MC samples
            'loss_function': f_loss,
            'KL_weight': kl_weight,
            'first_order': True,
            'inner_lr': per_task_lr,
            'meta_lr': meta_lr,

            'network_architecture': 'CNN',
            'num_ways': n_ways,
            'batchnorm': True,
            'strided': True,

            'scheduler': scheduler,
            'step_epochs': step_epochs
        }
        self.vampire = Vampire2(config)
        self.epoch = 0
        self.loss = f_loss
        self.device = device
        self.data_loader = data_loader
        self.num_models_test = num_models_test

    def meta_train(self, train_taskset, validation_taskset, n_epochs):
        self.data_loader = train_taskset
        self.vampire.config['num_epochs'] = n_epochs
        self.vampire.config['num_inner_updates'] = self.train_adapt_steps
        self.vampire.train(train_dataloader=train_taskset, val_dataloader=validation_taskset)

    def meta_test_on_task(self, D_task_xs_adapt, D_task_ys_adapt, D_task_xs_error_eval, D_task_ys_error_eval, n_epochs):
        self.vampire.config['num_inner_updates'] = n_epochs
        self.vampire.config['num_models'] = self.num_models_test
        self.vampire.config['train_flag'] = False
        model = self.vampire.load_model(resume_epoch=self.vampire.config["num_epochs"],
                                        hyper_net_class=self.vampire.hyper_net_class,
                                        eps_dataloader=self.data_loader)
        loss, acc = self.vampire.evaluation(D_task_xs_adapt, D_task_ys_adapt, D_task_xs_error_eval,
                                            D_task_ys_error_eval, model=model)
        return loss, acc/100.0, None, None

    def load_saved_model(self, model_name):
        highest_epoch = 0
        for path, dirs, files in os.walk(self.vampire.config["logdir"]):
            for file_name in files:
                if "Epoch" not in file_name:
                    continue
                file_epoch_num = int(file_name.split("_")[1].split(".")[0])
                if file_epoch_num > highest_epoch:
                    highest_epoch = file_epoch_num
        self.vampire.config["num_epochs"] = highest_epoch
        # done, auto-load

    def save_model(self, model_name):
        pass  # auto-saved each epoch
