

# code from https://github.com/cnguyen10/few_shot_meta_learning/blob/master/Vampire2.py
import math

import higher
import typing

from meta_learning.vampire.HyperNetClasses import NormalVariationalNet
from meta_learning.vampire.MLBaseClass import MLBaseClass
from meta_learning.vampire.Maml import Maml

import os

import torch
import torch.utils.data

from meta_learning.base_meta_learner import BaseMetaLearner
from meta_learning.vampire._utils import train_val_split


class MetaAdaptationMetaLearner(BaseMetaLearner):
    def __init__(self, per_task_lr, meta_lr, kl_weight, f_loss, train_adapt_steps, test_adapt_steps,
                 meta_batch_size,
                 device, seed, n_ways, k_shots, num_models, num_models_test,
                 step_epochs, scheduler, adaptive_kl_factor, hyper_kl_factor,
                 data_loader=None, dataset_name=None, args_hash=""):
        self.train_adapt_steps = train_adapt_steps
        config = {
            'resume_epoch': 0,
            'logdir': "artifacts/vi" if not dataset_name else f"artifacts/{dataset_name}/vi/{args_hash}",
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
            'step_epochs': step_epochs,
            'test_hyper_kl': hyper_kl_factor,
            'test_adaptive_kl': adaptive_kl_factor,
            'meta_adaptation_mode': False,
        }
        self.meta_learner = MetaAdaptationClass(config)
        self.epoch = 0
        self.loss = f_loss
        self.device = device
        self.data_loader = data_loader
        self.num_models_test = num_models_test

    def meta_train(self, train_taskset, validation_taskset, n_epochs):
        self.data_loader = train_taskset
        self.meta_learner.config['num_epochs'] = n_epochs
        self.meta_learner.config['num_inner_updates'] = self.train_adapt_steps
        self.meta_learner.train(train_dataloader=train_taskset, val_dataloader=validation_taskset)

    def meta_test_on_task(self, D_task_xs_adapt, D_task_ys_adapt, D_task_xs_error_eval, D_task_ys_error_eval, n_epochs):
        self.meta_learner.config['num_models'] = self.num_models_test
        self.load_saved_model(None) # for early stopping

        old_shots = self.meta_learner.config["k_shot"]
        if n_epochs > 0:
            self.meta_learner.config['meta_adaptation_mode'] = True
            self.meta_learner.config['train_flag'] = True
            self.meta_learner.config["resume_epoch"] = self.meta_learner.config["num_epochs"]
            self.meta_learner.config["num_epochs"] += n_epochs

            self.meta_learner.config["k_shot"] = self.meta_learner.config["k_shot"] // 2
            dataset = torch.utils.data.TensorDataset(D_task_xs_adapt.to("cpu"), D_task_ys_adapt.to("cpu"))
            meta_adapt_loader = torch.utils.data.DataLoader(dataset, batch_size=len(D_task_ys_adapt), shuffle=True)
            self.meta_learner.train(train_dataloader=meta_adapt_loader, val_dataloader=None)
            self.meta_learner.config['meta_adaptation_mode'] = False

        model = self.meta_learner.load_model(resume_epoch=self.meta_learner.config["num_epochs"],
                                             hyper_net_class=self.meta_learner.hyper_net_class,
                                             eps_dataloader=self.data_loader)
        self.meta_learner.config['train_flag'] = False
        self.meta_learner.config["k_shot"] = old_shots
        self.meta_learner.config['num_inner_updates'] = self.meta_learner.config['num_episodes']
        loss, acc = self.meta_learner.evaluation(D_task_xs_adapt, D_task_ys_adapt, D_task_xs_error_eval,
                                                 D_task_ys_error_eval, model=model)
        return loss, acc/100.0, None, None

    def load_saved_model(self, model_name):
        highest_epoch = 0
        for path, dirs, files in os.walk(self.meta_learner.config["logdir"]):
            for file_name in files:
                if "Epoch" not in file_name:
                    continue
                file_epoch_num = int(file_name.split("_")[1].split(".")[0])
                if file_epoch_num > highest_epoch:
                    highest_epoch = file_epoch_num
        self.meta_learner.config["num_epochs"] = highest_epoch
        # done, auto-load

    def save_model(self, model_name):
        pass  # auto-saved each epoch


class MetaAdaptationClass(MLBaseClass):
    def __init__(self, config: dict) -> None:
        super().__init__(config=config)

        self.hyper_net_class = NormalVariationalNet

    def load_model(self, resume_epoch: int, eps_dataloader: torch.utils.data.DataLoader, **kwargs) -> dict:
        maml_temp = Maml(config=self.config)
        return maml_temp.load_model(resume_epoch=resume_epoch, eps_dataloader=eps_dataloader, **kwargs)

    def adaptation(self, x: torch.Tensor, y: torch.Tensor, model: dict) -> higher.patch._MonkeyPatchBase:
        # convert hyper_net to its functional form
        f_hyper_net = higher.patch.monkeypatch(
            module=model["hyper_net"],
            copy_initial_weights=False,
            track_higher_grads=self.config["train_flag"]
        )

        p_params = [params.detach() for params in f_hyper_net.fast_params]

        for _ in range(self.config['num_inner_updates']):
            q_params = f_hyper_net.fast_params  # parameters of the task-specific hyper_net

            grads_accum = [0] * len(q_params)  # accumulate gradients of Monte Carlo sampling

            # KL divergence
            KL_loss = self.KL_divergence(p=p_params, q=q_params)
            delta = 0.1
            n_samples = len(y)
            KL_complexity = torch.sqrt((KL_loss + math.log(2 * n_samples * self.config['num_models'] / delta)) / (2 * (n_samples - 1)))

            for _ in range(self.config['num_models']):
                # generate parameter from task-specific hypernet
                base_net_params = f_hyper_net.forward()

                y_logits = model["f_base_net"].forward(x, params=base_net_params)

                cls_loss = self.config['loss_function'](input=y_logits, target=y)

                #loss = cls_loss + self.config['KL_weight'] * KL_loss
                loss = cls_loss + self.config['KL_weight'] * KL_complexity

                if self.config['first_order']:
                    grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=q_params,
                        retain_graph=True
                    )
                else:
                    grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=q_params,
                        create_graph=True
                    )

                # accumulate gradients from Monte Carlo sampling and average out
                for i in range(len(grads)):
                    grads_accum[i] = grads_accum[i] + grads[i] / self.config['num_models']

            new_q_params = []
            for param, grad in zip(q_params, grads_accum):
                new_q_params.append(higher.optim._add(tensor=param, a1=-self.config['inner_lr'], a2=grad))

            p_params = [params.detach() for params in f_hyper_net.fast_params]
            f_hyper_net.update_params(new_q_params)

        return f_hyper_net

    def prediction(self, x: torch.Tensor, adapted_hyper_net: higher.patch._MonkeyPatchBase, model: dict) -> typing.List[
        torch.Tensor]:

        logits = [None] * self.config['num_models']
        for model_id in range(self.config['num_models']):
            # generate parameter from task-specific hypernet
            base_net_params = adapted_hyper_net.forward()

            logits_temp = model["f_base_net"].forward(x, params=base_net_params)

            logits[model_id] = logits_temp

        return logits

    def validation_loss(self, x: torch.Tensor, y: torch.Tensor, adapted_hyper_net: higher.patch._MonkeyPatchBase,
                        model: dict) -> torch.Tensor:

        logits = self.prediction(x=x, adapted_hyper_net=adapted_hyper_net, model=model)

        loss = 0

        # classification loss
        for logits_ in logits:
            loss = loss + self.config['loss_function'](input=logits_, target=y)

        loss = loss / len(logits)

        # KL loss
        KL_loss = self.KL_divergence_standard_normal(p=adapted_hyper_net.fast_params)
        loss = loss + self.config["test_hyper_kl"] * KL_loss

        if self.config['meta_adaptation_mode']:
            f_hyper_net = higher.patch.monkeypatch(
                module=model["hyper_net"],
                copy_initial_weights=False,
                track_higher_grads=self.config["train_flag"]
            )

            p_params = [params.detach() for params in f_hyper_net.fast_params]
            adaptive_KL_loss = self.KL_divergence(p=p_params, q=adapted_hyper_net.fast_params)
            loss = loss + self.config["test_adaptive_kl"] * adaptive_KL_loss

        return loss

    def evaluation(self, x_t: torch.Tensor, y_t: torch.Tensor, x_v: torch.Tensor, y_v: torch.Tensor, model: dict) -> \
    typing.Tuple[float, float]:

        adapted_hyper_net = self.adaptation(x=x_t, y=y_t, model=model)

        logits = self.prediction(x=x_v, adapted_hyper_net=adapted_hyper_net, model=model)

        loss = self.validation_loss(x=x_v, y=y_v, adapted_hyper_net=adapted_hyper_net, model=model)

        y_pred = 0
        for logits_ in logits:
            y_pred = y_pred + torch.softmax(input=logits_, dim=1)

        y_pred = y_pred / len(logits)

        accuracy = (y_pred.argmax(dim=1) == y_v).float().mean().item()

        return loss.item(), accuracy * 100


    @staticmethod
    def KL_divergence(p, q):
        KL_div = 0

        n = len(p) // 2

        for i in range(n):
            p_mean = p[i]
            p_log_std = p[n + i]
            p_var = torch.exp(2 * p_log_std)
            q_mean = q[i]
            q_log_std = q[n + i]
            q_var = torch.exp(2 * q_log_std)

            numerator = (q_mean - p_mean).pow(2) + q_var
            denominator = p_var
            div_elem = 0.5 * torch.sum(p_log_std - q_log_std + numerator / denominator - 1)
            KL_div += div_elem

        return KL_div

    @staticmethod
    def KL_divergence_standard_normal(p: typing.List[torch.Tensor], sigma=1e2) -> typing.Union[torch.Tensor, float]:
        """Calculate KL divergence between a diagonal Gaussian with N(0, \sigma^2 * I)
        """
        KL_div = 0

        n = len(p) // 2

        for i in range(n):
            p_mean = p[i]
            p_log_std = p[n + i]
            numerator = torch.square(input=p_mean) + torch.exp(input=2 * p_log_std)
            denominator = sigma**2
            div_elem = 0.5 * torch.sum(math.log(sigma) - p_log_std + numerator / denominator - 1)
            KL_div += div_elem

        return KL_div