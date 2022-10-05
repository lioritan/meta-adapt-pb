import abc
import numpy as np
import torch


class BaseMetaLearner(abc.ABC):

    @classmethod
    def accuracy(cls, predictions, targets):
        predictions = predictions.argmax(dim=1).view(targets.shape)
        return (predictions == targets).sum().float() / targets.size(0)

    def split_adapt_eval(self, task_batch, train_frac=2, device=None):
        device = device if device is not None else self.device
        D_task_xs, D_task_ys = task_batch
        D_task_xs, D_task_ys = D_task_xs.to(device), D_task_ys.to(device)
        task_batch_size = D_task_xs.size(0)
        # Separate data into adaptation / evaluation sets - works even if labels are ordered
        adapt_indices = np.zeros(task_batch_size, dtype=bool)
        train_samples = round(task_batch_size / train_frac)
        adapt_indices[np.arange(train_samples) * train_frac] = True
        error_eval_indices = ~adapt_indices
        # numpy -> torch
        adapt_indices = torch.from_numpy(adapt_indices)
        error_eval_indices = torch.from_numpy(error_eval_indices)
        D_task_xs_adapt, D_task_ys_adapt = D_task_xs[adapt_indices], D_task_ys[adapt_indices]
        D_task_xs_error_eval, D_task_ys_error_eval = D_task_xs[error_eval_indices], D_task_ys[error_eval_indices]
        return D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval

    @abc.abstractmethod
    def meta_train(self, train_taskset, validation_taskset, n_epochs):
        pass

    def meta_test(self, test_taskset, n_epochs, test_shots_mult):
        D_test_batch = test_taskset.sample()
        D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval = self.split_adapt_eval(
            D_test_batch, train_frac=test_shots_mult)
        return self.meta_test_on_task(D_task_xs_adapt, D_task_ys_adapt, D_task_xs_error_eval, D_task_ys_error_eval, n_epochs)

    @abc.abstractmethod
    def meta_test_on_task(self, D_task_xs_adapt, D_task_ys_adapt, D_task_xs_error_eval, D_task_ys_error_eval, n_epochs):
        pass

    @abc.abstractmethod
    def load_saved_model(self, model_name):
        pass

    @abc.abstractmethod
    def save_model(self, model_name):
        pass
