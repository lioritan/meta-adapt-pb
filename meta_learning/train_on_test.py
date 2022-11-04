import torch

from meta_learning.base_meta_learner import BaseMetaLearner


class TrainOnTestLearner(BaseMetaLearner):
    def __init__(self, lr, shots_mult, loss_func, device, seed, n_ways, model, weight_decay=0):
        self.device = device
        self.model = model.to(device)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        self.shots_mult = shots_mult
        self.loss_func = loss_func
        self.seed = seed
        self.n_ways = n_ways

    def meta_train(self, train_taskset, validation_taskset, n_epochs):
        pass

    def meta_test_on_task(self, D_task_xs_adapt, D_task_ys_adapt, D_task_xs_error_eval, D_task_ys_error_eval, n_epochs):
        self.model.train()
        for epoch in range(n_epochs):
            outputs = self.model(D_task_xs_adapt)
            objective = self.loss_func(outputs, D_task_ys_adapt)
            self.optimizer.zero_grad()
            objective.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            test_predictions = self.model(D_task_xs_error_eval)
            test_loss = self.loss_func(test_predictions, D_task_ys_error_eval)
            test_acc = BaseMetaLearner.accuracy(test_predictions, D_task_ys_error_eval)
        return test_loss.item(), test_acc.item(), None, None

    def load_saved_model(self, model_name):
        pass

    def save_model(self, model_name):
        pass
