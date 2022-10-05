import torch
import learn2learn as l2l

from meta_learning.base_meta_learner import BaseMetaLearner

# TODO: regularization? wd?
class MamlMetaLearner(BaseMetaLearner):
    def __init__(self, per_task_lr, meta_lr,
                 train_adapt_steps, test_adapt_steps,
                 meta_batch_size, nn_model, f_loss, device, seed, n_ways, shots_mult):
        self.meta_batch_size = meta_batch_size
        self.train_adapt_steps = train_adapt_steps
        self.test_adapt_steps = test_adapt_steps
        self.device = device
        self.maml = l2l.algorithms.MAML(nn_model, lr=per_task_lr, first_order=False).to(device)
        self.loss = f_loss
        self.optimizer = torch.optim.Adam(self.maml.parameters(), meta_lr)
        self.seed = seed
        self.n_ways = n_ways
        self.shots_mult = shots_mult
        self.updated_model = False

    def calculate_meta_loss(self, task_batch, learner, adapt_steps):
        D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval = self.split_adapt_eval(task_batch)

        # Adapt the model
        for step in range(adapt_steps):
            adaptation_error = self.loss(learner(D_task_xs_adapt), D_task_ys_adapt)
            learner.adapt(adaptation_error)

        # Evaluate the adapted model
        predictions = learner(D_task_xs_error_eval)
        evaluation_error = self.loss(predictions, D_task_ys_error_eval)
        evaluation_accuracy = BaseMetaLearner.accuracy(predictions, D_task_ys_error_eval)
        return evaluation_error, evaluation_accuracy

    def meta_train(self, train_taskset, validation_taskset, n_epochs):
        for epoch in range(n_epochs):
            print(epoch)
            self.optimizer.zero_grad()
            meta_train_error = 0.0
            meta_train_accuracy = 0.0

            for task in range(self.meta_batch_size):
                # Compute meta-training loss
                learner = self.maml.clone().to(self.device)
                # sample
                batch = train_taskset.sample()
                evaluation_error, evaluation_accuracy = \
                    self.calculate_meta_loss(batch, learner, self.train_adapt_steps)

                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

            # Average the accumulated task gradients and optimize
            for p in self.maml.parameters():  # Note: this is somewhat bad practice
                p.grad.data.mul_(1.0 / self.meta_batch_size)
            self.optimizer.step()

        # TODO: make use of validation taskset somehow

    def meta_test_on_task(self, D_task_xs_adapt, D_task_ys_adapt, D_task_xs_error_eval, D_task_ys_error_eval, n_epochs):
        learner = self.maml.clone()
        for step in range(self.test_adapt_steps):
            adaptation_error = self.loss(learner(D_task_xs_adapt), D_task_ys_adapt)
            learner.adapt(adaptation_error)

        with torch.no_grad():
            test_predictions = learner(D_task_xs_error_eval)
            test_loss = self.loss(test_predictions, D_task_ys_error_eval)
            test_acc = BaseMetaLearner.accuracy(test_predictions, D_task_ys_error_eval)
        return test_loss.item(), test_acc.item(), None, None

    def load_saved_model(self, model_name):
        self.maml.module.load_state_dict(torch.load(model_name))

    def save_model(self, model_name):
        torch.save(self.maml.module.state_dict(), model_name)
