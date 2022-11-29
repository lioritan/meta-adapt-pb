# standard stochastic meta-training and testing, optionally add low norm (uncertainty) to testing

import math
import os

from meta_learning.base_meta_learner import BaseMetaLearner
import torch

from prior_analysis_graph import run_prior_analysis
from utils.complexity_terms import get_hyper_divergnce, get_meta_complexity_term, get_task_complexity, \
    get_net_densities_divergence


def clone_model(base_model, ctor, device):
    post_model = ctor(0).to(device)
    post_model.load_state_dict(base_model.state_dict())
    return post_model


class BayesianVI(BaseMetaLearner):
    def __init__(self, per_task_lr, meta_lr,
                 train_adapt_steps, test_adapt_steps,
                 meta_batch_size,
                 f_loss, device, seed, n_ways,
                 stochastic_model,
                 model_ctor,
                 shots_mult, optimizer_weight_decay,
                 lr_decay_epochs, lr_schedule_type, early_stop, args_hash, test_penalty=0):
        self.model_ctor = model_ctor
        self.shots_mult = shots_mult
        self.stochastic_model = stochastic_model.to(device)
        self.n_ways = n_ways
        self.seed = seed
        self.device = device
        self.f_loss = f_loss
        self.meta_batch_size = meta_batch_size
        self.test_adapt_steps = test_adapt_steps
        self.train_adapt_steps = train_adapt_steps
        self.meta_lr = meta_lr
        self.per_task_lr = per_task_lr
        self.optimizer = torch.optim.Adam
        self.opt_params = {"lr": meta_lr, "weight_decay": optimizer_weight_decay}

        self.early_stop = early_stop
        self.lr_decay_epochs = lr_decay_epochs
        self.lr_schedule_type = lr_schedule_type

        self.test_penalty = test_penalty  # special parameter for comparison
        self.args_hash = args_hash

    @classmethod
    def run_eval_max_posterior(cls, model, batch, loss):
        ''' Estimates the the loss by using the mean network parameters'''
        loss_criterion = loss
        model.eval()

        inputs, targets = batch
        n_samples = len(targets)
        old_eps_std = model.set_eps_std(0.0)  # test with max-posterior
        outputs = model(inputs)
        model.set_eps_std(old_eps_std)  # return model to normal behaviour
        avg_loss = loss_criterion(outputs, targets) / n_samples
        n_correct = cls.accuracy(outputs, targets)
        return avg_loss, n_correct

    def get_pb_terms_single_task(self, x, y, prior, posterior, hyper_dvrg=0, n_tasks=1):
        n_MC = 3
        avg_empiric_loss = 0.0
        n_samples = len(y)
        for i_MC in range(n_MC):
            # Empirical Loss on current task:
            outputs = posterior(x)
            avg_empiric_loss_curr = 1 * self.f_loss(outputs, y)
            avg_empiric_loss += (1 / n_MC) * avg_empiric_loss_curr

        complexity = get_task_complexity(bound_type="McAllester", delta=0.1, kappa_post=1e-3,
                                         prior_model=prior, post_model=posterior,
                                         n_samples=n_samples, avg_empiric_loss=avg_empiric_loss, hyper_dvrg=hyper_dvrg,
                                         n_train_tasks=n_tasks, noised_prior=True)
        return avg_empiric_loss, complexity

    def meta_train(self, train_taskset, validation_taskset, n_epochs):
        lowest_loss = torch.inf
        count = 0
        patience = 50
        iters_per_validation = 10

        for epoch in range(n_epochs):
            self.stochastic_model.train()
            # make posterior models
            posterior_models = [clone_model(self.stochastic_model, self.model_ctor, self.device)
                                for i in range(self.meta_batch_size)]
            all_post_param = sum([list(posterior_model.parameters()) for posterior_model in posterior_models], [])
            prior_params = list(self.stochastic_model.parameters())
            all_params = all_post_param + prior_params
            optimizer = self.optimizer(all_params, **self.opt_params)
            for step in range(self.train_adapt_steps):
                hyper_dvrg = get_hyper_divergnce(var_prior=1e2, var_posterior=1e-3, prior_model=self.stochastic_model,
                                                 device=self.device)
                meta_complex_term = get_meta_complexity_term(hyper_dvrg, delta=0.1, n_train_tasks=self.meta_batch_size)
                losses = torch.zeros(self.meta_batch_size, device=self.device)
                complexities = torch.zeros(self.meta_batch_size, device=self.device)
                for i, task in enumerate(range(self.meta_batch_size)):
                    batch = train_taskset.sample()
                    losses[i], complexities[i] = self.get_pb_terms_single_task(
                        batch[0].to(self.device), batch[1].to(self.device),
                        self.stochastic_model, posterior_models[i],
                        hyper_dvrg=hyper_dvrg, n_tasks=self.meta_batch_size)
                pb_objective = losses.mean() + complexities.mean() + meta_complex_term
                optimizer.zero_grad()
                pb_objective.backward()
                optimizer.step()

            if not self.early_stop or (epoch % iters_per_validation != 0) or (epoch < (n_epochs // 10)):
                if self.lr_schedule_type == 'step' and epoch % self.lr_decay_epochs == 0:
                    self.opt_params["lr"] *= 0.9
                continue
            # early stopping
            val_loss = torch.tensor(self.get_validation_loss(validation_taskset))
            if val_loss <= lowest_loss - 1e-3:
                lowest_loss = val_loss
                count = 0
                os.makedirs("artifacts/tmp/bayes_vi", exist_ok=True)
                self.save_model(f"artifacts/tmp/bayes_vi/model{self.args_hash}.pkl")
            else:
                count += iters_per_validation
                if count >= patience:
                    print(f"early stop condition met, epoch: {epoch}, {val_loss.item()}, {lowest_loss.item()}")
                    if epoch < (n_epochs // 10):
                        count = 0
                        continue
                    else:
                        self.load_saved_model(f"artifacts/tmp/bayes_vi/model{self.args_hash}.pkl")
                        os.remove(f"artifacts/tmp/bayes_vi/model{self.args_hash}.pkl")
                        break
            if self.lr_schedule_type == 'step' and epoch % self.lr_decay_epochs == 0:
                self.opt_params["lr"] *= 0.9
        # TODO: use validation set for tuning

    def get_validation_loss(self, validation_taskset):
        batch = validation_taskset.sample()
        D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval = self.split_adapt_eval(batch)
        evaluation_error, evaluation_accuracy, _, _ = \
            self.meta_test_on_task(D_task_xs_adapt, D_task_ys_adapt, D_task_xs_error_eval, D_task_ys_error_eval,
                                   n_epochs=self.test_adapt_steps)
        del D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval
        torch.cuda.empty_cache()
        return evaluation_error

    def calculate_pb_bound(self, D_task_xs_adapt, D_task_ys_adapt, delta=0.1):
        trn_error, trn_acc = BayesianVI.run_eval_max_posterior(self.stochastic_model,
                                                               (D_task_xs_adapt, D_task_ys_adapt),
                                                               self.f_loss)
        m = len(D_task_ys_adapt)
        complexity_term = math.sqrt((math.log(2 * m / delta)) / (2 * m - 1))
        acc_bound = trn_acc - complexity_term
        err_bound = trn_error + complexity_term
        return err_bound, acc_bound

    def meta_test_on_task(self, D_task_xs_adapt, D_task_ys_adapt, D_task_xs_error_eval, D_task_ys_error_eval, n_epochs):
        self.stochastic_model.train()

        prior = clone_model(self.stochastic_model, self.model_ctor, self.device)
        optimizer = self.optimizer(self.stochastic_model.parameters(), **self.opt_params)
        for step in range(self.test_adapt_steps):
            loss, complexity = self.get_pb_terms_single_task(D_task_xs_adapt, D_task_ys_adapt,
                                                             prior, self.stochastic_model)
            hyper_dvrg = get_hyper_divergnce(var_prior=1e2, var_posterior=1e-3,
                                            prior_model=self.stochastic_model, device=self.device)
            pb_objective = loss + complexity + self.test_penalty * hyper_dvrg
            optimizer.zero_grad()
            pb_objective.backward()
            optimizer.step()

        self.stochastic_model.eval()
        evaluation_error, evaluation_accuracy = BayesianVI.run_eval_max_posterior(self.stochastic_model,
                                                                                  (D_task_xs_error_eval,
                                                                                   D_task_ys_error_eval),
                                                                                  self.f_loss)

        err_bound, acc_bound = self.calculate_pb_bound(D_task_xs_adapt, D_task_ys_adapt)
        return evaluation_error.item(), evaluation_accuracy.item(), err_bound.item(), acc_bound.item()

    def load_saved_model(self, model_name):
        self.stochastic_model.load_state_dict(torch.load(model_name))

    def save_model(self, model_name):
        torch.save(self.stochastic_model.state_dict(), model_name)
