import math

import torch
import copy
import numpy as np
import learn2learn as l2l
from learn2learn.utils import clone_module
from learn2learn.vision.models import ConvBase

import models.stochastic_models

from optimizers.sgld_variant_optim import SimpleSGLDPriorSampling
from prior_analysis_graph import run_prior_analysis
from utils.complexity_terms import get_hyper_divergnce, get_meta_complexity_term, get_task_complexity, \
    get_net_densities_divergence


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def run_eval_max_posterior(model, batch, loss):
    ''' Estimates the the loss by using the mean network parameters'''
    loss_criterion = loss
    model.eval()

    inputs, targets = batch
    n_samples = len(targets)
    old_eps_std = model.set_eps_std(0.0)  # test with max-posterior
    outputs = model(inputs)
    model.set_eps_std(old_eps_std)  # return model to normal behaviour
    avg_loss = loss_criterion(outputs, targets) / n_samples
    n_correct = accuracy(outputs, targets)
    return avg_loss, n_correct


def clone_model(base_model, ctor, device):
    post_model = ctor(0).to(device)
    post_model.load_state_dict(base_model.state_dict())
    return post_model


class MetaLearner(object):
    def __init__(
            self,
            per_task_lr,
            meta_lr,
            train_adapt_steps,
            test_adapt_steps,
            meta_batch_size,
            f_loss,
            device,
            seed,
            n_ways,
            reset_clf_on_meta_loop,
            stochastic_model,
            stochastic_ctor,
            shots_mult,
            adaptive
    ):

        self.meta_batch_size = meta_batch_size
        self.train_adapt_steps = train_adapt_steps
        self.test_adapt_steps = test_adapt_steps
        self.device = device
        self.loss = f_loss
        self.optimizer = torch.optim.Adam
        self.opt_params = {"lr": per_task_lr}
        self.seed = seed
        self.n_ways = n_ways
        self.reset_clf_on_meta_loop = reset_clf_on_meta_loop
        self.stochastic_model = stochastic_model
        self.stochastic_ctor = stochastic_ctor
        self.shots_mult = shots_mult

        self.is_adaptive_prior = adaptive
        self.use_training_prior = adaptive
        self.analyze_layer_variance = False

    def calculate_meta_loss(self, task_batch, learner, adapt_steps):
        D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval = self.split_adapt_eval(task_batch)

        # Adapt the model
        for step in range(adapt_steps):
            adaptation_error = self.loss(learner(D_task_xs_adapt), D_task_ys_adapt)
            learner.adapt(adaptation_error)

        # Evaluate the adapted model
        predictions = learner(D_task_xs_error_eval)
        evaluation_error = self.loss(predictions, D_task_ys_error_eval)
        evaluation_accuracy = accuracy(predictions, D_task_ys_error_eval)

        del D_task_xs_error_eval, D_task_xs_adapt

        return evaluation_error, evaluation_accuracy

    def split_adapt_eval(self, task_batch, train_frac=2):
        D_task_xs, D_task_ys = task_batch
        D_task_xs, D_task_ys = D_task_xs.to(self.device), D_task_ys.to(self.device)
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

    def meta_train_pb_bound(self, n_epochs, train_taskset):
        for epoch in range(n_epochs):
            self.stochastic_model.train()
            # make posterior models
            posterior_models = [clone_model(self.stochastic_model, self.stochastic_ctor, self.device)
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
            print(f"epoch={epoch + 1}/{n_epochs}")

    def meta_test(self, test_meta_epochs, test_taskset):
        D_test_batch = test_taskset.sample()
        D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval = self.split_adapt_eval(
            D_test_batch, train_frac=self.shots_mult)

        self.stochastic_model.train()
        if self.analyze_layer_variance:
            run_prior_analysis(self.stochastic_model, False, save_path="./hyper-prior_model")

        # Hyper-KL from hyper-prior, const (data-free)
        orig_hyper_prior = clone_model(self.stochastic_model, self.stochastic_ctor, self.device)
        base_hyper_prior = clone_model(self.stochastic_model, self.stochastic_ctor, self.device)

        for epoch in range(test_meta_epochs):
            # Hyper-KL from hyper-prior, each loop (data-dependent)
            if self.is_adaptive_prior:
                base_hyper_prior = clone_model(self.stochastic_model, self.stochastic_ctor, self.device)
            # make posterior models
            posterior_models = [clone_model(self.stochastic_model, self.stochastic_ctor, self.device)
                                for i in range(self.meta_batch_size)]
            all_post_param = sum([list(posterior_model.parameters()) for posterior_model in posterior_models], [])
            prior_params = list(self.stochastic_model.parameters())
            all_params = all_post_param + prior_params
            optimizer = self.optimizer(all_params, **self.opt_params)
            for step in range(self.train_adapt_steps):
                if self.use_training_prior:
                    hyper_dvrg = get_net_densities_divergence(base_hyper_prior, self.stochastic_model, prm=1e-3)
                else:  # low norm
                    hyper_dvrg = get_hyper_divergnce(var_prior=1e2, var_posterior=1e-3,
                                                     prior_model=self.stochastic_model, device=self.device)
                meta_complex_term = get_meta_complexity_term(hyper_dvrg, delta=0.1, n_train_tasks=self.meta_batch_size)
                pb_objective = self.get_pb_objective(D_task_xs_adapt, D_task_ys_adapt, hyper_dvrg,
                                                     meta_complex_term, posterior_models)
                optimizer.zero_grad()
                pb_objective.backward()
                optimizer.step()

        if self.analyze_layer_variance:
            run_prior_analysis(self.stochastic_model, False, save_path="./hyper-posterior_model")
        # Standard meta-testing from here on
        prior = clone_model(self.stochastic_model, self.stochastic_ctor, self.device)
        optimizer = self.optimizer(self.stochastic_model.parameters(), **self.opt_params)
        for step in range(self.test_adapt_steps):
            loss, complexity = self.get_pb_terms_single_task(D_task_xs_adapt, D_task_ys_adapt,
                                                             prior, self.stochastic_model)
            pb_objective = loss + complexity
            optimizer.zero_grad()
            pb_objective.backward()
            optimizer.step()

        # Evaluate the adapted model
        evaluation_error, evaluation_accuracy = run_eval_max_posterior(self.stochastic_model,
                                                                       (D_task_xs_error_eval, D_task_ys_error_eval),
                                                                       self.loss)

        err_bound, acc_bound = self.calculate_pb_bound(D_task_xs_adapt, D_task_ys_adapt, orig_hyper_prior)

        # Logging
        print('Meta Test Error', evaluation_error.item(), flush=True)
        print('Meta Test Accuracy', evaluation_accuracy.item(), flush=True)
        print('Error bound', err_bound.item(), flush=True)
        print('Accuracy bound', acc_bound.item(), flush=True)

        return evaluation_error.item(), evaluation_accuracy.item(), err_bound.item(), acc_bound.item()

    def calculate_pb_bound(self, D_task_xs_adapt, D_task_ys_adapt, orig_hyper_prior, delta = 0.1):
        trn_error, trn_acc = run_eval_max_posterior(self.stochastic_model, (D_task_xs_adapt, D_task_ys_adapt),
                                                    self.loss)
        if self.use_training_prior:
            hyper_kl = get_net_densities_divergence(orig_hyper_prior, self.stochastic_model, prm=1e-3)
        else:
            hyper_kl = get_hyper_divergnce(var_prior=1e2, var_posterior=1e-3,
                                             prior_model=self.stochastic_model, device=self.device)
        m = len(D_task_ys_adapt)
        complexity_term = torch.sqrt((hyper_kl - math.log(2 * m / delta)) / (2 * m - 1))
        acc_bound = trn_acc - complexity_term
        err_bound = trn_error + complexity_term
        return err_bound, acc_bound

    def get_pb_objective(self, x_data, y_data, hyper_dvrg, meta_complex_term, posterior_models):
        losses = torch.zeros(self.meta_batch_size, device=self.device)
        complexities = torch.zeros(self.meta_batch_size, device=self.device)
        for i, task in enumerate(range(self.meta_batch_size)):
            shuffled_indices = torch.randperm(len(y_data))
            batch = (x_data[shuffled_indices], y_data[shuffled_indices])
            D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval = self.split_adapt_eval(
                batch)

            # Note: we can potentially use D_task_xs_adapt to optimize the bound
            losses[i], complexities[i] = self.get_pb_terms_single_task(D_task_xs_error_eval, D_task_ys_error_eval,
                                                                       self.stochastic_model, posterior_models[i],
                                                                       hyper_dvrg=hyper_dvrg,
                                                                       n_tasks=self.meta_batch_size)
        pb_objective = losses.mean() + complexities.mean()
        return pb_objective

    def get_pb_terms_single_task(self, x, y, prior, posterior, hyper_dvrg=0, n_tasks=1):
        n_MC = 3
        avg_empiric_loss = 0.0
        n_samples = len(y)
        for i_MC in range(n_MC):
            # Empirical Loss on current task:
            outputs = posterior(x)
            avg_empiric_loss_curr =1*  self.loss(outputs, y)
            avg_empiric_loss += (1 / n_MC) * avg_empiric_loss_curr

        complexity = get_task_complexity(bound_type="McAllester", delta=0.1, kappa_post=1e-3,
                                         prior_model=prior, post_model=posterior,
                                         n_samples=n_samples, avg_empiric_loss=avg_empiric_loss, hyper_dvrg=hyper_dvrg,
                                         n_train_tasks=n_tasks, noised_prior=True)
        return avg_empiric_loss, complexity
