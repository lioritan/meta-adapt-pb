
from __future__ import absolute_import, division, print_function


import torch
from torch.autograd import Variable
import math
from utils import common as cmn
import torch.nn.functional as F
from models.stochastic_layers import StochasticLayer
from utils.common import net_weights_magnitude, count_correct
# -----------------------------------------------------------------------------------------------------------#


def get_hyper_divergnce(var_prior, var_posterior, prior_model, device):   # corrected
    ''' calculates a divergence between hyper-prior and hyper-posterior....
     which is, in our case, just a regularization term over the prior parameters  '''

    # Note:  the hyper-prior is N(0, kappa_prior^2 * I)
    # Note:  the hyper-posterior is N(parameters-of-prior-distribution, kappa_post^2 * I)


    # KLD between hyper-posterior and hyper-prior:
    norm_sqr = net_weights_magnitude(prior_model, device, p=2)
    hyper_dvrg = (norm_sqr + var_posterior**2) / (2 * var_prior**2) + math.log(var_prior / var_posterior) - 1/2

    assert hyper_dvrg >= 0
    return hyper_dvrg


# -----------------------------------------------------------------------------------------------------------#

def get_meta_complexity_term(hyper_kl, delta, n_train_tasks):   # corrected
    if n_train_tasks == 0:
        meta_complex_term = 0.0  # infinite tasks case
    else:
        meta_complex_term = torch.sqrt(
            (hyper_kl + math.log(2 * n_train_tasks / delta)) / (2 * (n_train_tasks - 1)))  # corrected

    return meta_complex_term
# -----------------------------------------------------------------------------------------------------------#


def get_task_complexity(bound_type, delta, kappa_post, prior_model, post_model, n_samples, avg_empiric_loss, hyper_dvrg=0, n_train_tasks=1, dvrg=None, noised_prior=False):   # corrected
    #  Intra-task complexity for posterior distribution

    if not dvrg:
        # calculate divergence between posterior and sampled prior
        dvrg = get_net_densities_divergence(prior_model, post_model, kappa_post, noised_prior)

    if bound_type == 'McAllester':
        # According to 'Simplified PAC-Bayesian Margin Bounds', McAllester 2003
        complex_term = torch.sqrt((hyper_dvrg + dvrg + math.log(2 * n_samples * n_train_tasks / delta)) / (2 * (n_samples - 1)))  # corrected

    elif bound_type == 'Seeger':
        # According to 'Simplified PAC-Bayesian Margin Bounds', McAllester 2003
        seeger_eps = (dvrg + hyper_dvrg + math.log(2 * n_train_tasks * math.sqrt(n_samples) / delta)) / n_samples # corrected

        sqrt_arg = 2 * seeger_eps * avg_empiric_loss
        # sqrt_arg = F.relu(sqrt_arg)  # prevent negative values due to numerical errors
        complex_term = 2 * seeger_eps + torch.sqrt(sqrt_arg)

    elif bound_type == 'Catoni':
        # See "From PAC-Bayes Bounds to KL Regularization" Germain 2009
        # & Olivier Catoni. PAC-Bayesian surpevised classification: the thermodynamics of statistical learning
        complex_term = avg_empiric_loss + (2 / n_samples) * (hyper_dvrg + dvrg + math.log(n_train_tasks/ delta))  # corrected

    return complex_term
# -------------------------------------------------------------------------------------------


def get_net_densities_divergence(prior_model, post_model, prm, noised_prior=False):

    prior_layers_list = [layer for layer in prior_model.children() if isinstance(layer, StochasticLayer)]
    post_layers_list = [layer for layer in post_model.children() if isinstance(layer, StochasticLayer)]

    total_dvrg = 0
    for i_layer, prior_layer in enumerate(prior_layers_list):
        post_layer = post_layers_list[i_layer]
        if hasattr(prior_layer, 'w'):
            total_dvrg += get_dvrg_element(post_layer.w, prior_layer.w, prm, noised_prior)
        if hasattr(prior_layer, 'b'):
            total_dvrg += get_dvrg_element(post_layer.b, prior_layer.b, prm, noised_prior)

    return total_dvrg
# -------------------------------------------------------------------------------------------

def  get_dvrg_element(post, prior, kappa_post, noised_prior=False):
    """KL divergence D_{KL}[post(x)||prior(x)] for a fully factorized Gaussian"""

    if noised_prior and kappa_post > 0:
        prior_log_var = add_noise(prior['log_var'], kappa_post)
        prior_mean = add_noise(prior['mean'], kappa_post)
    else:
        prior_log_var = prior['log_var']
        prior_mean = prior['mean']

    post_var = torch.exp(post['log_var'])
    prior_var = torch.exp(prior_log_var)
    post_std = torch.exp(0.5 * post['log_var'])
    prior_std = torch.exp(0.5 * prior_log_var)

    numerator = (post['mean'] - prior_mean).pow(2) + post_var
    denominator = prior_var
    div_elem = 0.5 * torch.sum(prior_log_var - post['log_var'] + numerator / denominator - 1)


    # note: don't add small number to denominator, since we need to have zero KL when post==prior.

    assert div_elem >= 0
    return div_elem
# -------------------------------------------------------------------------------------------

def add_noise(param, std):
    return param + Variable(param.data.new(param.size()).normal_(0, std), requires_grad=False)
# -------------------------------------------------------------------------------------------

def add_noise_to_model(model, std):

    layers_list = [layer for layer in model.children() if isinstance(layer, StochasticLayer)]

    for i_layer, layer in enumerate(layers_list):
        if hasattr(layer, 'w'):
            add_noise(layer.w['log_var'], std)
            add_noise(layer.w['mean'], std)
        if hasattr(layer, 'b'):
            add_noise(layer.b['log_var'], std)
            add_noise(layer.b['mean'], std)