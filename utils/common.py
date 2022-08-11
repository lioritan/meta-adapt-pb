from __future__ import absolute_import, division, print_function

from datetime import datetime
import os
import torch
import numpy as np
import random
import sys
import pickle
from functools import reduce

# -----------------------------------------------------------------------------------------------------------#
# General auxilary functions
# -----------------------------------------------------------------------------------------------------------#

def list_mult(L):
    return reduce(lambda x, y: x* y, L)

# -----------------------------------------------------------------------------------------------------------#

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# -----------------------------------------------------------------------------------------------------------#

def get_prediction(outputs):
    if outputs.shape[1] == 1:
        # binary classification
        pred = (outputs > 0)
    else:
        # multi-class classification
        ''' Determine the class prediction by the max output and compare to ground truth'''
        pred = outputs.data.max(1, keepdim=True)[1]  # get the index of the max output
    return pred


# -----------------------------------------------------------------------------------------------------------#

def count_correct(outputs, targets):
    pred = get_prediction(outputs)
    return pred.eq(targets.data.view_as(pred)).cpu().sum().item()


# -----------------------------------------------------------------------------------------------------------#

def correct_rate(outputs, targets):
    n_correct = count_correct(outputs, targets)
    n_samples = outputs.shape[0]
    return n_correct / n_samples


# -----------------------------------------------------------------------------------------------------------#

def save_model_state(model, f_path):
    with open(f_path, 'wb') as f_pointer:
        torch.save(model.state_dict(), f_pointer)
    return f_path


# -----------------------------------------------------------------------------------------------------------#

def load_model_state(model, f_path):
    if not os.path.exists(f_path):
        raise ValueError('No file found with the path: ' + f_path)
    with open(f_path, 'rb') as f_pointer:
        model.load_state_dict(torch.load(f_pointer))


def net_weights_magnitude(model, device, p=2):  # corrected
    ''' Calculates the total p-norm of the weights  |W|_p^p
        If exp_on_logs flag is on, then parameters with log_var in their name are exponented'''
    total_mag = torch.zeros(1, device=device, requires_grad=True)[0]
    for (param_name, param) in model.named_parameters():
        total_mag = total_mag + param.pow(p).sum()
    return total_mag


# -----------------------------------------------------------------------------------------------------------#
# Optimizer
# -----------------------------------------------------------------------------------------------------------#

# Gradient step function:
def grad_step(objective, optimizer, lr_schedule=None, initial_lr=None, i_epoch=None):
    if lr_schedule:
        adjust_learning_rate_schedule(optimizer, i_epoch, initial_lr, **lr_schedule)
    optimizer.zero_grad()
    objective.backward()
    # torch.nn.utils.clip_grad_norm(parameters, 0.25)
    optimizer.step()


def adjust_learning_rate_interval(optimizer, epoch, initial_lr, gamma, decay_interval):
    """Sets the learning rate to the initial LR decayed by gamma every decay_interval epochs"""
    lr = initial_lr * (gamma ** (epoch // decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_schedule(optimizer, epoch, initial_lr, decay_factor, decay_epochs):
    """The learning rate is decayed by decay_factor at each interval start """

    # Find the index of the current interval:
    interval_index = len([mark for mark in decay_epochs if mark < epoch])

    lr = initial_lr * (decay_factor ** interval_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
