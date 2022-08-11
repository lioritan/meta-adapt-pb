#
# the code is inspired by: https://github.com/katerakelly/pytorch-maml

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import list_mult
from models.stochastic_layers import StochasticLinear, StochasticConv2d, StochasticLayer
from models.layer_inits import init_layers


# -------------------------------------------------------------------------------------------
# Auxiliary functions
# -------------------------------------------------------------------------------------------
def get_size_of_conv_output(input_shape, conv_func):
    # generate dummy input sample and forward to get shape after conv layers
    batch_size = 1
    input = torch.rand(batch_size, *input_shape)
    output_feat = conv_func(input)
    conv_out_size = output_feat.data.view(batch_size, -1).size(1)
    return conv_out_size


def count_weights(model):
    # note: don't counts batch-norm parameters
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            count += list_mult(m.weight.shape)
            if hasattr(m, 'bias'):
                count += list_mult(m.bias.shape)
        elif isinstance(m, StochasticLayer):
            count += m.weights_count
    return count


#  -------------------------------------------------------------------------------------------
#  Main function
#  -------------------------------------------------------------------------------------------
def get_model(dataset, log_var_init, input_shape, output_dim, model_type='Stochastic'):
    # Get task info:

    # Define default layers functions
    def linear_layer(in_dim, out_dim, use_bias=True):
        if model_type == 'Standard':
            return nn.Linear(in_dim, out_dim, use_bias)
        elif model_type == 'Stochastic':
            return StochasticLinear(in_dim, out_dim, log_var_init, use_bias)

    def conv2d_layer(in_channels, out_channels, kernel_size, use_bias=True, stride=1, padding=1, dilation=1):
        if model_type == 'Standard':
            return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        elif model_type == 'Stochastic':
            return StochasticConv2d(in_channels, out_channels, kernel_size, log_var_init, use_bias, stride, padding,
                                    dilation)

    if dataset == 'mini-imagenet':
        model = ConvNet4(model_type, dataset, linear_layer, conv2d_layer, input_shape, output_dim)
    elif dataset in ['omniglot', 'mnist']:
        model = OmConvNet(model_type, dataset, linear_layer, conv2d_layer, input_shape, output_dim)
    else:
        raise ValueError('Invalid model_name')

    # init model:
    init_layers(model, log_var_init)

    model.weights_count = count_weights(model)

    # # For debug: set the STD of epsilon variable for re-parametrization trick (default=1.0)
    # if hasattr(prm, 'override_eps_std'):
    #     model.set_eps_std(prm.override_eps_std)  # debug

    return model


#  -------------------------------------------------------------------------------------------
#   Base class for all stochastic models
# -------------------------------------------------------------------------------------------
class general_model(nn.Module):
    def __init__(self):
        super(general_model, self).__init__()

    def set_eps_std(self, eps_std):
        old_eps_std = None
        for m in self.modules():
            if isinstance(m, StochasticLayer):
                old_eps_std = m.set_eps_std(eps_std)
        return old_eps_std

    def _init_weights(self, log_var_init):
        init_layers(self, log_var_init)


# -------------------------------------------------------------------------------------------
# Models collection
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
#  ConvNet
# -------------------------------------------------------------------------------- -----------
class ConvNet4(general_model):  # Model from Ravi et Larochelle, 2017. Same as Finn et. al
    def __init__(self, model_type, model_name, linear_layer, conv2d_layer, input_shape, output_dim, filt_size=32):
        super(ConvNet4, self).__init__()
        self.model_name = model_name
        self.model_type = model_type
        self.layers_names = ('conv1', 'conv2', 'conv3', 'conv4', 'FC_out')
        color_channels = input_shape[0]
        n_in_channels = input_shape[0]
        self.conv1 = conv2d_layer(n_in_channels, filt_size, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(filt_size, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv2d_layer(filt_size, filt_size, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(filt_size, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv2d_layer(filt_size, filt_size, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(filt_size, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = conv2d_layer(filt_size, filt_size, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(filt_size, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_out_size = get_size_of_conv_output(input_shape, self._forward_conv_layers)
        self.fc_out = linear_layer(conv_out_size, output_dim)

        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        self.bns = [self.bn1, self.bn2, self.bn3.self.bn4]

        # self._init_weights(log_var_init)  # Initialize weights

    def _forward_conv_layers(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x


# -------------------------------------------------------------------------------------------
#  OmConvNet
# -------------------------------------------------------------------------------- -----------
class OmConvNet(general_model):  # model from Finn et al. 2017 (MAML paper)
    def __init__(self, model_type, model_name, linear_layer, conv2d_layer, input_shape, output_dim, filt_size=64):
        super(OmConvNet, self).__init__()
        self.model_name = model_name
        self.model_type = model_type
        self.layers_names = ('conv1', 'conv2', 'conv3', 'conv4', 'FC_out')
        color_channels = input_shape[0]
        n_in_channels = input_shape[0]
        self.conv1 = conv2d_layer(n_in_channels, filt_size, kernel_size=3, stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(filt_size, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv2d_layer(filt_size, filt_size, kernel_size=3, stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(filt_size, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv2d_layer(filt_size, filt_size, kernel_size=3, stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(filt_size, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = conv2d_layer(filt_size, filt_size, kernel_size=3, stride=(2, 2))
        self.bn4 = nn.BatchNorm2d(filt_size, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU(inplace=True)
        conv_out_size = get_size_of_conv_output(input_shape, self._forward_conv_layers)
        self.fc_out = linear_layer(conv_out_size, output_dim)

        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        self.bns = [self.bn1, self.bn2, self.bn3, self.bn4]

        # self._init_weights(log_var_init)  # Initialize weights

    def _forward_conv_layers(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = x.mean(dim=[2, 3])
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x