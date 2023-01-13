import copy

from torch import nn, Tensor
from torch.nn.utils import weight_norm

from typing import List

# Temporal convolutional network (TCN) -----------------------------------------
#
# Implementation of Temporal Convolutional Network as described in 
# Bai et al. (2018) An Empirical Evaluation of Generic Convolutional and 
# Recurrent Networks for Sequence Modeling (https://arxiv.org/abs/1803.01271)
#
# Code copied from https://github.com/locuslab/TCN
# Comments and Conv1d deepcopy fix added by Patrick Rockenschaub


class Chomp1d(nn.Module):
    """
    Module that takes the output of a conventional Conv1d layer and truncates 
    the padding on the right to turn it into a causal convolution
    """
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size # = number of padding steps on each side

    def forward(self, x: Tensor) -> Tensor:
        """Apply forward step to x

        Args:
            x: input data of dimension (batch_size, num_features, padding + seq_length + padding)

        Returns:
            model prediction of dimension (batch_size, num_features, padding + seq_length)
        """
        return x[:, :, :-self.chomp_size].contiguous()


def deepcopy_for_conv1d(self, memo):
    """Workaround for RuntimeError "Only Tensors created explicitly by the user 
    (graph leaves) support the deepcopy protocol at the moment" when copying a 
    layer with weight norm. 
    
    Adapted from https://github.com/pytorch/pytorch/issues/28594#issuecomment-679534348
    """

    # save and delete the weightnorm weight on self
    weight = None
    if hasattr(self, 'weight'):
        weight = getattr(self, 'weight')
        delattr(self, 'weight')

    # remove this deepcopy method to avoid infinite recursion
    __deepcopy__ = self.__deepcopy__
    del self.__deepcopy__

    # actually do the copy
    result = copy.deepcopy(self)

    # restore weightnorm weight on self
    if weight is None:
        setattr(self, 'weight', weight)
    self.__deepcopy__ = __deepcopy__

    return result

class Conv1dWithWN(nn.Conv1d):
    """Wrapper around nn.Conv1d that automatically applies weight norm and 
    fixes a deepcopy issue."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        weight_norm(self)
        self.__deepcopy__ = deepcopy_for_conv1d.__get__(self, self.__class__)


class TemporalBlock(nn.Module):
    """A single temporal convolution block + residual connection. Consists of 
    two 1D convolutions followed by weight normalisation, ReLU activation and 
    dropout. 

    See reference at the top of this file for more information.

    Args:
        n_inputs: dimension of the input (i.e., number of features per
            timestep).
        n_outputs: dimension of the output (i.e., number of embedded 
            features after applying the temporal block)
        kernel_size: number of past time steps (including the current
            time step) used to calculate the output
        dilation: distance in the sequence between adjacent inputs to 
            the kernel
        padding: amount of zero padding to add on both sides of the 
            input (note: this is automatically set by TemporalConvNet to achieve
            a causal convolution, i.e., a convolution that uses only past and 
            no future information)
        dropout: proportion of nodes to disable during training. Defaults to 0.
    """
    def __init__(
        self, 
        n_inputs: int, 
        n_outputs: int, 
        kernel_size: int, 
        stride: int, 
        dilation: int, 
        padding: int, 
        dropout: float = 0.
    ):
        super(TemporalBlock, self).__init__()
        # First causal convolution layer
        self.conv1 = Conv1dWithWN(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second causal convolution layer
        self.conv2 = Conv1dWithWN(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Downsampling (not used in this repo)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        # Weight initialisation
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: Tensor) -> Tensor:
        """Apply forward step to x

        Args:
            x: input data of dimension (batch_size, num_features, seq_length) 

        Returns:
            model prediction of dimension (batch_size, num_features, seq_length)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network created by stacking one or more 
    TemporalBlock modules.

    Args:
        num_inputs: dimension of the input (i.e., number of features per
            timestep).
        num_channels: dimension of the output (i.e., number of 
            embedded features after applying the temporal block) at each level 
            of the stack. Input size of a higher blocks is defined by the output
            size of their predecessor. `len(num_channels)` defines the number of 
            temporal blocks used.
        kernel_size: number of past time steps (including the current
            time step) used to calculate the output
        dropout: proportion of nodes to disable during training. Defaults to 0.
    """
    def __init__(
        self, 
        num_inputs: int, 
        num_channels: List[int], 
        kernel_size: int = 2,
        dropout: float = 0.2
    ):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.out_dim = out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Apply forward step to x

        Args:
            x: input data of dimension (batch_size, seq_length, num_features)

        Returns:
            model prediction of dimension (batch_size, seq_length, num_features)
        """
        # Note: TCN expect seq_length in the last dimension --> transpose
        return self.network(x.transpose(1, 2)).transpose(1, 2)
