
from torch import nn
from torch.nn.utils import weight_norm

# Temporal convolutional network (TCN) -----------------------------------------
#
# Implementation of Temporal Convolutional Network as described in 
# Bai et al. (2018) An Empirical Evaluation of Generic Convolutional and 
# Recurrent Networks for Sequence Modeling (https://arxiv.org/abs/1803.01271)
#
# Code copied from https://github.com/locuslab/TCN
# Comments added by Patrick Rockenschaub


class Chomp1d(nn.Module):
    """
    Module that takes the output of a conventional Conv1d layer and truncates 
    the padding on the right to turn it into a causal convolution
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size # = number of padding steps on each side

    def forward(self, x):
        # x: (batch_size, num_features, padding + seq_length + padding)
        return x[:, :, :-self.chomp_size].contiguous()
        # return (batch_size, num_features, padding + seq_length)

class TemporalBlock(nn.Module):
    """A single temporal convolution block + residual connection. Consists of 
    two 1D convolutions followed by weight normalisation, ReLU activation and 
    dropout. 

    See reference at the top of this file for more information.

    Args:
        n_inputs (int): dimension of the input (i.e., number of features per
            timestep).
        n_outputs (int): dimension of the output (i.e., number of embedded 
            features after applying the temporal block)
        kernel_size (int): number of past time steps (including the current
            time step) used to calculate the output
        dilation (int): distance in the sequence between adjacent inputs to 
            the kernel
        padding (int): amount of zero padding to add on both sides of the 
            input (note: this is automatically set by TemporalConvNet to achieve
            a causal convolution, i.e., a convolution that uses only past and 
            no future information)
        dropout (float): proportion of nodes to disable during training
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # First causal convolution layer
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second causal convolution layer
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
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

    def forward(self, x):
        # x: (batch_size, num_features, seq_length)
        out = self.net(x)
        # out: (batch_size, num_features, seq_length)
        res = x if self.downsample is None else self.downsample(x)
        # res: (batch_size, num_features, seq_length)
        return self.relu(out + res)
        # return (batch_size, num_features, seq_length)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network created by stacking one or more 
    TemporalBlock modules.

    Args:
        num_inputs (int): dimension of the input (i.e., number of features per
            timestep).
        num_channels (list of int): dimension of the output (i.e., number of 
            embedded features after applying the temporal block) at each level 
            of the stack. Input size of a higher blocks is defined by the output
            size of their predecessor. `len(num_channels)` defines the number of 
            temporal blocks used.
        kernel_size (int): number of past time steps (including the current
            time step) used to calculate the output
        dropout (float): proportion of nodes to disable during training
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
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

    def forward(self, x):
        # x: (batch_size, seq_length, num_features)
        # Note: TCN expect seq_length in the last dimension --> transpose
        return self.network(x.transpose(1, 2)).transpose(1, 2)
        # return (batch_size, seq_length, num_features)
