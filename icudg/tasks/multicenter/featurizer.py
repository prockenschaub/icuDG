
import torch.nn as nn

from icudg import networks


class TCNet(nn.Module):
    def __init__(self, num_inputs, hidden_dims, num_layers, kernel_size, dropout):
        super().__init__()
        self.tcn = networks.TCN(
            num_inputs,
            [hidden_dims] * num_layers,
            kernel_size,
            dropout
        )
        self.n_outputs = hidden_dims

    def forward(self, x):
        return self.tcn(x)


class TransformerNet(nn.Module):
    def __init__(self, num_inputs, hidden_dims, num_layers, num_heads, dropout):
        super().__init__()
        self.tf = networks.Transformer(
            num_inputs,
            hidden_dims,
            num_heads,
            1,
            num_layers,
            dropout=dropout
        )
        self.n_outputs = hidden_dims

    def forward(self, x):
        return self.tf(x)


class LastStep(nn.Module):
    def __init__(self, featurizer):
        super().__init__()
        self.featurizer = featurizer
    
    @property
    def n_outputs(self):
        return self.featurizer.n_outputs
    
    def forward(self, x):
        return self.featurizer(x)[:, -1, :]