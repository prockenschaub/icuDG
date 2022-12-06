import torch
import torch.nn as nn

from icudg import networks


class GRUNet(nn.Module):
    def __init__(self, num_inputs, hidden_dims, num_layers, dropout):
        super().__init__()
        self.rnn = nn.GRU(num_inputs, hidden_dims, num_layers, batch_first=True, dropout=dropout)
        self.n_outputs = hidden_dims

    def forward(self, x):
        out, hn = self.rnn(x)
        return out


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
            emb=num_inputs,
            hidden=hidden_dims,
            heads=num_heads,
            ff_hidden_mult=2,
            depth=num_layers,
            dropout=dropout, 
            dropout_att=dropout
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