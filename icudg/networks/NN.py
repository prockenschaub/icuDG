import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """Just an MLP"""
    def __init__(self, n_inputs, hidden_dims, num_layers, n_outputs, dropout=0.):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hidden_dims)
        self.dropout = nn.Dropout(dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(hidden_dims,hidden_dims)
            for _ in range(num_layers-2)])
        self.output = nn.Linear(hidden_dims, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x
