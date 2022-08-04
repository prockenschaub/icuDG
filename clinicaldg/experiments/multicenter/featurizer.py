
import torch.nn as nn

from clinicaldg import networks


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