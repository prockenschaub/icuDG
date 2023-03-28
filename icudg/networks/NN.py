from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """Just a Multilayer Perceptron
    
    Args:
        n_inputs: dimension of input feature vector
        hidden_dims: dimension of hidden layers
        num_layers: total number of layers (including input and output layer)
        n_outputs: dimensionality of output, e.g., 2 in the case of binary classification
        dropout: proportion of nodes to disable during training. Defaults to 0..
    """
    def __init__(self, n_inputs: int, hidden_dims: int, num_layers: int, n_outputs: int, dropout: float = 0.):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hidden_dims)
        self.dropout = nn.Dropout(dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(hidden_dims,hidden_dims)
            for _ in range(num_layers-2)])
        self.output = nn.Linear(hidden_dims, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x: Tensor) -> Tensor:
        """Apply forward step to x

        Args:
            x: input data of dimension (batch_size, ..., n_inputs) 

        Returns:
            model prediction of dimension (batch_size, ..., n_output)
        """
        x = self.input(x)
        x = self.dropout(x)
        x = F.leaky_relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.leaky_relu(x)
        x = self.output(x)
        return x
