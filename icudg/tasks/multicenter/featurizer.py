from torch import Tensor
import torch.nn as nn

from icudg import networks


class GRUNet(nn.Module):
    """Featurizer that uses a stacked Gated Recurrent Unit to embed the input features

    Args: see torch.nn.GRU
    """
    def __init__(self, num_inputs: int, hidden_dims: int, num_layers: int, dropout: float):
        super().__init__()
        self.rnn = nn.GRU(num_inputs, hidden_dims, num_layers, batch_first=True, dropout=dropout)
        self.n_inputs = num_inputs
        self.n_outputs = hidden_dims

    def forward(self, x: Tensor) -> Tensor:
        out, hn = self.rnn(x)
        return out


class TCNet(nn.Module):
    """Featurizer that uses a Temporal Convolutional Network to embed the input features

    Args: see icudg.networks.TCN
    """
    def __init__(self, num_inputs: int, hidden_dims: int, num_layers: int, kernel_size: int, dropout: float):
        super().__init__()
        self.tcn = networks.TCN(
            num_inputs,
            [hidden_dims] * num_layers,
            kernel_size,
            dropout
        )
        self.n_inputs = num_inputs
        self.n_outputs = hidden_dims

    def forward(self, x: Tensor) -> Tensor:
        return self.tcn(x)


class TransformerNet(nn.Module):
    """Featurizer that uses self-attention to embed the input features

    Args: see icudg.networks.Transformer
    """
    def __init__(self, num_inputs: int, hidden_dims: int, num_layers: int, num_heads: int, dropout: float):
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
        self.n_inputs = num_inputs
        self.n_outputs = hidden_dims

    def forward(self, x: Tensor) -> Tensor:
        return self.tf(x)


class LastStep(nn.Module):
    """Wrap another featurizer and only forward the last time step

    Note: assumes the following output dimensions of the featurzier (batch_size, seq_len, n_outputs)

    Args: 
        featurizer: base featurizer to wrap 
    """
    def __init__(self, featurizer: nn.Module):
        super().__init__()
        self.featurizer = featurizer
    
    @property
    def n_inputs(self) -> int:
        return self.featurizer.n_inputs

    @property
    def n_outputs(self) -> int:
        return self.featurizer.n_outputs
    
    def forward(self, x: Tensor) -> Tensor:
        return self.featurizer(x)[:, -1, :]


class NeuMissFeaturizer(nn.Module):
    """Wrap an existing featurizer, allowing it to deal with missing data without imputation

    Note: assumes the following output dimensions of the featurzier (batch_size, seq_len, n_outputs)

    Args: 
        featurizer: base featurizer to wrap 
        neumiss_depth: number of layers in the NeuMiss block that handles missing data
    """
    def __init__(self, featurizer: nn.Module, neumiss_depth=1):
        super().__init__()
        self.neumiss = networks.NeuMissBlock(
            featurizer.n_inputs, 
            neumiss_depth
        )
        self.featurizer = featurizer
    
    @property
    def n_inputs(self) -> int:
        return self.featurizer.n_inputs

    @property
    def n_outputs(self) -> int:
        return self.featurizer.n_outputs
    
    def forward(self, x: Tensor) -> Tensor:
        return self.featurizer(self.neumiss(x))
