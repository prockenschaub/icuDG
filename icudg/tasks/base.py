import abc
import torch
from typing import List, Tuple, Callable

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from icudg.algorithms.base import Algorithm
from icudg.lib.hparams_registry import HparamMixin

class Task(HparamMixin):
    """Base class for any task definition
    """
    
    @abc.abstractmethod
    def get_torch_dataset(self, envs: List[str], fold: str) -> Dataset:
        """Get one or more envs as a torch dataset

        Args:
            envs: list of environment names to get
            fold: fold to return, can be 'train', 'val', or 'test'
        """
        pass

    @abc.abstractmethod
    def get_featurizer(self) -> torch.nn.Module:
        """Get the torch module used to embed the preprocessed input
        """
        pass

    @abc.abstractmethod
    def get_loss_fn(self) -> Callable:
        """Get the loss function for this task
        """
        pass

    @abc.abstractmethod
    def get_extended_loss_fn(self) -> Callable:
        """Get the loss function with extended gradient calculations for Fishr
        """
        pass

    @abc.abstractmethod
    def get_mask(self, batch: Tuple[Tensor]) -> Tensor:
        """Get a boolean mask to remove any targets that should not contribute to the loss

        If all targets should be used, return a tensor with all True of the same size as the targets.

        Args:
            batch: a single batch of data

        Returns: 
            mask (True = consider when calculating the loss)
        """
        pass

    @abc.abstractmethod
    def eval_metrics(
        self, 
        algorithm: Algorithm, 
        loader: DataLoader, 
        device: str, 
        **kwargs
    ):
        """Calculate evaluation metrics for this task
        
        Args:
            algorithm: the model used for prediction
            loader: a data loader with the data used for evaluation
            device: the device on which to run the model ('cpu' or 'cuda')

        Returns:
            metrics as dictionary
        """
        pass

    @abc.abstractmethod
    def save_task(self, file_path: str):
        """Save the task state for reproducibility (e.g., splits and means)

        Args:
            file_path: file path specifying where to save the task
        """
        pass
