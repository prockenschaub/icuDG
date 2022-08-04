import torch

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

def cat(lst):
    if torch.is_tensor(lst[0]):
        return torch.cat(lst)
    elif isinstance(lst[0], dict):
        return {i: torch.cat([j[i] for j in lst]) if torch.is_tensor(lst[0][i]) else list(chain([j[i] for j in lst])) for i in lst[0]}

    