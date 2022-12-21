import torch.nn as nn
from torch import Tensor
from typing import Dict, Any


class KeySelector(nn.Module):
    #           (self, input: str, target: str = None, loss: nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        arg_names = ["input", "target", "loss"]
        required_arg_names = ["input", "loss"]

        for name in arg_names:
            setattr(self, name, None)

        if len(args) + len(kwargs) <= len(required_arg_names):
            arg_names = required_arg_names

        args_set = []
        for name, value in zip(arg_names, args):
            setattr(self, name, value)
            args_set += [name]

        for name, value in kwargs.items():
            if name in arg_names and name not in args_set:
                setattr(self, name, value)
            else:
                raise TypeError(f"__init__() got an unexpected keyword argument '{name}'")

        not_set_args = [name for name in required_arg_names if name not in args_set]
        if len(not_set_args) > 0:
            raise TypeError(f"__init__() missing {len(not_set_args)} required "
                            f"positional argument{'s' if len(not_set_args) > 1 else ''}: '" +
                            "', '".join(not_set_args) + "'")

    def forward(self, container: Dict[str, Any]) -> Tensor:
        if self.target is None:
            return self.loss(container[self.input])
        return self.loss(container[self.input], container[self.target])
