from dataclasses import dataclass

import torch


@dataclass
class ANNSpecification:
    hidden_layers_size: list[int] | None = None
    output_size: int | None = None


def create_ann(input_size: int, hidden_layers_size: list[int] | None, output_size: int) -> torch.nn.Sequential:
    if hidden_layers_size:
        layer_sizes = [input_size, *hidden_layers_size, output_size]
    else:
        layer_sizes = [input_size, output_size]
    ann = []
    for ind in range(len(layer_sizes) - 2):
        ann.append(torch.nn.Linear(layer_sizes[ind], layer_sizes[ind + 1], dtype=torch.float32))
        ann.append(torch.nn.ReLU())
    ann.append(torch.nn.Linear(layer_sizes[-2], layer_sizes[-1], dtype=torch.float32))
    return torch.nn.Sequential(*ann)
