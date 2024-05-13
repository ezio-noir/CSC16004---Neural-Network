import os

import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, **kwagrs):
        super(Model, self).__init__()

        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

def get_model_by_id(id: int, input_size: int, output_size: int) -> Model:
    hidden_sizes = []
    while id > 0:
        hidden_sizes.append((id % 10) * 100)
        id //= 10
    return Model(input_size, hidden_sizes, output_size)


def init_model_with_weight(path: str) -> tuple[Model, int]:
    filename = os.path.splitext(os.path.basename(path))[0]
    model_id, epoch = filename.split('_')[:2]
    model = get_model_by_id(model_id)
    model.load_state_dict(torch.load(path))
    return model, int(epoch)
    