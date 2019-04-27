import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self, latents, actions):
        super().__init__()
        self.fc = nn.Linear(latents, actions)

    def forward(self, *param):
        inputs = toch.cat(param, dim=1)
        return self.fc(inputs)
