import torch
import torch.nn as nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, latents, actions):
        super().__init__()
        self.fc = nn.Linear(latents, actions)

    def forward(self, *param):
        inputs = torch.cat(param, dim=1)
        return F.softmax(self.fc(inputs), 1)
