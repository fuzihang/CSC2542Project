IN_DIM = 28 * 28
LATENT_DIM = 2
HIDDEN_UNITS = 500

from utils import DEVICE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
import numpy as np

def sample_from_diagonal_gaussian(mean, std):
    dim = mean.shape
    epsilon = torch.randn(dim).to(DEVICE)

    return mean + std * epsilon



# sampler from Bernoulli
def sample_from_bernoulli(p):
    dim = p.shape
    return (np.random.random_sample(dim) < p).astype(int)




# log-pdf of x under Diagonal Gaussian N(x|μ,σ^2 I)
def compute_log_pdf_diagonal_gaussian(x, mean, std):
    result = torch.sum(-((x - mean) ** 2) / (2 * (std ** 2)) - torch.log(torch.sqrt(2 * np.pi * (std ** 2))))


    return result


x = np.random.randn(1000, 2)
m = np.zeros((1000, 2))
s = np.ones((1000, 2))


# log-pdf of x under Bernoulli
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs


def compute_log_pdf_bernoulli(x, p):

    logits = probs_to_logits(p, is_binary=True)
    logits, x = broadcast_all(logits, x)

    return -F.binary_cross_entropy_with_logits(logits, x, reduction='sum')


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.uniform_(m.bias)


# Define MLP for recognition model / "encoder"
# Provides parameters for q(z|x)
class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.ly1 = nn.Linear(IN_DIM, HIDDEN_UNITS)
        self.ly2_mean = nn.Linear(HIDDEN_UNITS, LATENT_DIM)
        self.ly2_logvar = nn.Linear(HIDDEN_UNITS, LATENT_DIM)
        self.ly3 = nn.Linear(LATENT_DIM, HIDDEN_UNITS)
        self.ly4 = nn.Linear(HIDDEN_UNITS, IN_DIM)

    def encode(self, x):
        h = F.relu(self.ly1(x))
        mean = self.ly2_mean(h)
        log_var = self.ly2_logvar(h)
        return mean, log_var

    # Define sample from recognition model
    # Samples z ~ q(z|x)
    def sample_from_recognition_model(self, mean, log_var):
        std = torch.sqrt(torch.exp(log_var))

        return sample_from_diagonal_gaussian(mean, std)

    # Define MLP for generative model / "decoder"
    # Provides parameters for distribution p(x|z)
    def decode(self, z):
        h = F.relu(self.ly3(z))
        return torch.sigmoid(self.ly4(h))

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.sample_from_recognition_model(mean, log_var)
        return self.decode(z), z, mean, log_var