LATENT_DIM = 64
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


def compute_loss(x_batch, z_batch, mean, log_var, out):
    # # log_q(z|x) logprobability of z under approximate posterior N(μ,σ^2)
    # log_q_z_x = compute_log_pdf_diagonal_gaussian(z_batch, mean, torch.sqrt(torch.exp(log_var)))
    # print(log_q_z_x)
    #
    # # log_p_z(z) log probability of z under prior
    # log_p_z_z = compute_log_pdf_diagonal_gaussian(z_batch, torch.tensor(0).to(DEVICE), torch.tensor(1.0).to(DEVICE))
    # print(log_p_z_z)
    #
    # # log_p(x|z) - conditional probability of data given latents.
    # log_p_x_z = compute_log_pdf_bernoulli(x_batch, out)
    # print(log_p_x_z)
    #
    # L = -(log_p_x_z + log_p_z_z - log_q_z_x) / x_batch.shape[0]
    # return L
    BCE = F.mse_loss(out, x_batch, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return BCE
    # return (BCE + KLD) / x_batch.shape[0]


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

        self.ly1 = nn.Conv2d(3, 32, 4, stride=2)
        self.ly2 = nn.Conv2d(32, 64, 4, stride=2)
        self.ly3 = nn.Conv2d(64, 128, 4, stride=2)
        self.ly4 = nn.Conv2d(128, 256, 4, stride=2)
        self.mean = nn.Linear(2*2*256, LATENT_DIM)
        self.logvar = nn.Linear(2*2*256, LATENT_DIM)
        self.ly5 = nn.Linear(LATENT_DIM, 2*2*256)
        self.ly6 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.ly7 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.ly8 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.ly9 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def encode(self, x):
        x = F.relu(self.ly1(x))
        x = F.relu(self.ly2(x))
        x = F.relu(self.ly3(x))
        x = F.relu(self.ly4(x))
        x = x.view(x.size(0), -1)
        mean = self.mean(x)
        log_var = self.logvar(x)
        return mean, log_var

    # Define sample from recognition model
    # Samples z ~ q(z|x)
    def sample_from_recognition_model(self, mean, log_var):
        std = torch.sqrt(torch.exp(log_var))

        return sample_from_diagonal_gaussian(mean, std)

    # Define MLP for generative model / "decoder"
    # Provides parameters for distribution p(x|z)
    def decode(self, z):
        x = F.relu(self.ly5(z))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.ly6(x))
        x = F.relu(self.ly7(x))
        x = F.relu(self.ly8(x))
        return torch.sigmoid(self.ly9(x))

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = mean
        # z = self.sample_from_recognition_model(mean, log_var)
        return self.decode(z), z, mean, log_var