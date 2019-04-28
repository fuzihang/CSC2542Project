"""
Define MDRNN model, supposed to be used as a world model
on the latent space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal

def gmm_loss(batch, mus, sigmas, logpi): # pylint: disable=too-many-arguments
    '''
    Compute cross entropy gmm loss
    :param batch:
    :param mus:
    :param sigmas:
    :param logpi:
    :return:
    '''
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze(2) + torch.log(probs)
    return - torch.mean(log_prob)

class MDRNNBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            hiddens, (2 * latents + 1) * gaussians + 1)

    def forward(self, *inputs):
        pass

class MDRNN_Train(MDRNNBase):
    '''
    MDRNN used with nn.LSTM for training
    '''
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTM(latents + actions, hiddens)

    def forward(self, actions, latents): # pylint: disable=arguments-differ
        """ MULTI STEPS forward.
        :args actions: (SEQ_LEN, BSIZE, ACTION_SIZE) torch tensor
        :args latents: (SEQ_LEN, BSIZE, VAE_LATENT_DIM) torch tensor
        """
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        hiddens, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(hiddens)

        partition = self.gaussians * self.latents

        means = gmm_outs[:, :, :partition]
        means = means.view(seq_len, bs, self.gaussians, self.latents)

        log_sigmas = gmm_outs[:, :, partition:2 * partition]
        log_sigmas = log_sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(log_sigmas)

        pi = gmm_outs[:, :, 2 * partition: 2 * partition + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        ds = gmm_outs[:, :, -1]

        return means, sigmas, logpi, ds

class MDRNN_Rollout(MDRNNBase):
    """ MDRNN with nn.LSTMCell for generating rollouts """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTMCell(latents + actions, hiddens)

    def forward(self, action, latent, hidden): # pylint: disable=arguments-differ
        """ ONE STEP forward.
        :args actions: (BSIZE, ACTION_SIZE) torch tensor
        :args latents: (BSIZE, VAE_LATENT_DIM) torch tensor
        :args hidden: (BSIZE, RNN_HIDDEN_DIM) torch tensor
        """
        input = torch.cat([action, latent], dim=1)

        next_hidden = self.rnn(input, hidden)
        out_rnn = next_hidden[0]

        out = self.gmm_linear(out_rnn)

        partition = self.gaussians * self.latents

        means = out[:, :partition]
        means = means.view(-1, self.gaussians, self.latents)

        log_sigmas = out[:, partition:2 * partition]
        log_sigmas = log_sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(log_sigmas)

        pi = out[:, 2 * partition:2 * partition + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        d = out[:, -1]

        return means, sigmas, logpi, d, next_hidden