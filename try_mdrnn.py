from other_mdrnn import MDRNN, MDRNNCell
from utils import DEVICE
from doom_dataset import DoomDataset
from torch.utils.data import DataLoader
import torch
from VAE import VAE
import numpy as np

vae = VAE().to(DEVICE)
vae.load_state_dict(torch.load('vae_final.weights'))

mdrnn = MDRNN(64, 2, 512, 5)
mdrnn_c = MDRNNCell(64, 2, 512, 5)
mdrnn.to(DEVICE)
mdrnn.load_state_dict(torch.load('log/mdrnn/checkpoint.tar')['state_dict'])
mdrnn.train()

mdrnn_c.to(DEVICE)
# mdrnn_c.load_state_dict(torch.load('log/mdrnn/checkpoint.tar')['state_dict'])

rnn_state_dict = {k.strip('_l0'): v for k, v in torch.load('log/mdrnn/checkpoint.tar')['state_dict'].items()}
mdrnn_c.load_state_dict(rnn_state_dict)

def to_latent(obs, next_obs):
    """ Transform observations to latent space.
    :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
    """
    with torch.no_grad():
        # obs, next_obs = [
        #     f.upsample(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE,
        #                mode='bilinear', align_corners=True)
        #     for x in (obs, next_obs)]
        #
        # (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
        #     vae(x)[1:] for x in (obs, next_obs)]
        #
        # latent_obs, latent_next_obs = [
        #     (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(BSIZE, SEQ_LEN, LSIZE)
        #     for x_mu, x_logsigma in
        #     [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
        latent_obs, latent_next_obs = vae(obs.squeeze(0))[1], vae(next_obs.squeeze(0))[1]
    return latent_obs.unsqueeze(0), latent_next_obs.unsqueeze(0)

TRAIN_DIR = '/home/zihang/Documents/npz_data'
dataset = DoomDataset(TRAIN_DIR, train_rnn=True)
data_iter = DataLoader(dataset, batch_size=1, shuffle=True)

data = enumerate(data_iter).__next__()[1]

obs = data['obs'].to(DEVICE)
acs = data['acs'].to(DEVICE)
next_obs = data['next_obs'].to(DEVICE)

latent_obs, latent_next_obs = to_latent(obs, next_obs)

mus, sigmas, logpi, ds = mdrnn(acs.transpose(0,1), latent_obs.transpose(0,1))

mus = mus.detach().cpu().numpy()
mus = np.moveaxis(mus, 1, 0)

from torch.distributions.categorical import Categorical
pi = torch.exp(logpi.squeeze())

# mixt = Categorical(pi).sample().detach().cpu().numpy()#.item()
mixt = torch.argmax(pi, dim=1)
# import numpy as np
#
# lstate = np.take(mus.squeeze().detach().cpu().numpy(), mixt, 1)

lstate = []
rnn_hidden_state = (torch.zeros((1, 512)).to(DEVICE),torch.zeros((1, 512)).to(DEVICE))

acs = acs.squeeze(0)
latent_obs = latent_obs.squeeze(0)

for i in range(mixt.shape[0]):
    lstate.append(mus[:, i, mixt[i], :])

from scipy import misc
import os
if not os.path.exists('rnn_regen_img'):
    os.mkdir('rnn_regen_img')
for ii in range(len(lstate)):
    decode_img = vae.decode(latent_next_obs[:, ii, :]).squeeze().detach().cpu().numpy()
    decode_rnn_img = vae.decode(torch.tensor(lstate[ii]).to(DEVICE)).squeeze().detach().cpu().numpy()
    mus, sigmas, logpi, d, next_hidden = mdrnn_c(acs[ii].unsqueeze(0), latent_obs[ii].unsqueeze(0), rnn_hidden_state)
    rnn_hidden_state = next_hidden
    pi = torch.exp(logpi.squeeze())
    mixt = torch.argmax(pi, dim=0)
    rnn_c_latent = mus[:, mixt, :]
    decode_rnn_c_img = vae.decode(rnn_c_latent).squeeze().detach().cpu().numpy()

    misc.imsave(os.path.join('rnn_regen_img', f'{ii}_vae_recon.png'), (np.moveaxis(decode_img, 0, 2) * 255.0).astype(np.uint8))
    misc.imsave(os.path.join('rnn_regen_img', f'{ii}_vae_rnn_recon.png'), (np.moveaxis(decode_rnn_img, 0, 2) * 255.0).astype(np.uint8))
    misc.imsave(os.path.join('rnn_regen_img', f'{ii}_vae_rnn_c_recon.png'), (np.moveaxis(decode_rnn_c_img, 0, 2) * 255.0).astype(np.uint8))


pass


