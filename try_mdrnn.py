from MDRNN import MDRNN_Train, MDRNN_Rollout
from misc import DEVICE
from misc import VAE_LATENT_DIM, ACTION_DIM, RNN_HIDDEN_DIM
from doom_dataset import DoomDataset
from torch.utils.data import DataLoader
import torch
from AE import AE
import numpy as np

'''
This file generates images from real observations, 
'''

ae = AE().to(DEVICE)
ae.load_state_dict(torch.load('vae_final.weights'))
num_gaussians = 5
mdrnn = MDRNN_Train(VAE_LATENT_DIM, ACTION_DIM, RNN_HIDDEN_DIM, num_gaussians)
mdrnn_c = MDRNN_Rollout(VAE_LATENT_DIM, ACTION_DIM, RNN_HIDDEN_DIM, num_gaussians)
mdrnn.to(DEVICE)
mdrnn.load_state_dict(torch.load('log/mdrnn/checkpoint.tar')['state_dict'])
mdrnn.train()

mdrnn_c.to(DEVICE)

rnn_state_dict = {k.strip('_l0'): v for k, v in torch.load('log/mdrnn/checkpoint.tar')['state_dict'].items()}
mdrnn_c.load_state_dict(rnn_state_dict)

def to_latent(obs, next_obs):
    """ Transform observations to latent space.
    :args obs: (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    """
    with torch.no_grad():
        latent_obs, latent_next_obs = ae(obs.squeeze(0))[1], ae(next_obs.squeeze(0))[1]
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

pi = torch.exp(logpi.squeeze())

mixt = torch.argmax(pi, dim=1)

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
    decode_img = ae.decode(latent_next_obs[:, ii, :]).squeeze().detach().cpu().numpy()
    decode_rnn_img = ae.decode(torch.tensor(lstate[ii]).to(DEVICE)).squeeze().detach().cpu().numpy()
    mus, sigmas, logpi, d, next_hidden = mdrnn_c(acs[ii].unsqueeze(0), latent_obs[ii].unsqueeze(0), rnn_hidden_state)
    rnn_hidden_state = next_hidden
    pi = torch.exp(logpi.squeeze())
    mixt = torch.argmax(pi, dim=0)
    rnn_c_latent = mus[:, mixt, :]
    decode_rnn_c_img = ae.decode(rnn_c_latent).squeeze().detach().cpu().numpy()

    misc.imsave(os.path.join('rnn_regen_img', f'{ii}_vae_recon.png'), (np.moveaxis(decode_img, 0, 2) * 255.0).astype(np.uint8))
    misc.imsave(os.path.join('rnn_regen_img', f'{ii}_vae_rnn_recon.png'), (np.moveaxis(decode_rnn_img, 0, 2) * 255.0).astype(np.uint8))
    misc.imsave(os.path.join('rnn_regen_img', f'{ii}_vae_rnn_c_recon.png'), (np.moveaxis(decode_rnn_c_img, 0, 2) * 255.0).astype(np.uint8))


pass


