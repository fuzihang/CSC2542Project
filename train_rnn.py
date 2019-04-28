from os.path import join, exists
from os import mkdir
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from misc import VAE_LATENT_DIM
from AE import AE
from MDRNN import MDRNN_Train, gmm_loss
from doom_dataset import DoomDataset
from misc import DEVICE
import torch
import argparse
import time

parser = argparse.ArgumentParser("MDRNN training")
parser.add_argument('--logdir', type=str,
                    help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true',
                    help="Do not reload if specified.")
parser.add_argument('--include_reward', action='store_true',
                    help="Add a reward modelisation term to the loss.")
args = parser.parse_args()

# constants
BSIZE = 16
SEQ_LEN = 32
epochs = 20
TRAIN_DIR = '/home/zihang/Documents/npz_data'
TEST_DIR = '/home/zihang/Documents/npz_data_test'
# Loading VAE
# vae_file = join(args.logdir, 'vae', 'best.tar')
# assert exists(vae_file), "No trained VAE in the logdir..."
# state = torch.load(vae_file)
# print("Loading VAE at epoch {} "
#       "with test error {}".format(
#     state['epoch'], state['precision']))

vae = AE().to(DEVICE)
# vae.load_state_dict(state['state_dict'])
vae.load_state_dict(torch.load('vae_final.weights'))
# Loading model
rnn_dir = join(args.logdir, 'mdrnn')
rnn_file = join(rnn_dir, 'best_rnn.tar')

if not exists(rnn_dir):
    mkdir(rnn_dir)

mdrnn = MDRNN_Train(64, 2, 512, 5)
mdrnn.to(DEVICE)
optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-3, alpha=.9)

if exists(rnn_file) and not args.noreload:
    rnn_state = torch.load(rnn_file)
    print("Loading MDRNN at epoch {} "
          "with test error {}".format(
        rnn_state["epoch"], rnn_state["precision"]))
    mdrnn.load_state_dict(rnn_state["state_dict"])
    optimizer.load_state_dict(rnn_state["optimizer"])
    # scheduler.load_state_dict(state['scheduler'])
    # earlystopping.load_state_dict(state['earlystopping'])

# Data Loading
transform = transforms.Lambda(
    lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)

train_dataset = DoomDataset(TRAIN_DIR, train_rnn=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# test_dataset = DoomDataset(TEST_DIR, train_rnn=True)
# test_loader = DataLoader(test_dataset)


def to_latent(obs, next_obs):

    with torch.no_grad():
        latent_obs, latent_next_obs = vae(obs.squeeze(0))[1], vae(next_obs.squeeze(0))[1]
    return latent_obs.unsqueeze(0), latent_next_obs.unsqueeze(0)


def get_loss(latent_obs, action, latent_next_obs):
    """ Compute losses.
    :args latent_obs: (BSIZE, SEQ_LEN, VAE_LATENT_DIM) torch tensor
    :args action: (BSIZE, SEQ_LEN, ACTION_SIZE) torch tensor
    :args reward: (BSIZE, SEQ_LEN) torch tensor
    :args latent_next_obs: (BSIZE, SEQ_LEN, VAE_LATENT_DIM) torch tensor
    """
    latent_obs, action, latent_next_obs = [arr.transpose(1, 0)
                                           for arr in [latent_obs,
                                                       action,
                                                       latent_next_obs,
                                                       ]]
    means, sigmas, logpi, ds = mdrnn(action, latent_obs)
    gmm = gmm_loss(latent_next_obs, means, sigmas, logpi)

    scale = VAE_LATENT_DIM + 1
    loss = (gmm) / scale
    return dict(gmm=gmm, loss=loss)


def train_rnn(model, iterator, opt, start_time):
    model.train()
    losses = []

    for i_batch, batch in enumerate(iterator):
        obs = batch['obs'].to(DEVICE)
        acs = batch['acs'].to(DEVICE)
        next_obs = batch['next_obs'].to(DEVICE)

        latent_obs, latent_next_obs = to_latent(obs, next_obs)
        loss = get_loss(latent_obs, acs, latent_next_obs)['loss']

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        import math
        PRINT_INTERVAL = math.ceil(len(iterator) / 100)
        if (i_batch + 1) % PRINT_INTERVAL == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {}\t Time: {:10.3f}'.format(
                i_batch, len(iterator),
                i_batch / len(iterator) * 100,
                np.asarray(losses)[-PRINT_INTERVAL:].mean(0),
                time.time() - start_time,
            ))


cur_best = None


for epoch in range(epochs):
    train_rnn(mdrnn, train_loader, optimizer, time.time())
    checkpoint_fname = join(rnn_dir, 'checkpoint.tar')
    torch.save({
        "state_dict": mdrnn.state_dict(),
        "epoch": epoch},
        checkpoint_fname)

