use_rnn = True
import torch
from misc import *


vae_file = 'vae_final.weights'
rnn_file = 'rnn_checkpoint.tar'
ctrl_file = 'temp/ctrl_rnn/best.tar' if use_rnn else 'temp/ctrl/best.tar'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
r_gen = RolloutGenerator(vae_file, ctrl_file, rnn_file, device, 1000, use_rnn=use_rnn)

if use_rnn:
    controller = Controller(VAE_LATENT_DIM + RNN_HIDDEN_DIM, ACTION_DIM)
else:
    controller = Controller(VAE_LATENT_DIM, ACTION_DIM)

if exists(ctrl_file):
    state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})
    cur_best = - state['reward']
    controller.load_state_dict(state['state_dict'])
    # print("Previous best was {}...".format(-cur_best))

params = flatten_parameters(controller.parameters())

for i in range(100):
    r_gen.rollout(params)