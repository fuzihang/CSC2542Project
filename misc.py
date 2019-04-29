""" Various auxiliary utilities """
import math
from os.path import join, exists
from torchvision import transforms
import numpy as np
from AE import AE
from controller import Controller
from vizdoom_take_cover import VizdoomTakeCover
from MDRNN import MDRNN_Rollout

import torch
import argparse


ACTION_DIM = 2
VAE_LATENT_DIM = 64
RNN_HIDDEN_DIM = 512
IMG_SIZE = 64
SIZE = 64

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def flatten_parameters(params):
    #Converts parameters into single 1D array
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    #Converts params to example dimensions
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    #load params into controller
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class RolloutGenerator(object):
    def __init__(self, vae_file, ctrl_file, rnn_file, device, time_limit, use_rnn=False):
        #initializes models and environment

        self.use_rnn = use_rnn

        self.vae = AE().to(device)
        self.mdrnn = MDRNN_Rollout(VAE_LATENT_DIM, ACTION_DIM, RNN_HIDDEN_DIM, 5).to(device)
        self.rnn_hidden_state = (torch.zeros(1, RNN_HIDDEN_DIM).to(device), torch.zeros(1, RNN_HIDDEN_DIM).to(device))

        if not torch.cuda.is_available():
            self.vae.load_state_dict(torch.load(vae_file, map_location='cpu'))
            rnn_state_dict = torch.load(rnn_file, map_location='cpu')
            self.mdrnn.load_state_dict({k.strip('_l0'): v for k, v in rnn_state_dict['state_dict'].items()})
        else:
            self.vae.load_state_dict(torch.load(vae_file))
            rnn_state_dict = torch.load(rnn_file)
            self.mdrnn.load_state_dict({k.strip('_l0'): v for k, v in rnn_state_dict['state_dict'].items()})

        if self.use_rnn:
            self.controller = Controller(VAE_LATENT_DIM + RNN_HIDDEN_DIM, ACTION_DIM).to(device)

        else:
            self.controller = Controller(VAE_LATENT_DIM, ACTION_DIM).to(device)


        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])

        self.env = VizdoomTakeCover()
        self.device = device

        self.time_limit = time_limit

    def get_action_and_transition(self, obs):
        _, _, latent_mu, _ = self.vae(obs)
        if self.use_rnn:
            action = self.controller(torch.cat((latent_mu, self.rnn_hidden_state[0]), dim=1))
            _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, self.rnn_hidden_state)
            self.rnn_hidden_state = next_hidden

        else:
            action = self.controller(latent_mu)
        return action.squeeze().detach().cpu().numpy()

    def rollout(self, params):
        if params is not None:
            load_parameters(params, self.controller)

        obs = self.env.reset()

        cumulative = 0
        i = 0
        while True:
            obs = transform(obs).unsqueeze(0).to(self.device)
            action = self.get_action_and_transition(obs)
            action = np.argmax(action)
            obs, reward, done, _ = self.env.step(action)


            cumulative += reward
            if done or i > self.time_limit:
                return -cumulative
            i += 1
