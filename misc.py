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

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    """ Flattening parameters.
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class RolloutGenerator(object):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, vae_file, ctrl_file, rnn_file, device, time_limit, use_rnn=False):
        """ Build vae, rnn, controller and environment. """

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
        """ Get action and transition.
        """
        _, _, latent_mu, _ = self.vae(obs)
        if self.use_rnn:
            action = self.controller(torch.cat((latent_mu, self.rnn_hidden_state[0]), dim=1))
            _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, self.rnn_hidden_state)
            self.rnn_hidden_state = next_hidden

        else:
            action = self.controller(latent_mu)
        return action.squeeze().cpu().numpy()

    def rollout(self, params):
        """ Execute a rollout and returns minus cumulative reward.

        """
        # copy params into the controller
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
