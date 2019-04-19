import numpy as np
from VAE import *
import scipy.misc as misc
import os
from doom_dataset import DoomDataset
from torch.utils.data import Dataset, DataLoader


if not os.path.exists('VAE_regen_img'):
    os.mkdir('VAE_regen_img')

dataset = DoomDataset('VAE_train_data')
data_iter = DataLoader(dataset, batch_size=10, shuffle=True)

data = enumerate(data_iter).__next__()[1].detach().cpu().numpy()

model = VAE().to(DEVICE)
model.load_state_dict(torch.load('vae_best_val_weights'))


for i in range(data.shape[0]):
    misc.imsave(os.path.join('VAE_regen_img', f'{i}.png'), np.moveaxis((data[i] * 255.0).astype(np.uint8), 0, 2))


    data_out = model(torch.Tensor(data[i]).to(DEVICE).unsqueeze(0))[0]

    output = data_out.detach().cpu().numpy()[0]
    misc.imsave(os.path.join('VAE_regen_img', f'{i}_recon.png'), (np.moveaxis(output, 0, 2) * 255.0).astype(np.uint8))
