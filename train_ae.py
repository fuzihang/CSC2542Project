
from torch.utils.data import Dataset, DataLoader

from doom_dataset import DoomDataset
from misc import DEVICE
from AE import *

EPOCHS = 10
TRAIN_DIR = '/home/zihang/Documents/npz_data'

def train_vae(model, iterator, opt, start_time):
    model.train()
    losses = []

    for i_batch, batch in enumerate(iterator):
        x_batch = batch.to(DEVICE).squeeze(0)

        x_batch = x_batch.to(DEVICE).float()

        opt.zero_grad()
        out, z_batch, mean, log_var = model(x_batch)

        loss = compute_loss(x_batch, z_batch, mean, log_var, out)
        loss.backward()
        losses.append(loss.item())
        opt.step()

        import math
        PRINT_INTERVAL = math.ceil(len(iterator) / 100)
        if (i_batch + 1) % PRINT_INTERVAL == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {}\t Time: {:10.3f}'.format(
                i_batch, len(iterator),
                i_batch / len(iterator) * 100,
                np.asarray(losses)[-PRINT_INTERVAL:].mean(0),
                time.time() - start_time,
            ))

    return np.asarray(losses).mean()

def val_loss(model, iterator):
    model.eval()
    losses = []
    for i_batch, batch in enumerate(iterator):
        x_batch = batch.squeeze(0)

        x_batch = x_batch.to(DEVICE).float()

        out, z_batch, mean, log_var = model(x_batch)

        loss = compute_loss(x_batch, z_batch, mean, log_var, out)
        losses.append(loss.item())

    return np.mean(losses)

if __name__ == '__main__':
    import time
    model = AE().to(DEVICE)
    model.cuda()
    opt = torch.optim.Adam(model.parameters())
    training_dataset = DoomDataset(TRAIN_DIR, 3)
    training_iterator = DataLoader(training_dataset, batch_size=1, shuffle=True)


    for epoch in range(EPOCHS):
        print(epoch)
        train_vae(model, training_iterator, opt, time.time())


        torch.save(model.state_dict(), 'vae_final.weights')

    torch.save(model.state_dict(), 'vae_final.weights')