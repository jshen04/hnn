# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse
import os, sys
import autograd
import autograd.numpy as np
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLPAutoencoder, MLP
from hnn import HNN, PixelHNN
from data import get_dataset, get_dataset_split, hamiltonian_fn_torch

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=8, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--latent_dim', default=1, type=int, help='pairs of coords in latent dimension of autoencoder')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--input_noise', default=0.0, type=float, help='std of noise added to HNN inputs')
    parser.add_argument('--batch_size', default=50, type=int, help='batch size')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=5000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=500, type=int, help='number of gradient steps between prints')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--name', default='vibration_machine', type=str, help='motor vibration data')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--scaled', dest='scaled', action='store_true', help='demeaned data?')
    parser.add_argument('--folder', default='NORMAL', type=str, help='which folder is the dataset in')
    parser.add_argument('--speed', default=14.0, type=float, help='corresponds to the speed in the dataset')
    parser.add_argument('--split_data', default=250000, type=int, help='creating model of part of the data')
    parser.add_argument('--sub_columns', default=1, type=int, help='which set of columns to run model on')
    parser.add_argument('--cpu', dest='cpu', action='store_true', help='use cpu?')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def hnn_ae_loss(x, x_next, model, return_scalar=True):
    # encode pixel space -> latent dimension
    z = model.encode(x)
    z_next = model.encode(x_next)

    # autoencoder loss
    x_hat = model.decode(z)
    ae_loss = ((x - x_hat)**2).mean(1)

    # hnn vector field loss
    noise = args.input_noise * torch.randn(*z.shape)
    z_hat_next = z + model.time_derivative(z + noise)  # replace with rk4
    hnn_loss = ((z_next - z_hat_next)**2).mean(1)

    # canonical coordinate loss
    # -> makes latent space look like (x, v) coordinates
    w, dw = z.split(args.latent_dim, 1)
    w_next, _ = z_next.split(args.latent_dim, 1)
    cc_loss = ((dw-(w_next - w))**2).mean(1)

    #energy loss
    energy_loss = ((hamiltonian_fn_torch(z_next) - hamiltonian_fn_torch(z))**2).mean(1)

    # sum losses and take a gradient step
    loss = ae_loss + cc_loss + 1e-1 * hnn_loss + energy_loss
    if return_scalar:
        return loss.mean()
    return loss

def train(args):
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda:0")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.empty_cache()
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.verbose:
        print("Data from {} {}".format(args.folder, args.speed))

    # get dataset (no test data for now)
    angular_velo, acc_1, acc_2, sound = get_dataset_split(args.folder, args.speed, args.scaled)
    sub_col = {0: [angular_velo, 1, 'v'], 1: [acc_1, 3, 'a1'], 2: [acc_2, 3, 'a2'], 3: [sound, 1, 's']}
    col2use = sub_col[args.sub_columns][0]

    print(sub_col[args.sub_columns][2])

    x = torch.tensor(col2use[:-1], dtype=torch.float)
    x_next = torch.tensor(col2use[1:], dtype=torch.float)

    autoencoder = MLPAutoencoder(sub_col[args.sub_columns][1], args.hidden_dim, args.latent_dim * 2)
    model = PixelHNN(args.latent_dim * 2, args.hidden_dim,
                     autoencoder=autoencoder, nonlinearity=args.nonlinearity)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-5)

    # vanilla ae train loop
    stats = {'train_loss': []}
    for step in range(args.total_steps + 1):
        # train step
        ixs = torch.randperm(x.shape[0])[:args.batch_size]
        x_train, x_next_train = x[ixs].to(device), x_next[ixs].to(device)
        loss = hnn_ae_loss(x_train, x_next_train, model)
        loss.backward(); optim.step(); optim.zero_grad()

        stats['train_loss'].append(loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}"
                  .format(step, loss.item()))

    train_dist = hnn_ae_loss(x, x_next, model, return_scalar=False)
    print('Final train loss {:.4e} +/- {:.4e}'
          .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0])))
    return model

if __name__ == "__main__":
    # timing the model
    import time
    t0 = time.time()

    args = get_args()
    model = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    fixed = '-scaled' if args.scaled else ''
    split = '-{}-datapoints'.format(args.split_data) if args.split_data != 250000 else ''
    subcol = ['v', 'a1', 'a2', 's'][args.sub_columns]
    path = "{}/saved_models/motor-{}-{}-{}-pair{}{}-{}.tar".format(args.save_dir, args.folder, args.speed,
                                                                     args.latent_dim, fixed, split, subcol)
    torch.save(model.state_dict(), path)
    # timing the model
    t1 = time.time()
    total = t1 - t0
    print(total)