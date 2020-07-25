# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as np
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

import torch, argparse

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLPAutoencoder, MLP
from hnn import HNN, PixelHNN
from data import get_measurments, get_dataset_clean, hamiltonian_fn_torch
from utils import L2_loss

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=8, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--latent_dim', default=1, type=int, help='pairs of coords in latent dimension of autoencoder')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--input_noise', default=0.0, type=float, help='std of noise added to HNN inputs')
    parser.add_argument('--batch_size', default=250, type=int, help='batch size')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=1000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--name', default='vibration_machine', type=str, help='either "real" or "sim" data')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--fitted', dest='fitted', action='store_true', help='demeaned data?')
    parser.set_defaults(feature=True)
    return parser.parse_args()

'''The loss for this model is a bit complicated, so we'll
    define it in a separate function for clarity.'''
def pixelhnn_loss(x, x_next, model, return_scalar=True):
  # encode pixel space -> latent dimension
  z = model.encode(x)
  z_next = model.encode(x_next)

  # autoencoder loss
  x_hat = model.decode(z)
  ae_loss = ((x - x_hat)**2).mean(1)

  # hnn vector field loss
  noise = args.input_noise * torch.randn(*z.shape)
  z_hat_next = z + model.time_derivative(z + noise) # replace with rk4
  hnn_loss = ((z_next - z_hat_next)**2).mean(1)

  # canonical coordinate loss
  # -> makes latent space look like (x, v) coordinates
  w, dw = z.split(args.latent_dim,1)
  w_next, _ = z_next.split(args.latent_dim,1)
  cc_loss = ((dw-(w_next - w))**2).mean(1)

  #energy loss
  energy_loss = (hamiltonian_fn_torch(z_next) - hamiltonian_fn_torch(z)).mean(1)

  # sum losses and take a gradient step
  loss = ae_loss + cc_loss + 1e-1 * hnn_loss + energy_loss
  if return_scalar:
    return loss.mean()
  return loss

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  autoencoder = MLPAutoencoder(args.input_dim, args.hidden_dim, args.latent_dim*2)
  model = PixelHNN(args.latent_dim*2, args.hidden_dim,
                   autoencoder=autoencoder, nonlinearity=args.nonlinearity)
  if args.verbose:
    print("Training fitted model:" if args.fitted else "Training noisy model:")
    print('{} pairs of coords in latent space '.format(args.latent_dim))
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-5)

  # get dataset
  if args.fitted:
      x_m = get_dataset_clean('NORMAL')

      x = torch.tensor(x_m[:-1], dtype=torch.float)
      x_next = torch.tensor(x_m[1:], dtype=torch.float)

      # above2_dm = x_m[x_m[:, 0] >= 2, :]
      # below0_dm = x_m[x_m[:, 0] <= 0, :]
      # high_energy= torch.tensor(above2_dm[:-1], dtype=torch.float)
      # high_energy_next = torch.tensor(above2_dm[1:], dtype=torch.float)
      #
      # low_energy = torch.tensor(below0_dm[:-1], dtype=torch.float)
      # low_energy_next = torch.tensor(below0_dm[1:], dtype=torch.float)

      test_x = torch.tensor(x_m[:-1], dtype=torch.float)
      test_next_x = torch.tensor(x_m[1:], dtype=torch.float)

  else:
      x_m= get_measurments('NORMAL')

      x = torch.tensor(x_m[:-1], dtype=torch.float)
      x_next = torch.tensor(x_m[1:], dtype=torch.float)

      # above4 = x_m[x_m[:, 0] >= 4, :]
      # below0 = x_m[x_m[:, 0] <= 0, :]
      # high_energy= torch.tensor(above4[:-1], dtype=torch.float)
      # high_energy_next = torch.tensor(above4[1:], dtype=torch.float)
      #
      # low_energy = torch.tensor(below0[:-1], dtype=torch.float)
      # low_energy_next = torch.tensor(below0[1:], dtype=torch.float)

      test_x = torch.tensor(x_m[:-1], dtype=torch.float)
      test_next_x = torch.tensor(x_m[1:], dtype=torch.float)

  # vanilla ae train loop
  stats = {'train_loss': [], 'test_loss': []}
  for step in range(args.total_steps+1):
    
    # train step
    ixs_low = torch.randperm(x.shape[0])[:args.batch_size]
    loss = pixelhnn_loss(x[ixs_low], x_next[ixs_low], model)
    loss.backward() ; optim.step() ; optim.zero_grad()

    stats['train_loss'].append(loss.item())
    if args.verbose and step % args.print_every == 0:
      # run validation
      test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
      test_loss = pixelhnn_loss(test_x[test_ixs], test_next_x[test_ixs], model)
      stats['test_loss'].append(test_loss.item())

      print("step {}, train_loss {:.4e}, test_loss {:.4e}"
        .format(step, loss.item(), test_loss.item()))

  test_dist = pixelhnn_loss(test_x, test_next_x, model, return_scalar=False)
  print('Final test loss {:.4e} +/- {:.4e}'
    .format(test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))
  return model

if __name__ == "__main__":
    args = get_args()
    model = train(args)

    if args.fitted:
        data = get_dataset_clean('NORMAL')
    else:
        data = get_measurments('NORMAL')

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    demeaned = '-demeaned' if args.fitted else '-noisy'
    path = '{}/{}-{}-pairs{}.tar'.format(args.save_dir, args.name, args.latent_dim, demeaned)
    torch.save(model.state_dict(), path)
