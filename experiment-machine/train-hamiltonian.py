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
from data import get_dataset, get_dataset_split, get_dataset_range
from utils import L2_loss

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim_ae', default=8, type=int, help='dimensionality of input tensor for autoencoder')
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor for HNN model')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--latent_dim', default=1, type=int, help='pairs of coords in latent dimension of autoencoder')
    parser.add_argument('--output_dim', default=2, type=int, help='dimensionality of output tensor')
    parser.add_argument('--ae_path', default='motor-NORMAL-14-1-pairs', type=str, help='autoencoder path')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--dropout_rate_ae', default=0.1, type=float, help='dropout rate for autoencoder')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate for hamiltonian model')
    parser.add_argument('--input_noise', default=0.0, type=float, help='std of noise added to HNN inputs')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=2500, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=500, type=int, help='number of gradient steps between prints')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--name', default='vibration_machine', type=str, help='motor vibration data')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--scaled', dest='scaled', action='store_true', help='demeaned data?')
    parser.add_argument('--folder', default='NORMAL', type=str, help='which folder is the dataset in')
    parser.add_argument('--experiment_dir', default='.', type=str, help='experiment directory, where the full dataset is')
    parser.add_argument('--speed', default='14', type=str, help='corresponds to the speed in the dataset')
    parser.add_argument('--split_data', default=250000, type=int, help='creating model of part of the data')
    parser.add_argument('--cpu', dest='cpu', action='store_true', help='use cpu?')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.set_defaults(feature=True)
    return parser.parse_args()

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

    print("{} {}".format(args.folder, args.speed))
    print("Training scaled model:" if args.scaled else "Training noisy model:")
    print('{} pairs of coords in latent space '.format(args.latent_dim))
    
    
    #using universal autoencoder, pre-encode the training points
    autoencoder = MLPAutoencoder(args.input_dim_ae, args.hidden_dim, args.latent_dim * 2, nonlinearity='relu')
    full_model = PixelHNN(args.latent_dim * 2, args.hidden_dim,
                     autoencoder=autoencoder, nonlinearity=args.nonlinearity, baseline=args.baseline)
    path = "{}/saved_models/{}.tar".format(args.save_dir, args.ae_path)
    full_model.load_state_dict(torch.load(path))
    full_model.eval()
    autoencoder_model = full_model.autoencoder
    
    # get dataset (no test data for now)
    data = get_dataset(args.folder, args.speed, scaled=args.scaled, split=args.split_data, experiment_dir=args.experiment_dir, tensor=True)
    gcoords = autoencoder_model.encode(data).cpu().detach().numpy()
    x = torch.tensor(gcoords, dtype=torch.float, requires_grad=True)
    dx_np = full_model.time_derivative(torch.tensor(gcoords, dtype=torch.float, requires_grad=True)).cpu().detach().numpy()
    dx = torch.tensor(dx_np, dtype=torch.float)
    
    nnmodel = MLP(args.input_dim,args.hidden_dim,args.output_dim)
    model = HNN(2, nnmodel)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=args.weight_decay)

    # vanilla ae train loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps + 1):

        # train step
        ixs = torch.randperm(x.shape[0])[:args.batch_size]
        x_train, dxdt = x[ixs].to(device), dx[ixs].to(device)
        dxdt_hat = model.time_derivative(x_train)

        loss = L2_loss(dxdt, dxdt_hat)
        loss.backward(); optim.step(); optim.zero_grad()

        stats['train_loss'].append(loss.item())

        if step % args.print_every == 0:
            print("step {}, train_loss {:.4e}"
                       .format(step, loss.item()))

    # train_dist = hnn_ae_loss(x, x_next, model, return_scalar=False)
    # print('Final train loss {:.4e} +/- {:.4e}'
    #       .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0])))
    return model


if __name__ == "__main__":
    args = get_args()
    model = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    fixed = '-scaled' if args.scaled else ''
    split = '-{}-datapoints'.format(args.split_data) if args.split_data != 250000 else ''
    baseline = '-baseline' if args.baseline else ''
    folder = args.folder
    folder = folder.replace('/', '')
    path = "{}/saved_models/motor-{}-{}-{}-pairs{}{}{}-hnn-{}.tar".format(args.save_dir, folder, args.speed,
                                                                     args.latent_dim, fixed, split, baseline, args.ae_path)
    torch.save(model.state_dict(), path)
