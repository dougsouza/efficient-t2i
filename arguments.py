import argparse
from pathlib import Path
from ganlib.priors import get_priors_names
from ganlib.losses import get_losses_names
from models import get_dmodels_names, get_gmodels_names
from ganlib.weight_init import get_initializer_names
from data_loaders import get_loader_names


def read_args():
    parser = argparse.ArgumentParser()
    train_args = parser.add_argument_group('Training', 'Training Parameters')
    train_args.add_argument('exp_name', type=str,
                            help='experiment name')
    train_args.add_argument('--dataset', type=str,
                        default='CUB', help='dataset name', 
                        choices=get_loader_names())
    train_args.add_argument('--from_file', type=Path,
                        default=None, help='import config from a json file')
    train_args.add_argument('--resume', type=Path,
                        default=None, help='path to a saved model')
    train_args.add_argument('--n_epochs', type=int,
                        default=500, help='number of epochs to run training')
    train_args.add_argument('--num_workers', type=int,
                        default=8, help='number of loader workers')
    train_args.add_argument('--batch_size', type=int,
                        default=16, help='input batch size')
    train_args.add_argument('--ngpu', type=int,
                        default=1, help='number of gpus to use')
    train_args.add_argument('--save_checkpoint_ep', type=int,
                        default=50,
                        help='number of steps to save checkpoints')
    train_args.add_argument('--weight_init', type=str,
                        default='orthogonal', choices=get_initializer_names(),
                        help='weight init stategy')
    train_args.add_argument('--save_sample_step', type=int,
                        default=500,
                        help='number of steps to save samples')
    train_args.add_argument('--n_samples', type=int,
                        default=16,
                        help='number of samples save')
    train_args.add_argument('--log_step', type=int,
                        default=50, help='number of steps to print status')
    train_args.add_argument('--loss', type=str,
                        default='hingegan', choices=get_losses_names(),
                        help='loss type')
    train_args.add_argument('--interp_sentences', default=False,
                        action='store_true',
                        help='wheter to interpolate sentence embeddings for training')
    train_args.add_argument('--resume_on_new_folder', default=False,
                        action='store_true',
                        help='wheter to to create a new folder for resumed training')
    train_args.add_argument('--local_rank', type=int, default=0)
    train_args.add_argument('--fp16', default=False,
                        action='store_true',
                        help='whether to to run mixed precision trnaing')
    train_args.add_argument('--opt_level', type=str, default="O3",
                        help='level of optmization (only for fp16 training)')


    d_model_args = parser.add_argument_group('D Model', 'D Parameters')
    d_model_args.add_argument('--d_model', type=str,
                        default='biggan_deep', choices=get_dmodels_names(),
                        help='discriminator model')
    d_model_args.add_argument('--ndf', type=int,
                        default=64, help='filter multiplier for D')
    d_model_args.add_argument('--d_lr', type=float,
                        default=0.0004, help='learning rate for D')
    d_model_args.add_argument('--d_steps', type=int,
                        default=1, help='number of D steps per G step')


    g_model_args = parser.add_argument_group('G Model', 'G Parameters')
    g_model_args.add_argument('--z_dim', type=int,
                        default=128, help='number of dimensions of latent vector')
    g_model_args.add_argument('--prior', type=str,
                        default='normal', choices=get_priors_names(),
                        help='Type of prior distribution to use for G')
    g_model_args.add_argument('--g_model', type=str,
                        default='biggan_deep', choices=get_gmodels_names(),
                        help='generator model')
    g_model_args.add_argument('--ngf', type=int,
                        default=64, help='filter multiplier for G')
    g_model_args.add_argument('--g_lr', type=float,
                        default=0.0001, help='learning rate for G')
    g_model_args.add_argument('--EMA_G', default=False,
                        action='store_true',
                        help='whether to keep an expontial moving average of G')
    g_model_args.add_argument('--normalize_prior', default=False,
                        action='store_true',
                        help='whether to normalize latent vector')
    g_model_args.add_argument('--g_norm', type=str,
                        default='batch', choices=['batch', 'instance'],
                        help='G normalization type (batch/instance)')
    args = parser.parse_args()
    return args
