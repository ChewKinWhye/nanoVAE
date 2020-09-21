import argparse


def parse_args():
    parser = argparse.ArgumentParser('Train your VAE')
    parser.add_argument('--data_path', type=str, default='data', help='path to data directory')
    parser.add_argument('--output_filename', type=str, default='vae_model', help='name of the output file')

    # Other hyper-parameters
    parser.add_argument('--vae_epochs', type=int, default=300, help='Number of epochs to train VAE')
    parser.add_argument('--vae_batch_size', type=int, default=256, help='Batch size of VAE training')
    parser.add_argument('--predictor_epochs', type=int, default=300, help='Number of epochs to train VAE')
    parser.add_argument('--predictor_batch_size', type=int, default=32, help='Batch size of predictor training')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension of encoding space')
    parser.add_argument('--data_size', type=int, default=500000, help='size of dataset to use')
    parser.add_argument('--rc_loss_scale', type=float, default=3.425, help='Scale factor of reconstruction loss')
    parser.add_argument('--feature_scale', type=float, nargs=5, default=(16, 23, 15, 1.79, 29.88),
                        help='Scale factor of the features, k-mer, mean, std, len, signals respectively')
    parser.add_argument('--vae_lr', type=float, default=0.000686, help='Learning rate of vae')
    args = parser.parse_args()
    return args
