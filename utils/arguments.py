import argparse


def parse_args():
    parser = argparse.ArgumentParser('Train your VAE')
    parser.add_argument('--data_path', type=str, default='data', help='path to data directory')
    parser.add_argument('--output_filename', type=str, default='vae_model', help='name of the output file')

    # Other hyper-parameters
    parser.add_argument('--vae_epochs', type=int, default=100, help='Number of epochs to train VAE')
    parser.add_argument('--vae_batch_size', type=int, default=128, help='Batch size of VAE training')
    parser.add_argument('--predictor_epochs', type=int, default=30, help='Number of epochs to train VAE')
    parser.add_argument('--predictor_batch_size', type=int, default=128, help='Batch size of predictor training')
    parser.add_argument('--latent_dim', type=int, default=20, help='Latent dimension of encoding space')
    parser.add_argument('--data_size', type=int, default=500000, help='size of dataset to use')
    args = parser.parse_args()
    return args
