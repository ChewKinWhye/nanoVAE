import argparse


def parse_args():
    parser = argparse.ArgumentParser('Train your VAE')
    parser.add_argument('--data_path', type=str, default='data', help='path to data directory')
    parser.add_argument('--output_filename', type=str, default='vae_model', help='name of the output file')

    # Other hyper-parameters
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=512, help='')
    parser.add_argument('--latent_dim', type=int, default=20, help='Latent dimension of encoding space')
    parser.add_argument('--data_size', type=int, default=200000, help='size of dataset to use')
    args = parser.parse_args()
    return args
