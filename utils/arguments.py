import argparse


def parse_args():
    parser = argparse.ArgumentParser('Train your VAE')
    parser.add_argument('--data_path', type=str, default='/hdd/modifications/ecoli/deepsignal', help='path to data directory')
    parser.add_argument('--output_filename', type=str, default='vae_model', help='name of the output file')

    # Other hyper-parameters
    parser.add_argument('--vae_epochs', type=int, default=300, help='Number of epochs to train VAE')
    parser.add_argument('--vae_batch_size', type=int, default=256, help='Batch size of VAE training')
    parser.add_argument('--predictor_epochs', type=int, default=300, help='Number of epochs to train VAE')
    parser.add_argument('--predictor_batch_size', type=int, default=32, help='Batch size of predictor training')
    parser.add_argument('--latent_dim', type=int, default=80, help='Latent dimension of encoding space')
    parser.add_argument('--data_size', type=int, default=500000, help='size of dataset to use')
    #12.95
    parser.add_argument('--rc_loss_scale', type=float, default=13, help='Scale factor of reconstruction loss')
    # 9.28, 11.83, 16.45, 16.55, 23.55
    parser.add_argument('--feature_scale', type=float, nargs=5, default=(10, 10, 16, 51, 25),
                        help='Scale factor of the features, k-mer, mean, std, len, signals respectively')
    parser.add_argument('--kmer_loss_scale', type=float, default=2000, help='Scale factor of kmer_loss')
    parser.add_argument('--vae_lr', type=float, default=0.00028874, help='Learning rate of vae')
    args = parser.parse_args()
    return args
