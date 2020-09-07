from tensorflow import keras
import tensorflow as tf
import numpy as np
from utils.data import load_dna_data_vae, load_multiple_reads_data
from utils.arguments import parse_args
from utils.evaluate import compute_metrics_standardized, plot_label_clusters, print_results
from utils.model import VaeDNA, load_vae_predictor, load_vae_dna_model
from utils.save import save_results
from tensorflow.keras.callbacks import EarlyStopping


if __name__ == "__main__":
    args = parse_args()

    x_train, y_train, x_test, y_test, x_val, y_val = load_dna_data_vae(args.data_size, args.data_path)
    # Train VAE
    # vae = VaeDNA(args.latent_dim, args.rc_loss_scale)
    encoder, decoder, vae = load_vae_dna_model(args.latent_dim, args.rc_loss_scale)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(x_train, epochs=args.vae_epochs, batch_size=args.vae_batch_size, verbose=2)

    # Visualize cluster
    encoding_cluster_plt = plot_label_clusters(encoder, x_train, y_train)

    # Train predictor
    predictor = load_vae_predictor(args.latent_dim)
    # Prepared predictor training input
    predictor_size = int(len(x_train)/10)
    x_train_mean, x_train_sd, _ = encoder.predict(x_train[0:predictor_size])
    x_train = np.concatenate((x_train_mean, x_train_sd), axis=1)
    predictor.fit(x_train, y_train[0:predictor_size], epochs=args.predictor_epochs,
                  batch_size=args.predictor_batch_size)

    # Test model
    x_test_mean, x_test_sd, _ = encoder.predict(x_test)
    x_test = np.concatenate((x_test_mean, x_test_sd), axis=1)
    predictions = predictor.predict(x_test)
    test_results = compute_metrics_standardized(predictions, y_test)
    print_results(test_results)

    save_results(args.output_filename, test_results, encoding_cluster_plt, encoder, predictor)
    # Test model with multiple reads
    x_test_10, y_test_10 = load_multiple_reads_data(args)
    predictions = []
    for x in x_test_10:
        x_test_mean, x_test_sd, _ = encoder.predict(x)
        x_test_predictor = np.concatenate((x_test_mean, x_test_sd), axis=1)
        x_test_prediction = predictor.predict(x_test_predictor)
        predictions.append(np.average(x_test_prediction))
    test_results_10 = compute_metrics_standardized(np.asarray(predictions), y_test_10)
    print_results(test_results_10)

