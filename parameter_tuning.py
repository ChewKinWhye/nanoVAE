from tensorflow import keras
import numpy as np
from utils.data import load_dna_data_vae
from utils.arguments import parse_args
from utils.evaluate import compute_metrics_standardized, plot_label_clusters, print_results
from utils.model import VaeDNA, load_vae_predictor
from random import uniform


if __name__ == "__main__":
    args = parse_args()
    x_train, y_train, x_test, y_test, _, _ = load_dna_data_vae(args.data_size, args.data_path)
    best_accuracy = float("-inf")
    num_experiments = 10
    for i in range(num_experiments):
        rc_scale = uniform(1, 10)
        mean_scale = uniform(1, 10)
        std_scale = uniform(1, 10)
        len_scale = uniform(1, 10)
        signal_scale = uniform(10, 20)

        x_train_scaled = np.concatenate((x_train[:, 0:68], x_train[:, 68:85], x_train[:, 85:102]*std_scale,
                                         x_train[:, 102:119]*len_scale, x_train[:, 119:]*signal_scale), axis=1)
        x_test_scaled = np.concatenate((x_test[:, 0:68], x_test[:, 68:85], x_test[:, 85:102] * std_scale,
                                        x_test[:, 102:119] * len_scale, x_test[:, 119:] * signal_scale), axis=1)

        vae = VaeDNA(args.latent_dim, rc_scale)
        vae.compile(optimizer=keras.optimizers.Adam())
        vae.fit(x_train_scaled, epochs=args.vae_epochs, batch_size=args.vae_batch_size, verbose=0)

        # Visualize cluster
        # Train predictor
        predictor = load_vae_predictor(args.latent_dim)
        # Prepared predictor training input
        predictor_size = int(len(x_train)/10)
        x_train_mean, x_train_sd, _ = vae.encoder.predict(x_train_scaled[0:predictor_size])
        x_train_predictor = np.concatenate((x_train_mean, x_train_sd), axis=1)
        predictor.fit(x_train_predictor, y_train[0:predictor_size], epochs=args.predictor_epochs,
                      batch_size=args.predictor_batch_size, verbose=0)

        # Test model
        x_test_mean, x_test_sd, _ = vae.encoder.predict(x_test_scaled)
        x_test_predictor = np.concatenate((x_test_mean, x_test_sd), axis=1)
        predictions = predictor.predict(x_test_predictor)
        test_results = compute_metrics_standardized(predictions, y_test)
        if test_results[0] > best_accuracy:
            best_accuracy = test_results[0]
            best_params = (rc_scale, mean_scale, std_scale, len_scale, signal_scale)
        print("---------------")
        print(rc_scale, mean_scale, std_scale, len_scale, signal_scale)
        print(test_results[0])
    print(best_params)
    print(best_accuracy)
