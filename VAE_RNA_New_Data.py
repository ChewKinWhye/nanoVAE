from tensorflow import keras
import numpy as np
from utils.data import load_rna_data_vae_new
from utils.arguments import parse_args
from utils.evaluate import compute_metrics_standardized, plot_label_clusters, print_results
from utils.model import VaeRNA, load_vae_predictor
from utils.save import save_results

if __name__ == "__main__":
    args = parse_args()
    x_train, y_train, x_test, y_test, x_val, y_val = load_rna_data_vae_new(args.data_size, args.data_path)

    # Train encoder
    vae = VaeRNA(args.latent_dim, input_dim=x_train.shape[1])
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(x_train, epochs=args.vae_epochs, batch_size=args.vae_batch_size)

    # Visualize cluster
    plt = plot_label_clusters(vae.encoder, x_train, y_train)

    # Train predictor
    predictor = load_vae_predictor(args.latent_dim)
    x_train_mean, x_train_sd, _ = vae.encoder.predict(x_train[0:5000])
    x_train = np.concatenate((x_train_mean, x_train_sd), axis=1)
    predictor.fit(x_train, y_train[0:5000], epochs=args.predictor_epochs, batch_size=args.predictor_batch_size)

    # Test model
    x_test_mean, x_test_sd, _ = vae.encoder.predict(x_test)
    x_test = np.concatenate((x_test_mean, x_test_sd), axis=1)
    predictions = predictor.predict(x_test)
    results = compute_metrics_standardized(predictions, y_test)
    print_results(results)
    # Save model
    save_results(args.output_filename, results, plt, vae.encoder, predictor)
