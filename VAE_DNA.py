import numpy as np
from utils.data import load_dna_data_vae, load_multiple_reads_data
from utils.arguments import parse_args
from utils.evaluate import compute_metrics_standardized, plot_label_clusters, print_results
from utils.model import load_vae_predictor, load_vae_dna_model
from utils.save import save_results
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import time
import tensorflow as tf
import tensorflow_addons as tfa


if __name__ == "__main__":
    print("Starting")
    args = parse_args()
    supervised_size = 10000
    NAME = f"VAE_DNA-{int(time.time())}"
    x_train, y_train, x_test, y_test, x_val, y_val = load_dna_data_vae(args.data_size, args.data_path, args.feature_scale)
    # Train VAE
    encoder, decoder, vae = load_vae_dna_model(args.latent_dim, args.rc_loss_scale, args.vae_lr)
    es = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    tensorboard = TensorBoard(log_dir=f"logs/{NAME}")
    vae.fit(x_train[0:int(len(x_train)*0.8)], validation_data=(x_train[int(len(x_train)*0.8):], None), epochs=args.vae_epochs, batch_size=args.vae_batch_size, verbose=2, callbacks=[es, tensorboard])

    #encoder.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tfa.losses.TripletSemiHardLoss())
    #encoder.fit(x_train[0:supervised_size], y_train[0:supervised_size], epochs=15)

    # Visualize cluster
    encoding_cluster_plt = plot_label_clusters(encoder, x_train, y_train)
    
    # Train predictor
    predictor = load_vae_predictor(args.latent_dim)
    # Prepared predictor training input
    x_train_predictor = encoder.predict(x_train[0:supervised_size])
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
    predictor.fit(x_train_predictor, y_train[0:supervised_size], epochs=args.predictor_epochs, validation_split=0.2,
                  batch_size=args.predictor_batch_size, callbacks=[es])

    # Test model
    x_test_predictor = encoder.predict(x_test)
    predictions = predictor.predict(x_test_predictor)
    test_results = compute_metrics_standardized(predictions, y_test)
    print_results(test_results)
    save_results(args.output_filename, test_results, encoding_cluster_plt, encoder, predictor)
    # Test model with multiple reads
    '''
    x_test_10, y_test_10 = load_multiple_reads_data(args.data_size, args.data_path, args.feature_scale)
    predictions = []
    for x in x_test_10:
        x_test_predictor = encoder.predict(x)
        x_test_prediction = predictor.predict(x_test_predictor)
        predictions.append(np.average(x_test_prediction))
    test_results_10 = compute_metrics_standardized(np.asarray(predictions), y_test_10)
    print_results(test_results_10)
    '''

