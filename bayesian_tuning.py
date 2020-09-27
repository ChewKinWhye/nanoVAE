from tensorflow import keras
import tensorflow as tf
import numpy as np
from utils.data import load_dna_data_vae, load_multiple_reads_data
from utils.arguments import parse_args
from utils.evaluate import compute_metrics_standardized, plot_label_clusters, print_results
from utils.model import load_vae_predictor, load_vae_dna_model, load_vae_dna_model_deepsignal
from utils.save import save_results
from tensorflow.keras.callbacks import EarlyStopping
from hyperopt import hp, tpe, Trials, fmin

args = parse_args()

x_train, y_train, x_test, y_test, x_val, y_val, standardize_scale = load_dna_data_vae(args.data_size, args.data_path, [1, 1, 1, 1, 1])


def objective(params):
    print(params)
    x_train_scaled = np.concatenate((x_train[:, 0:68]*params['kmer_scale'], x_train[:, 68:85]*params['mean_scale'], x_train[:, 85:102]*params['std_scale'], x_train[:, 102:119]*params['len_scale'], x_train[:, 119:]*params['signal_scale']), axis=1)
    x_test_scaled = np.concatenate((x_test[:, 0:68]*params['kmer_scale'], x_test[:, 68:85]*params['mean_scale'], x_test[:, 85:102]*params['std_scale'], x_test[:, 102:119]*params['len_scale'], x_test[:, 119:]*params['signal_scale']), axis=1)
    encoder, decoder, vae = load_vae_dna_model_deepsignal(int(params['latent_dim']), params['rc_scale'], params['vae_lr'], params['kmer_loss_scale'])
    try:
        supervised_size = 10000
        es = EarlyStopping(monitor='val_loss', mode='min', patience=20)
        vae.fit(x_train_scaled[0:int(len(x_train) * 0.8)], validation_data=(x_train_scaled[int(len(x_train) * 0.8):], None), epochs=args.vae_epochs, batch_size=args.vae_batch_size, verbose=0, callbacks=[es])

        predictor = load_vae_predictor(int(params['latent_dim']))
        # Prepared predictor training input
        x_train_predictor = encoder.predict(x_train_scaled[0:supervised_size])

        es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
        predictor.fit(x_train_predictor, y_train[0:supervised_size], epochs=args.predictor_epochs, validation_split=0.2,
                      batch_size=args.predictor_batch_size, callbacks=[es], verbose=0)

        # Test model
        x_test_predictor = encoder.predict(x_test_scaled)
        predictions = predictor.predict(x_test_predictor)
        test_results = compute_metrics_standardized(predictions, y_test)
        if test_results[0] < 0.51:
            encoding_cluster_plt = plot_label_clusters(encoder, x_train, y_train)
            save_results(args.output_filename, test_results, encoding_cluster_plt, encoder, predictor)
        print_results(test_results)
        return test_results[0] * -1
    except:
        return 0

space = {
    'rc_scale': hp.uniform('rc_scale', 1, 15),
    'kmer_scale': hp.uniform('kmer_scale', 0, 20),
    'mean_scale': hp.uniform('mean_scale', 10, 30),
    'std_scale': hp.uniform('std_scale', 0, 20),
    'len_scale': hp.uniform('len_scale', 0, 20),
    'signal_scale': hp.uniform('signal_scale', 10, 30),
    'latent_dim': hp.quniform('latent_dim', 0, 150, 5),
    'vae_lr': hp.loguniform('vae_lr', np.log(0.0001), np.log(0.01)),
    'kmer_loss_scale': hp.uniform('kmer_loss_scale', 0, 15)
    }
bayes_trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=bayes_trials)
print(best)
