from tensorflow import keras
import tensorflow as tf
import numpy as np
from utils.data import load_dna_data_vae, load_multiple_reads_data
from utils.arguments import parse_args
from utils.evaluate import compute_metrics_standardized, plot_label_clusters, print_results
from utils.model import VaeDNA, load_vae_predictor
from utils.save import save_results
from tensorflow.keras.callbacks import EarlyStopping
from hyperopt import hp, tpe, Trials, fmin

args = parse_args()

x_train, y_train, x_test, y_test, x_val, y_val = load_dna_data_vae(args.data_size, args.data_path)

def objective(params):
    x_train_scaled = np.concatenate((x_train[:, 0:68], x_train[:, 68:85]*params['mean_scale'], x_train[:, 85:102]*params['std_scale'], x_train[:, 102:119]*params['len_scale'], x_train[:, 119:]*params['signal_scale']), axis=1)
    x_test_scaled = np.concatenate((x_test[:, 0:68], x_test[:, 68:85]*params['mean_scale'], x_test[:, 85:102]*params['std_scale'], x_test[:, 102:119]*params['len_scale'], x_test[:, 119:]*params['signal_scale']), axis=1)
    vae = VaeDNA(args.latent_dim, params['rc_scale'])
    vae.compile(optimizer=keras.optimizers.Adam())
    try:
        vae.fit(x_train_scaled, epochs=args.vae_epochs, batch_size=args.vae_batch_size, verbose=0)
        predictor = load_vae_predictor(args.latent_dim)
        # Prepared predictor training input
        predictor_size = int(len(x_train)/10)
        x_train_mean, x_train_sd, _ = vae.encoder.predict(x_train_scaled[0:predictor_size])
        x_train_pred = np.concatenate((x_train_mean, x_train_sd), axis=1)
        predictor.fit(x_train_pred, y_train[0:predictor_size], epochs=args.predictor_epochs, 
                batch_size=args.predictor_batch_size, verbose=0)
        # Test model
        x_test_mean, x_test_sd, _ = vae.encoder.predict(x_test_scaled)
        x_test_pred = np.concatenate((x_test_mean, x_test_sd), axis=1)
        predictions = predictor.predict(x_test_pred)
        test_results = compute_metrics_standardized(predictions, y_test)
        print_results(test_results)
        return test_results[0] * -1
    except:
        return 0

space = {
        'rc_scale': hp.uniform('rc_scale', 1, 15),
        'mean_scale': hp.uniform('mean_scale', 0, 20),
        'std_scale': hp.uniform('std_scale', 0, 20),
        'len_scale': hp.uniform('len_scale', 1, 3),
        'signal_scale': hp.uniform('signal_scale', 1, 20)
        }
bayes_trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials = bayes_trials)
print(best)
