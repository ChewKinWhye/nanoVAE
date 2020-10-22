from utils.data import load_dna_data_vae, load_multiple_reads_data
from utils.arguments import parse_args
from utils.model import load_explainable_doc_model
from utils.save import save_results
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import time
from utils.evaluate import compute_metrics_standardized, plot_label_clusters, print_results
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import itertools
from sklearn.utils import shuffle
import random


def add_noise(clean_data):
    noise = np.random.normal(0, 0.1, clean_data.shape[0]*clean_data.shape[1])\
        .reshape((clean_data.shape[0], clean_data.shape[1]))
    noisy_data = clean_data + noise
    y_train_noisy = np.ones(clean_data.shape[0])
    return noisy_data, y_train_noisy

def add_noise_2(clean_data):
    noisy_data = np.zeros((clean_data.shape[0], clean_data.shape[1]))
    for i in range(clean_data.shape[0]):
        noise_mean = np.random.normal(0, 0.1, 1)
        noise = np.random.normal(noise_mean, 0.1, clean_data.shape[1])
        noisy_data[i] = clean_data[i] + noise
    y_train_noisy = np.ones(clean_data.shape[0])
    return noisy_data, y_train_noisy

def add_noise_3(clean_data):
    noisy_data = np.zeros((clean_data.shape[0], clean_data.shape[1]))
    for i in range(clean_data.shape[0]):
        noise_mean = np.random.normal(0, 0.1, 18)
        for ii in range(18):
            noise = np.random.normal(noise_mean[ii], 0.1, 20)
            noisy_data[i, ii*20:ii*20+20] = clean_data[i, ii*20:ii*20+20] + noise
    y_train_noisy = np.ones(clean_data.shape[0])
    return noisy_data, y_train_noisy


def add_noise_4(clean_data):
    noisy_data = np.zeros((clean_data.shape[0], clean_data.shape[1]))
    noise_sampling_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    #noise_sampling_mean = np.array([0.02, 0.04, 0.06, 0.07, 0.10, 0.12, 0.14, 0.16, 0.18, 0.5, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02])
    neg_noise_sampling_mean = -noise_sampling_mean
    for i in range(clean_data.shape[0]):
        for ii in range(18):
            pos_noise = np.random.normal(noise_sampling_mean[ii], 0.1, 20)
            neg_noise = np.random.normal(neg_noise_sampling_mean[ii], 0.1, 20)
            noise = random.choice([pos_noise, neg_noise])
            noisy_data[i, ii*20:ii*20+20] = clean_data[i, ii*20:ii*20+20] + noise
    y_train_noisy = np.ones(clean_data.shape[0])
    return noisy_data, y_train_noisy

            
if __name__ == "__main__":
    args = parse_args()
    #x_train contains only non-modified
    x_train, y_train, x_test, y_test, x_val, y_val, standardize_scale = load_dna_data_vae(args.data_size, args.data_path, args.feature_scale)
    #Obtain only signal values
    x_train, x_test, x_val = x_train[:, 119:], x_test[:, 119:], x_val[:, 119:]

    x_train_noisy, y_train_noisy = add_noise_4(x_train)
    x_train_total = np.concatenate((x_train, x_train_noisy), axis=0)
    y_train_total = np.concatenate((y_train, y_train_noisy), axis=0)
    x_train_total, y_train_total = shuffle(x_train_total, y_train_total, random_state=0)
    model = load_explainable_doc_model()
    model.summary()
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
    model.fit(x_train_total, y_train_total, validation_split=0.2, epochs=args.vae_epochs, batch_size=args.vae_batch_size, verbose=2, callbacks=[es])
    
    test_predictions = model.predict(x_test)
    noise_predictions = np.average(model.predict(x_train_noisy), axis=1)

    non_mod_predictions = np.average(test_predictions[0:int(test_predictions.shape[0]/2)], axis=1)
    mod_predictions = np.average(test_predictions[int(test_predictions.shape[0]/2):], axis=1)
    print(f"Noise median: {np.median(noise_predictions)}")
    print(f"Non-mod median: {np.median(non_mod_predictions)}")
    print(f"Non-mod std: {np.std(non_mod_predictions)}")
    print(f"Mod median: {np.median(mod_predictions)}")
    print(f"Mod std: {np.std(mod_predictions)}")
    threshold = (np.median(non_mod_predictions) + np.median(mod_predictions)) / 2
     
    print(f"Threshold: {threshold}")
    predictions = []
    for score in test_predictions:
        total_score = np.average(score)
        if total_score > threshold:
            predictions.append(1)
        else:
            predictions.append(0)
    print(1 - accuracy_score(y_test, predictions))
