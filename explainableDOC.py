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


def add_noise(clean_data):
    noise = np.random.normal(0, 0.1, clean_data.shape[0]*clean_data.shape[1])\
        .reshape((clean_data.shape[0], clean_data.shape[1]))
    noisy_data = clean_data + noise
    y_train_noisy = np.ones(clean_data.shape[0])
    return noisy_data, y_train_noisy


if __name__ == "__main__":
    args = parse_args()
    # x_train contains only non-modified
    # x_train, y_train, x_test, y_test, x_val, y_val, standardize_scale = load_dna_data_vae(args.data_size, args.data_path, args.feature_scale)
    # # Obtain only signal values
    # x_train, x_test, x_val = x_train[:, :119], x_test[:, :119], x_val[:, :119]
    # x_train_noisy, y_train_noisy = add_noise(x_train)
    model = load_explainable_doc_model()
