import numpy as np
import csv
import os
import random
from sklearn.utils import shuffle
from math import sqrt

def standardize_and_scale_data(non_modified_data, modified_data, feature_scale, standardize_scale=None):
    if standardize_scale is None:
        kmer_mean = (np.mean(non_modified_data[:, 0:68].flatten()) + np.mean(modified_data[:, 0:68].flatten())) / 2
        kmer_mean_squared = (np.mean(np.square(non_modified_data[:, 0:68].flatten())) + np.mean(np.square(modified_data[:, 0:68].flatten()))) / 2
        kmer_std = sqrt(kmer_mean_squared - kmer_mean**2)
        mean_mean = (np.mean(non_modified_data[:, 68:85].flatten()) + np.mean(modified_data[:, 68:85].flatten())) / 2
        mean_mean_squared = (np.mean(np.square(non_modified_data[:, 68:85].flatten())) + np.mean(np.square(modified_data[:, 68:85].flatten()))) / 2
        mean_std = sqrt(mean_mean_squared - mean_mean**2)
        std_mean = (np.mean(non_modified_data[:, 85:102].flatten()) + np.mean(modified_data[:, 85:102].flatten())) / 2
        std_mean_squared = (np.mean(np.square(non_modified_data[:, 85:102].flatten())) + np.mean(np.square(modified_data[:, 85:102].flatten()))) / 2
        std_std = sqrt(std_mean_squared - std_mean**2)
        length_mean = (np.mean(non_modified_data[:, 102:119].flatten()) + np.mean(modified_data[:, 102:119].flatten())) / 2
        length_mean_squared = (np.mean(np.square(non_modified_data[:, 102:119].flatten())) + np.mean(np.square(modified_data[:, 102:119].flatten()))) / 2
        length_std = sqrt(length_mean_squared - length_mean**2)
        signal_mean = (np.mean(non_modified_data[:, 119:].flatten()) + np.mean(modified_data[:, 119:].flatten())) / 2
        signal_mean_squared = (np.mean(np.square(non_modified_data[:, 119:].flatten())) + np.mean(np.square(modified_data[:, 119:].flatten()))) / 2
        signal_std = sqrt(signal_mean_squared - signal_mean**2)
        standardize_scale = [kmer_mean, kmer_std, mean_mean, mean_std, std_mean, std_std, length_mean, length_std, signal_mean, signal_std]
    
    non_modified_data[:, 0:68] = (non_modified_data[:, 0:68] - standardize_scale[0]) / standardize_scale[1] * feature_scale[0]
    modified_data[:, 0:68] = (modified_data[:, 0:68] - standardize_scale[0]) / standardize_scale[1] * feature_scale[0]
    
    non_modified_data[:, 68:85] = (non_modified_data[:, 68:85] - standardize_scale[2]) / standardize_scale[3] * feature_scale[1]
    modified_data[:, 68:85] = (modified_data[:, 68:85] - standardize_scale[2]) / standardize_scale[3] * feature_scale[1]

    non_modified_data[:, 85:102] = (non_modified_data[:, 85:102] - standardize_scale[4]) / standardize_scale[5] * feature_scale[2]
    modified_data[:, 85:102] = (modified_data[:, 85:102] - standardize_scale[4]) / standardize_scale[5] * feature_scale[2]

    non_modified_data[:, 102:119] = (non_modified_data[:, 102:119] - standardize_scale[6]) / standardize_scale[7] * feature_scale[3]
    modified_data[:, 102:119] = (modified_data[:, 102:119] - standardize_scale[6]) / standardize_scale[7] * feature_scale[3]
    
    non_modified_data[:, 119:] = (non_modified_data[:, 119:] - standardize_scale[8]) / standardize_scale[9] * feature_scale[4]
    modified_data[:, 119:] = (modified_data[:, 119:] - standardize_scale[8]) / standardize_scale[9] * feature_scale[4]
    
    return non_modified_data.tolist(), modified_data.tolist(), standardize_scale


def check_data(row):
    signal_float = [float(i) for i in row[10].split(",")]
    len_float = [float(i) for i in row[9].split(",")]
    sd_float = [float(i) for i in row[8].split(",")]
    # Check for data outliers
    if max(signal_float) > 8 or min(signal_float) < -8 or max(len_float) > 300 or max(sd_float) > 2:
        return False
    # Check for data errors
    if row[5].lower() == 'c':
        return False
#    if row[6][6:11] != "ATCGA":
#        return False
    if np.any(np.isnan(signal_float)) or np.any(np.isnan(len_float)) or np.any(np.isnan(sd_float)):
        return False
    return True


def process_data(row):
    dna_lookup = {"A": [0, 0, 0, 1], "T": [0, 0, 1, 0], "G": [0, 1, 0, 0], "C": [1, 0, 0, 0]}
    row_data = []
    for i in row[6]:
        row_data.extend([j for j in dna_lookup[i]])
    row_data.extend([float(i) for i in row[7].split(",")])
    row_data.extend([float(i) for i in row[8].split(",")])
    row_data.extend([float(i) for i in row[9].split(",")])
    row_data.extend([float(i) for i in row[10].split(",")])
    row_data_float = [float(i) for i in row_data]
    return row_data_float


def load_dna_data_vae(data_size, data_path, feature_scale):
    # Global parameters
    train_size = int(data_size * 0.8 / 2)
    test_size = int(data_size * 0.1 / 2)
    val_size = int(data_size * 0.1 / 2)
    total_size = train_size + test_size + val_size
    file_path_normal = os.path.join(data_path, "pcr.tsv")
    file_path_modified = os.path.join(data_path, "msssi.tsv")

    # Extract data from non-modified
    with open(file_path_normal) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        non_modified_data = []
        data_count, outlier_counter = 0, 0
        for row in read_tsv:
            if data_count == total_size:
                break
            if check_data(row) is False:
                outlier_counter += 1
                continue
            row_data = process_data(row)
            non_modified_data.append(row_data)
            data_count += 1

    # Extract data from modified
    with open(file_path_modified) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        modified_data = []
        data_count = 0
        for i, row in enumerate(read_tsv):
            if data_count == total_size:
                break
            if check_data(row) is False:
                outlier_counter += 1
                continue
            row_data = process_data(row)
            modified_data.append(row_data)
            data_count += 1

    print(f"Number of outliers: {outlier_counter}")
    non_modified_data, modified_data, standardize_scale = standardize_and_scale_data(np.asarray(non_modified_data), np.asarray(modified_data), feature_scale)
    standardize_scale = None
    random.shuffle(non_modified_data)
    random.shuffle(modified_data)

    #train_x = modified_data[0:train_size]
    #train_x.extend(non_modified_data[0:train_size])
    train_x = non_modified_data[0:train_size]
    train_x = np.asarray(train_x)
    #train_y = np.append(np.ones(train_size), np.zeros(train_size))
    train_y = np.zeros(train_size)
    train_x, train_y = shuffle(train_x, train_y, random_state=0)

    test_x = modified_data[train_size:train_size + test_size]
    test_x.extend(non_modified_data[train_size:train_size + test_size])
    test_x = np.asarray(test_x)
    test_y = np.append(np.ones(test_size), np.zeros(test_size))
    test_x, test_y = shuffle(test_x, test_y, random_state=0)

    val_x = modified_data[train_size + test_size:]
    val_x.extend(non_modified_data[train_size + test_size:])
    val_x = np.asarray(val_x)
    val_y = np.append(np.ones(val_size), np.zeros(val_size))

    print(f"Train data shape: {train_x.shape}")
    print(f"Train data labels shape: {train_y.shape}")
    print(f"Test data shape: {test_x.shape}")
    print(f"Test data labels shape: {test_y.shape}")
    print(f"Validation data shape: {val_x.shape}")
    print(f"Validation data labels shape: {val_y.shape}")

    return train_x, train_y.astype(int), test_x, test_y.astype(int), val_x, val_y.astype(int), standardize_scale


def load_multiple_reads_data(data_size, data_path, feature_scale, standardize_scale):
    test_size = 10000
    total_size = 1000000
    # Global parameters
    file_path_normal = os.path.join(data_path, "pcr.tsv")
    file_path_modified = os.path.join(data_path, "msssi.tsv")

    non_modified_duplicate = {}
    test_x = []
    # Extract data from non-modified
    with open(file_path_normal) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        data_count = 0
        for index, row in enumerate(read_tsv):
            # Ignore the first x data points used for training, and check data
            if index <= data_size / 2 or check_data(row) is False:
                continue
            if data_count == total_size:
                break
            if row[3] not in non_modified_duplicate:
                non_modified_duplicate[row[3]] = []
            # Append data
            row_data = process_data(row)
            non_modified_duplicate[row[3]].append(row_data)
            data_count += 1

    # Find the ones with more than 10 reads
    for x in non_modified_duplicate:
        if len(non_modified_duplicate[x]) >= 10:
            test_x.append(non_modified_duplicate[x][0:10])
    non_modified_duplicate.clear()
    print(len(test_x))
    test_x = test_x[0:test_size]

    modified_duplicate = {}
    # Extract data from modified
    with open(file_path_modified) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        data_count = 0
        for index, row in enumerate(read_tsv):
            # Ignore the first x data points used for training, and check data
            if index <= data_size / 2 or check_data(row) is False:
                continue
            if data_count == total_size:
                break
            if row[3] not in modified_duplicate:
                modified_duplicate[row[3]] = []
            row_data = process_data(row)
            modified_duplicate[row[3]].append(row_data)
            data_count += 1

    # Find the ones with more than 10 reads
    for x in modified_duplicate:
        if len(modified_duplicate[x]) >= 10:
            test_x.append(modified_duplicate[x][0:10])
    print(len(test_x))
    test_x[0:test_size], test_x[test_size:], _ = standardize_and_scale_data(test_x[0:test_size], test_x[test_size:], feature_scale, standardize_scale) 
    test_x = test_x[0:2 * test_size]
    test_x = np.asarray(test_x)
    test_y = np.append(np.zeros(test_size), np.ones(test_size))
    return test_x, test_y
