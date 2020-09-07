import numpy as np
import csv
import os
import random
from sklearn import preprocessing
from sklearn.utils import shuffle
import pandas as pd
import imblearn
from sklearn.model_selection import train_test_split


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
    if row[6][6:11] != "ATCGA":
        return False
    return True


def process_data(row):
    dna_lookup = {"A": [0, 0, 0, 1], "T": [0, 0, 1, 0], "G": [0, 1, 0, 0], "C": [1, 0, 0, 0]}
    row_data = []
    for i in row[6]:
        row_data.extend(dna_lookup[i])
    row_data.extend([float(i) for i in row[7].split(",")])
    row_data.extend([float(i) for i in row[8].split(",")])
    row_data.extend([float(i) for i in row[9].split(",")])
    row_data.extend([float(i) for i in row[10].split(",")])
    row_data_float = [float(i) for i in row_data]
    return row_data_float


def load_rna_data_vae_new(data_size, data_path):
    train_size = int(data_size * 0.8)
    test_size = int(data_size * 0.1)
    file_path = os.path.join(data_path, "ecoli_MSssI_50mil_extracted_features.tsv")
    df = pd.read_csv(file_path, sep='\t', error_bad_lines=False)
    # Label by motif
    # CpG
    bool = []
    for row in df.kmer:
        if row[5] == "C" and row[6] == "G":
            bool.append(True)
        else:
            bool.append(False)
    df.loc[bool, 'y'] = 1
    # non-CpG
    df.loc[[not i for i in bool], 'y'] = 0
    # Label proportions
    rus = imblearn.under_sampling.RandomUnderSampler(random_state=66)
    df, _ = rus.fit_resample(df, df.y)
    df.y.value_counts()
    print(df.y.value_counts())
    X = df.drop(['y', 'chrom', 'kmer', 'ref_pos0'], axis=1).copy().to_numpy()
    y = df.y.copy().to_numpy()
    print(type(X[0][0]))
    print(y[0])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=40)

    return x_train, y_train, x_test, y_test, x_val, y_val


def load_rna_data_vae(data_size, data_path):
    train_size = int(data_size * 0.8)
    test_size = int(data_size * 0.1)
    file_path_normal = os.path.join(data_path, "ecoli_MSssI_50mil_coverage10_readqual_extracted.tsv")
    file_path_modified = os.path.join(data_path, "ecoli_pcr_50mil_coverage10_readqual_extracted.tsv")
    X = []
    Y = []
    with open(file_path_normal) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for i, row in enumerate(read_tsv):
            if i == 0:
                continue
            if i == int(data_size/2) + 1:
                break
            Y.append(int(0))
            row_float = [float(x) for x in row[3:]]
            X.append(row_float)

    with open(file_path_modified) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for i, row in enumerate(read_tsv):
            if i == 0:
                continue
            if i == int(data_size/2) + 1:
                break
            Y.append(int(1))
            row_float = [float(x) for x in row[3:]]
            X.append(row_float)
    # Normalize between 0 and 1
    min_max_scalar = preprocessing.MinMaxScaler()
    X = min_max_scalar.fit_transform(np.asarray(X))
    Y = np.asarray(Y)
    X, Y = shuffle(X, Y, random_state=0)
    x_train = X[0:train_size, :]
    y_train = Y[0:train_size]
    x_test = X[train_size: train_size + test_size, :]
    y_test = Y[train_size: train_size + test_size]

    x_val = X[train_size + test_size:, :]
    y_val = Y[train_size + test_size:]

    print(f"Train data shape: {x_train.shape}")
    print(f"Train data labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test data labels shape: {y_test.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print(f"Validation data labels shape: {y_val.shape}")

    return x_train, y_train, x_test, y_test, x_val, y_val


def load_dna_data_vae(data_size, data_path):
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

    random.shuffle(non_modified_data)
    random.shuffle(modified_data)

    train_x = modified_data[0:train_size]
    train_x.extend(non_modified_data[0:train_size])
    train_x = np.asarray(train_x)
    train_y = np.append(np.ones(train_size), np.zeros(train_size))
    train_x, train_y = shuffle(train_x, train_y, random_state=0)

    test_x = modified_data[train_size:train_size + test_size]
    test_x.extend(non_modified_data[train_size:train_size + test_size])
    test_x = np.asarray(test_x)
    test_y = np.append(np.ones(test_size), np.zeros(test_size))

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

    return train_x, train_y.astype(int), test_x, test_y.astype(int), val_x, val_y.astype(int)


def load_multiple_reads_data(args):
    test_size = 2500
    total_size = 2000000
    # Global parameters
    file_path_normal = os.path.join(args.data_path, "pcr.tsv")
    file_path_modified = os.path.join(args.data_path, "msssi.tsv")

    non_modified_duplicate = {}
    test_x = []
    # Extract data from non-modified
    with open(file_path_normal) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        data_count = 0
        for index, row in enumerate(read_tsv):
            # Ignore the first x data points used for training, and check data
            if index <= args.data_size / 2 or check_data(row) is False:
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
    test_x = test_x[0:test_size]

    modified_duplicate = {}
    # Extract data from modified
    with open(file_path_modified) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        data_count = 0
        for index, row in enumerate(read_tsv):
            # Ignore the first x data points used for training, and check data
            if index <= args.data_size / 2 or check_data(row) is False:
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
    test_x = test_x[0:2 * test_size]
    test_x = np.asarray(test_x)
    print(test_x.shape)
    test_y = np.append(np.zeros(test_size), np.ones(test_size))
    return test_x, test_y
