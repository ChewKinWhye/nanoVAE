from utils.data import load_dna_data_vae, load_multiple_reads_data
from utils.arguments import parse_args
from utils.model import build_five_mer_model, load_vae_predictor
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

def original_loss(y_true, y_pred):
    lc = 1 / (64 * batchsize) * batchsize ** 2 * K.sum((y_pred - K.mean(y_pred, axis=0)) ** 2, axis=[1]) / ((batchsize - 1) ** 2)
    return lc


def extract_five_mer_data(x_train):
    x_train_five_mer = x_train[:, 68:]
    y_train_five_mer = np.zeros(x_train.shape[0])
    five_mer_label = []
    idx_to_label = ["C", "G", "T", "A"]
    for idx, row in enumerate(x_train):
        one_hot_five_mer = row[24:44]
        label = 0
        nuc_label = ""
        for i in range(5):
            five_mer_int = np.where(one_hot_five_mer[i*4:i*4+4] == 1)[0][0]
            label += five_mer_int * 4**i
            nuc_label += idx_to_label[five_mer_int]
        y_train_five_mer[idx] = label
        five_mer_label.append(nuc_label)
    label_encoder = LabelEncoder()
    y_train_five_mer = label_encoder.fit_transform(y_train_five_mer).reshape(-1, 1)
    enc = OneHotEncoder(sparse=False)
    y_train_five_mer = enc.fit_transform(y_train_five_mer)
    return x_train_five_mer, y_train_five_mer, five_mer_label

if __name__ == "__main__":
    retrain = False
    nanodoc_model_path = os.path.join("trained_models", "nanodoc_models")
    args = parse_args()
    model_path = os.path.join("trained_models", "five_mer_model.h5")
    x_train, y_train, x_test, y_test, x_val, y_val, standardize_scale = load_dna_data_vae(args.data_size, args.data_path, args.feature_scale)
    # Five-mer model training
    x_train_five_mer, y_train_five_mer, five_mer_label = extract_five_mer_data(x_train)
    nucs = ('A','T','C','G')
    models = []
    model_names = []
    if retrain is True:
        for n1, n2, n5 in itertools.product(nucs, nucs, nucs):
            nuc = n1 + n2 + 'CG' + n5
            nuc_idx = []
            for idx, x in enumerate(five_mer_label):
                if x == nuc:
                    nuc_idx.append(idx)
            print(f"Nucleotide: {nuc}")
            kmer_data = x_train_five_mer[nuc_idx, :]
            print(f"Data shape: {kmer_data.shape}")
            if not os.path.isfile(model_path):
                five_mer_model = build_five_mer_model()
                es = EarlyStopping(monitor='val_loss', mode='min', patience=20)
                five_mer_model.fit(x_train_five_mer, y_train_five_mer, epochs=100, batch_size=256, verbose=1, shuffle=True, validation_split=0.2, callbacks=[es])
                five_mer_model.save(model_path)
            else:
                # Load model
                print("Loading Model")
                five_mer_model = keras.models.load_model(model_path)
            
            # DOC training
            # Remove last 3 layers
            five_mer_model.layers.pop()
            five_mer_model.layers.pop()
            five_mer_model.layers.pop()   
            for layer in five_mer_model.layers:
                if layer.name == "stop_freeze":
                    print("Stop freeze")
                    break
                else:
                    layer.trainable = False
            model_t_in = keras.Input(shape=(411,))
            model_t_out = five_mer_model(model_t_in)
            model_t = keras.models.Model(inputs=model_t_in, outputs=model_t_out)
            
            model_r_in = keras.Input(shape=(411,))
            x = five_mer_model(model_r_in)
            model_r_out = keras.layers.Dense(64, activation="softmax")(x)
            model_r = keras.models.Model(inputs=model_r_in, outputs=model_r_out)
            optimizer = keras.optimizers.SGD(lr=5e-5, decay=0.00005)
            model_r.compile(optimizer=optimizer, loss="categorical_crossentropy")
            model_t.compile(optimizer=optimizer, loss=original_loss)
             
            loss, loss_c = [], []
            epochs = []
            batchsize = 128
            epoch_num = 3
            for epochnumber in range(epoch_num):
                lc, ld = [], []
                for i in range(int(len(kmer_data) / batchsize)):
                    batch_x = x_train_five_mer[i*batchsize:(i+1)*batchsize]
                    batch_y = y_train_five_mer[i*batchsize:(i+1)*batchsize]
                    kmer_batch = kmer_data[i*batchsize:(i+1)*batchsize]
                    lc.append(model_t.train_on_batch(kmer_batch, np.zeros((batchsize, 64))))
                    ld.append(model_r.train_on_batch(batch_x, batch_y))
                loss.append(np.mean(ld))
                loss_c.append(np.mean(lc))
                print("epoch : {}, Descriptive loss : {}, Compact loss : {}".format(epochnumber + 1, loss[-1], loss_c[-1]))
            
            models.append(model_t)
            model_names.append(nuc)
            model_t.save(os.path.join(nanodoc_model_path, nuc + ".h5"))
        print("Done training")
    else:
        for f in os.listdir(nanodoc_model_path):
            model_path = os.path.join(nanodoc_model_path, f)
            model = build_five_mer_model()
            model.layers.pop()
            model.layers.pop()
            model.layers.pop()   
            for layer in model.layers:
                if layer.name == "stop_freeze":
                    break
                else:
                    layer.trainable = False
            model_t_in = keras.Input(shape=(411,))
            model_t_out = model(model_t_in)
            model_t = keras.models.Model(inputs=model_t_in, outputs=model_t_out)

            model_t.load_weights(model_path)
            models.append(model_t)
            model_names.append(f[:-3])
    # We now have models and model_names
    non_mod_encoding_map = np.zeros((64, 64))
    counter = np.zeros((64))
    non_mod_encoding = []

    for idx, row in enumerate(x_train_five_mer):
        label = five_mer_label[idx]
        model_idx = model_names.index(label)
        encoding = models[model_idx].predict(np.expand_dims(row, axis=0))
        non_mod_encoding.append(encoding[0])
    non_mod_encoding = np.array(non_mod_encoding)
    for idx, five_mer in enumerate(y_train_five_mer):
        five_mer_idx = np.where(five_mer==1)[0][0]
        non_mod_encoding_map[five_mer_idx] += non_mod_encoding[idx]
        counter[five_mer_idx] += 1
    for i in range(64):
        non_mod_encoding_map[i] = non_mod_encoding_map[i] / counter[i]
    
    x_test_five_mer, y_test_five_mer, test_five_mer_label = extract_five_mer_data(x_test)
    
    x_test_encoding = []
    for idx, row in enumerate(x_test_five_mer):
        label = test_five_mer_label[idx]
        model_idx = model_names.index(label)
        encoding = models[model_idx].predict(np.expand_dims(row, axis=0))
        x_test_encoding.append(encoding[0])
    x_test_encoding = np.array(x_test_encoding)
    
    mod_diff = [[] for i in range(64)]
    non_mod_diff = [[] for i in range(64)]
    for i, label in enumerate(y_test):
        five_mer_idx = np.where(y_test_five_mer[i]==1)[0][0]
        diff = np.mean((non_mod_encoding_map[five_mer_idx] - x_test_encoding[i]) ** 2)
        if label == 1:
            # Modified
            mod_diff[five_mer_idx].append(diff)
        else:
            non_mod_diff[five_mer_idx].append(diff)
    threshold = np.zeros(64)
    for i in range(64):
        threshold[i] = (sum(mod_diff[i])/len(mod_diff[i]) + sum(non_mod_diff[i])/len(non_mod_diff[i])) / 2
    
    print(f"Threshold: {threshold}")
    
    # Test model
    predictions = []
    x_val_five_mer, y_val_five_mer, val_five_mer_label = extract_five_mer_data(x_val)
    x_val_encoding = []
    for idx, row in enumerate(x_val_five_mer):
        label = val_five_mer_label[idx]
        model_idx = model_names.index(label)
        encoding = models[model_idx].predict(np.expand_dims(row, axis=0))
        x_val_encoding.append(encoding[0])
    x_val_encoding = np.array(x_val_encoding)
    x_val_encoding_diff = np.zeros(x_val_encoding.shape)
    for i, five_mer in enumerate(y_val_five_mer):
        five_mer_idx = np.where(five_mer==1)[0][0]
        x_val_encoding_diff[i] = (non_mod_encoding_map[five_mer_idx] - x_val_encoding[i]) ** 2
        diff = np.mean((non_mod_encoding_map[five_mer_idx] - x_val_encoding[i]) ** 2)
        if diff > threshold[five_mer_idx]:
            predictions.append(1)
        else:
            predictions.append(0)
    test_results = compute_metrics_standardized(predictions, y_val)
    print_results(test_results)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(x_val_encoding_diff)
    predictions = kmeans.labels_
    print(accuracy_score(y_val, predictions))
    print(accuracy_score(y_val, 1-predictions))
    '''
    predictor = load_vae_predictor(32)
    x_test_five_mer, _ = extract_five_mer_data(x_test)
    x_test_predictor = model_t.predict(x_test_five_mer[0:10000])
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
    predictor.fit(x_test_predictor, y_test[0:10000], epochs=args.predictor_epochs, validation_split=0.2, batch_size=args.predictor_batch_size, callbacks=[es])
    # Evaluate
    x_val_five_mer, _ = extract_five_mer_data(x_val)
    x_val_predictor = model_t.predict(x_val_five_mer[0:10000])
    predictions = predictor.predict(x_val_predictor)
    test_results = compute_metrics_standardized(predictions, y_val)
    print_results(test_results)
    '''
