#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
import random as python_random
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import random
import importlib

# Seed reset function
def reset_random_seeds(seed_value=42):
    python_random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

reset_random_seeds()

# AES S-box and utility functions
AES_Sbox = [
    # S-box array
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

def Sbox(input_byte):
    return AES_Sbox[input_byte]

def HW(s):
    return bin(s).count("1")

AES_Sbox_inv = np.array([
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
])

def get_reg(hp):
    if hp["regularization"] == "l1":
        return l1(l1=hp["l1"])
    elif hp["regularization"] == "l2":
        return l2(l2=hp["l2"])
    else:
        return hp["dropout"]

def get_hyperparameters_mlp(regularization=False, max_dense_layers=3):
    if regularization:
        return {
            "batch_size": random.randrange(100, 400, 100),
            "layers": random.randrange(1, max_dense_layers + 1, 1),
            "neurons": random.choice([100, 200, 300, 400]),
            "activation": random.choice(["relu", "selu"]),
            "learning_rate": random.choice([0.0005, 0.0001, 0.00005, 0.00001]),
            "optimizer": random.choice(["Adam", "RMSprop"]),
            "kernel_initializer": random.choice(["random_uniform", "glorot_uniform", "he_uniform"]),
            "regularization": random.choice(["l1", "l2", "dropout"]),
            "l1": random.choice([0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]),
            "l2": random.choice([0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]),
            "dropout": random.choice([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        }
    else:
        return {
            "batch_size": random.randrange(100, 1100, 100),
            "layers": random.choice([1, 2, 3, 4]),
            "neurons": random.choice([10, 20, 50, 100, 200, 300, 400, 500]),
            "activation": random.choice(["relu", "selu"]),
            "learning_rate": random.choice([0.005, 0.001, 0.0005, 0.0001]),
            "optimizer": random.choice(["Adam", "RMSprop"]),
            "kernel_initializer": random.choice(["random_uniform", "glorot_uniform", "he_uniform"]),
            "regularization": random.choice(["none"])
        }

def get_hyperparemeters_cnn(regularization=False):
    hyperparameters = {}
    hyperparameters_mlp = get_hyperparameters_mlp(regularization=regularization, max_dense_layers=3)
    for key, value in hyperparameters_mlp.items():
        hyperparameters[key] = value

    conv_layers = random.choice([1, 2, 3, 4])
    kernels = []
    strides = []
    filters = []
    pooling_types = []
    pooling_sizes = []
    pooling_strides = []
    pooling_type = random.choice(["Average", "Max"])

    for conv_layer in range(1, conv_layers + 1):
        kernel = random.randrange(2, 10, 1)
        kernels.append(kernel)
        strides.append(int(kernel / 2))
        if conv_layer == 1:
            filters.append(random.choice([u for u in range(8, 65, 8)]))
        else:
            filters.append(filters[conv_layer - 2] * 2)
        pool_size = random.choice([2, 3, 4, 5])
        pooling_sizes.append(pool_size)
        pooling_strides.append(pool_size)
        pooling_types.append(pooling_type)

    hyperparameters["conv_layers"] = conv_layers
    hyperparameters["kernels"] = kernels
    hyperparameters["strides"] = strides
    hyperparameters["filters"] = filters
    hyperparameters["pooling_sizes"] = pooling_sizes
    hyperparameters["pooling_strides"] = pooling_strides
    hyperparameters["pooling_types"] = pooling_types

    return hyperparameters

def mlp_random(classes, number_of_samples, regularization=False, hp=None):
    hp = get_hyperparameters_mlp(regularization=regularization) if hp is None else hp
    tf_random_seed = np.random.randint(1048576)
    tf.random.set_seed(tf_random_seed)

    inputs = Input(shape=number_of_samples)
    x = None
    for layer_index in range(hp["layers"]):
        if regularization and hp["regularization"] != "dropout":
            x = Dense(hp["neurons"], activation=hp["activation"], kernel_regularizer=get_reg(hp),
                      kernel_initializer=hp["kernel_initializer"],
                      name='dense_{}'.format(layer_index))(inputs if layer_index == 0 else x)
        else:
            x = Dense(hp["neurons"], activation=hp["activation"], kernel_initializer=hp["kernel_initializer"],
                      name='dense_{}'.format(layer_index))(inputs if layer_index == 0 else x)
        if regularization and hp["regularization"] == "dropout":
            x = Dropout(get_reg(hp))(x)

    outputs = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(inputs, outputs, name='random_mlp')
    optimizer = get_optimizer(hp["optimizer"], hp["learning_rate"])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, tf_random_seed, hp

def cnn_random(classes, number_of_samples, regularization=False, hp=None):
    hp = get_hyperparemeters_cnn(regularization=regularization) if hp is None else hp
    tf_random_seed = np.random.randint(1048576)
    tf.random.set_seed(tf_random_seed)

    inputs = Input(shape=(number_of_samples, 1))
    x = None
    for layer_index in range(hp["conv_layers"]):
        x = Conv1D(kernel_size=hp["kernels"][layer_index], strides=hp["strides"][layer_index], filters=hp["filters"][layer_index],
                   activation=hp["activation"], padding="same")(inputs if layer_index == 0 else x)
        if hp["pooling_types"][layer_index] == "Average":
            x = AveragePooling1D(pool_size=hp["pooling_sizes"][layer_index], strides=hp["pooling_strides"][layer_index], padding="same")(x)
        else:
            x = MaxPooling1D(pool_size=hp["pooling_sizes"][layer_index], strides=hp["pooling_strides"][layer_index], padding="same")(x)
        x = BatchNormalization()(x)
    x = Flatten()(x)

    for layer_index in range(hp["layers"]):
        if regularization and hp["regularization"] != "dropout":
            x = Dense(hp["neurons"], activation=hp["activation"], kernel_regularizer=get_reg(hp),
                      kernel_initializer=hp["kernel_initializer"], name='dense_{}'.format(layer_index))(x)
        else:
            x = Dense(hp["neurons"], activation=hp["activation"], kernel_initializer=hp["kernel_initializer"],
                      name='dense_{}'.format(layer_index))(x)
        if regularization and hp["regularization"] == "dropout":
            x = Dropout(hp["dropout"])(x)

    outputs = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(inputs, outputs, name='random_cnn')
    optimizer = get_optimizer(hp["optimizer"], hp["learning_rate"])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, tf_random_seed, hp

def get_optimizer(optimizer, learning_rate):
    module_name = importlib.import_module("tensorflow.keras.optimizers")
    optimizer_class = getattr(module_name, optimizer)
    return optimizer_class(learning_rate=learning_rate)

# Function for Guessing Entropy (GE) and Correlation Power Analysis (CPA)
def perform_attacks(nb_traces=1000, predictions=None, plt_attack=None, correct_key=None, leakage="HW", dataset=None, nb_attacks=100, shuffle=True):
    all_rk_evol = np.zeros((nb_attacks, nb_traces))
    all_key_log_prob = np.zeros(256)
    for i in tqdm(range(nb_attacks)):
        if shuffle:
            l = list(zip(predictions, plt_attack))
            random.shuffle(l)
            sp, splt = list(zip(*l))
            sp = np.array(sp)
            splt = np.array(splt)
            att_pred = sp[:nb_traces]
            att_plt = splt[:nb_traces]
        else:
            att_pred = predictions[:nb_traces]
            att_plt = plt_attack[:nb_traces]
        rank_evol, key_log_prob = rank_compute(att_pred, att_plt, correct_key, leakage=leakage, dataset=dataset)
        all_rk_evol[i] = rank_evol
        all_key_log_prob += key_log_prob
    return np.mean(all_rk_evol, axis=0), key_log_prob

def rank_compute(prediction, att_plt, correct_key, leakage, dataset):
    hw = [bin(x).count("1") for x in range(256)]
    (nb_traces, nb_hyp) = prediction.shape
    key_log_prob = np.zeros(256)
    prediction = np.log(prediction + 1e-40)
    rank_evol = np.full(nb_traces, 255)
    for i in range(nb_traces):
        for k in range(256):
            att_byte = int(att_plt[i, 0])
            if dataset == "AES_HD_ext":
                if leakage == 'ID':
                    key_log_prob[k] += prediction[i, AES_Sbox_inv[k ^ att_byte] ^ att_byte]
                else:
                    key_log_prob[k] += prediction[i, hw[AES_Sbox_inv[k ^ att_byte] ^ att_byte]]
            elif dataset == "AES_HD_ext_ID":
                if leakage == 'ID':
                    key_log_prob[k] += prediction[i, AES_Sbox_inv[k ^ att_byte]]
                else:
                    key_log_prob[k] += prediction[i, hw[AES_Sbox_inv[k ^ att_byte]]]
            else:
                if leakage == 'ID':
                    key_log_prob[k] += prediction[i, AES_Sbox[k ^ att_byte]]
                else:
                    key_log_prob[k] += prediction[i, hw[AES_Sbox[k ^ att_byte]]]
        rank_evol[i] = rk_key(key_log_prob, correct_key)
    return rank_evol, key_log_prob

def rk_key(rank_array, key):
    key_val = rank_array[key]
    final_rank = np.float32(np.where(np.sort(rank_array)[::-1] == key_val)[0][0])
    if math.isnan(float(final_rank)) or math.isinf(float(final_rank)):
        return np.float32(256)
    else:
        return np.float32(final_rank)

def NTGE_fn(GE):
    NTGE = float('inf')
    for i in range(GE.shape[0] - 1, -1, -1):
        if GE[i] > 0:
            NTGE = i
            break
    return NTGE

def aes_label_cpa(plaintexts, correct_key, leakage):
    num_traces = plaintexts.shape[0]
    labels_for_snr = np.zeros(num_traces)
    for i in range(num_traces):
        if leakage == 'HW':
            labels_for_snr[i] = HW(AES_Sbox[plaintexts[i, 0] ^ correct_key])
        elif leakage == 'ID':
            labels_for_snr[i] = AES_Sbox[plaintexts[i, 0] ^ correct_key]
    return labels_for_snr

def perform_cpa_all_keys(traces, plaintexts, num_keys=256):
    num_traces, num_samples = traces.shape
    correlations = np.zeros((num_keys, num_samples))
    for key_guess in tqdm(range(num_keys)):
        labels_for_cpa = aes_label_cpa(plaintexts, key_guess, leakage)
        for t in range(num_samples):
            correlations[key_guess, t] = abs(np.corrcoef(labels_for_cpa[:num_traces], traces[:, t])[1, 0])
    return correlations

def plot_correlations(correlations, correct_key):
    num_keys, num_samples = correlations.shape
    plt.figure(figsize=(10, 8))
    for key_guess in range(num_keys):
        if key_guess != correct_key:
            plt.plot(correlations[key_guess], color='grey', label='Other Keys' if 'Other Keys' not in plt.gca().get_legend_handles_labels()[1] else "", alpha=0.5, linewidth=1)
    plt.plot(correlations[correct_key], color='blue', label=f'Correct Key: {correct_key:02X}', linewidth=2)
    plt.title('Correlation Power Analysis Across All Key Guesses')
    plt.xlabel('Sample Points')
    plt.ylabel('Absolute Correlation')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    # Data loading and preprocessing
    num_traces = 10000
    trace_length = 5000
    leakage = "ID"
    chipwhisper_folder = '/home/localuserplr/Documents/FYP/chipwhisperer/'

    X = np.load(chipwhisper_folder + 'traces.npy')[:num_traces]
    y = np.load(chipwhisper_folder + 'labels.npy')[:num_traces]
    plaintexts = np.load(chipwhisper_folder + 'plain.npy')[:num_traces]
    keys = np.load(chipwhisper_folder + 'key.npy')[:num_traces]

    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_train_val, X_attack, y_train_val, y_attack, plaintexts_train_val, plaintexts_attack = train_test_split(
        X, y, plaintexts, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=256)
    y_val_categorical = tf.keras.utils.to_categorical(y_val, num_classes=256)

    classes = 256
    number_of_samples = trace_length
    model_type = "cnn"
    regularization = True
    
    models = []
    for i in range(10):
        try:
            if model_type == "mlp":
                hp = get_hyperparameters_mlp(regularization=regularization)
                model, seed, hp = mlp_random(classes, number_of_samples, hp=hp, regularization=regularization)
            else:
                hp = get_hyperparemeters_cnn(regularization=regularization)
                model, seed, hp = cnn_random(classes, number_of_samples, hp=hp, regularization=regularization)
            
            models.append((model, seed, hp))
            
            print(f"Model {i+1} summary:")
            model.summary()
            print(f"Hyperparameters for model {i+1}: {hp}")
            print(f"Seed for model {i+1}: {seed}")
            
            history = model.fit(X_train, y_train_categorical, epochs=50, validation_data=(X_val, y_val_categorical), batch_size=hp["batch_size"])
            
            predictions = model.predict(X_attack)
            predicted_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(predicted_classes == y_attack)
            print(f"Attack set accuracy for model {i+1}: {accuracy}")

            GE, key_prob = perform_attacks(nb_traces=1000, predictions=predictions, plt_attack=plaintexts_attack, correct_key=0x2B, leakage=leakage, dataset=None, nb_attacks=100, shuffle=True)
            NTGE = NTGE_fn(GE)
            print(f"GE for model {i+1}: {GE}")
            print(f"NTGE for model {i+1}: {NTGE}")

            plt.plot(GE)
            plt.title(f'Entropy vs Number of Traces for Model {i+1}')
            plt.xlabel('Number of traces')
            plt.ylabel('Guessing Entropy')
            plt.grid(True)
            plt.show()

            correlations = perform_cpa_all_keys(X_attack.squeeze(), plaintexts_attack, num_keys=256)
            plot_correlations(correlations, correct_key=0x2B)
        
        except Exception as e:
            print(f"An error occurred while creating model {i+1}: {e}")

    print(f"Total models created: {len(models)}")


# In[ ]:


from tensorflow.keras.models import load_model

model_path = '/home/localuserplr/Documents/FYP/chipwhisperer/Chipwhisperer_model.h5'
model.save(model_path)

