import random  # Ensure you are using the standard Python random module
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import *
import numpy as np
import importlib
import json
import os

def get_reg(hp):
    if hp["regularization"] == "l1":
        return l1(l=hp["l1"])
    elif hp["regularization"] == "l2":
        return l2(l=hp["l2"])
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

def cnn_random(classes, number_of_samples, regularization=False,  hp=None):
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
            x = Dropout(get_reg(hp))(x)

    outputs = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, outputs, name='random_cnn')
    optimizer = get_optimizer(hp["optimizer"], hp["learning_rate"])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, tf_random_seed, hp

def get_optimizer(optimizer, learning_rate):
    module_name = importlib.import_module("tensorflow.keras.optimizers")
    optimizer_class = getattr(module_name, optimizer)
    return optimizer_class(lr=learning_rate)

def save_model_and_hyperparameters(model, hyperparameters, model_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_path = os.path.join(output_dir, model_name + ".h5")
    hyperparams_path = os.path.join(output_dir, model_name + "_hyperparameters.json")
    
    model.save(model_path)
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparameters, f)

def generate_multiple_dnns(num_models, model_type, classes, number_of_samples, output_dir, regularization=False):
    for i in range(num_models):
        if model_type == "mlp":
            hp = get_hyperparameters_mlp(regularization=regularization)
            model, seed, hp = mlp_random(classes, number_of_samples, hp=hp, regularization=regularization)
        else:
            hp = get_hyperparemeters_cnn(regularization=regularization)
            model, seed, hp = cnn_random(classes, number_of_samples, hp=hp, regularization=regularization)
        
        model_name = f"{model_type}_model_{i+1}"
        save_model_and_hyperparameters(model, hp, model_name, output_dir)

if __name__ == "__main__":
    num_models = 10  # Number of models to generate
    model_type = "cnn"  # "mlp" or "cnn"
    classes = 256
    number_of_samples = 1000
    output_dir = "./generated_models"
    regularization = True

    generate_multiple_dnns(num_models, model_type, classes, number_of_samples, output_dir, regularization=regularization)
