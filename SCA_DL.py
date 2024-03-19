import numpy as np
import tensorflow as tf
import random as python_random
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------------------------------------------------------------
# Set seed for reproducibility
def reset_random_seeds(seed_value=42):
    python_random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

reset_random_seeds()

# -------------------------------------------------------------------------------------------------------------------------------

# Define the AES S-box
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

# -------------------------------------------------------------------------------------------------------------------------------
# Profiling Acquisition Phase
# num_traces is the number of synthetic power traces to generate.
# trace_length is the number of samples in each trace.
# noise_std is the standard deviation of the Gaussian noise added to the traces, simulating the measurement noise in real trace acquisition.
# plaintexts and keys are arrays of random values representing the inputs and keys used in AES operations.
# sbox_outputs represent the intermediate values obtained after applying the S-box operation, simulating an important step in the AES algorithm.
    
# Function to generate synthetic traces
def generate_traces(num_traces, trace_length, noise_std=0.01):
    plaintexts = np.random.randint(0, 256, size=num_traces, dtype=np.uint8)
    keys = np.random.randint(0, 256, size=num_traces, dtype=np.uint8)
    sbox_outputs = np.array([Sbox(pt ^ k) for pt, k in zip(plaintexts, keys)])
    traces = np.random.normal(0, noise_std, size=(num_traces, trace_length))
    # Injecting the relevant signal at a random index
    relevant_index = np.random.randint(0, trace_length)
    traces[np.arange(num_traces), relevant_index] = sbox_outputs
    return traces, sbox_outputs, plaintexts, keys
# -------------------------------------------------------------------------------------------------------------------------------
# Profiling Phase
# Synthetic traces and corresponding S-box outputs (X and y) are generated and split into training and validation datasets.
# A neural network model is defined, compiled, and trained using the training dataset.
# The history object contains information about the training process, which can be analyzed to understand the model's learning progression.

# Generate training and validation traces
num_training = 10000
trace_length = 20
X, y, _, _ = generate_traces(num_training, trace_length)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# -------------------------------------------------------------------------------------------------------------------------------
# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(trace_length,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(256, activation='softmax') # 8-bit S-box
])
# -------------------------------------------------------------------------------------------------------------------------------
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# -------------------------------------------------------------------------------------------------------------------------------
# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))


# -------------------------------------------------------------------------------------------------------------------------------
# Attack Acquistion Phase
# Here, new synthetic traces (X_attack, y_attack) are generated, mimicking a real-world scenario where the attacker would collect new measurements from the target device.

# Generate attack traces
num_attack = 1000
X_attack, y_attack, plaintexts_attack, keys_attack = generate_traces(num_attack, trace_length)

# -------------------------------------------------------------------------------------------------------------------------------

# Predictions Phase
# The model predicts the intermediate values based on the attack traces. These predictions are compared to the actual values to evaluate the model's performance.
# Predict on the attack traces
predictions = model.predict(X_attack)

# Evaluate model's performance
predicted_classes = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_classes == y_attack)
print("Accuracy on attack set:", accuracy)



