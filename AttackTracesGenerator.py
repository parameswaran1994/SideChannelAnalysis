import numpy as np

# Create a dummy dataset of 100 traces, each with 1000 samples (adjust as needed)
dummy_data = np.random.rand(100, 1000)  # Or whatever shape fits your scenario
np.save('attack_traces.npy', dummy_data)
