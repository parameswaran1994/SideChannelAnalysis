def generate_traces(num_traces, key_byte):
    plaintexts = np.random.randint(0, 256, size=(num_traces,))
    traces = np.random.normal(0, 1, size=(num_traces, 20))  # Simulated traces
    labels = np.zeros(num_traces, dtype=int)
    
    for i, pt in enumerate(plaintexts):
        sbox_output = Sbox(pt ^ key_byte)
        traces[i, 5] += sbox_output  # Assuming the sixth channel is related to the S-box output
        labels[i] = sbox_output

    return traces, labels
