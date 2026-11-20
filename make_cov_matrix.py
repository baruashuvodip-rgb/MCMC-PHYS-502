import numpy as np

def make_cov_mat():
    data = []

    for i in range(1000):
        samples = np.loadtxt(f"DHO_sample_{i+1}.txt")
        data.append(samples)

    data = np.array(data)

    cov_matrix = np.cov(data, rowvar = False)
    return cov_matrix