import numpy as np
from scipy.optimize import minimize
from DHO_MLE import negative_log_likelihood

# Load covariance matrix
C = np.loadtxt("oscillator_covariance.txt")

# Calculate inverse covariance matrix
inv_C = np.linalg.inv(C)

# Load time values
time = np.loadtxt("DHO_time.txt")

# Initial parameter guesses
initial_guess = [0.04, 30.0, 0.02, 0.1]

# Store best-fit parameters
results = []

# Fit each independent simulation run
for i in range(1000):

    data = np.loadtxt(f"DHO_sample_{i+1}.txt")

    result = minimize(negative_log_likelihood, initial_guess, args=(time, data, inv_C), method='Nelder-Mead')

    results.append(result.x)

    if i == 0:
        print("First fit:")
        print("A =", result.x[0])
        print("omega =", result.x[1])
        print("gamma =", result.x[2])
        print("phi =", result.x[3])

results = np.array(results)

# Average fitted parameters across all runs
mean_params = np.mean(results, axis=0)
std_params = np.std(results, axis=0)

print("\nAverage fitted parameters:")
print("A =", mean_params[0], "+/-", std_params[0])
print("omega =", mean_params[1], "+/-", std_params[1])
print("gamma =", mean_params[2], "+/-", std_params[2])
print("phi =", mean_params[3], "+/-", std_params[3])
