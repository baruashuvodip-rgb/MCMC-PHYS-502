import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

from DHO_values import DHO


# Log-prior function
def log_prior(params):

    A, omega, gamma, phi = params

    if A > 0 and omega > gamma > 0 and -np.pi < phi < np.pi:
        return 0.0

    return -np.inf


# Log-likelihood function
def log_likelihood(params, time, data, inv_C):

    A, omega, gamma, phi = params

    model = DHO(A, omega, gamma, phi, time)

    residual = data - model

    chi2 = residual.T @ inv_C @ residual

    return -0.5 * chi2


# Log-posterior function
def log_probability(params, time, data, inv_C):

    lp = log_prior(params)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(params, time, data, inv_C)

from scipy.optimize import minimize

# Load data
time = np.loadtxt("DHO_time.txt")
data = np.loadtxt("DHO_sample_1.txt")

# Load covariance matrix
C = np.loadtxt("oscillator_covariance.txt")

# Inverse covariance matrix
inv_C = np.linalg.inv(C)

# Initial guesses
initial_guess = [0.05, np.sqrt(1000), 0.01, 0.0]

# Minimize negative log-likelihood
def neg_log_likelihood(params):

    lp = log_prior(params)

    if not np.isfinite(lp):
        return 1e100

    return -log_likelihood(params, time, data, inv_C)


result = minimize(
    neg_log_likelihood,
    initial_guess,
    method="Nelder-Mead"
)

mle_params = result.x

print("Maximum likelihood parameters:")
print("A =", mle_params[0])
print("omega =", mle_params[1])
print("gamma =", mle_params[2])
print("phi =", mle_params[3])

ndim = 4
nwalkers = 32

# Small random displacements around MLE values
scales = np.array([
    0.01 * abs(mle_params[0]),
    0.01 * abs(mle_params[1]),
    0.01 * abs(mle_params[2]),
    0.01
])

pos = mle_params + scales * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_probability,
    args=(time, data, inv_C)
)

# Run MCMC for 10000 steps
sampler.run_mcmc(pos, 10000, progress=True)

print("MCMC sampling completed.")

chain = sampler.get_chain()

np.save("DHO_MCMC_chain.npy", chain)

chain = sampler.get_chain()

parameter_names = ["A", "omega", "gamma", "phi"]

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

for i in range(ndim):

    axes[i].plot(chain[:, :, i], alpha=0.3)

    axes[i].set_ylabel(parameter_names[i])

axes[-1].set_xlabel("MCMC step")

plt.suptitle("MCMC Chains for DHO Parameters")

plt.tight_layout()

plt.savefig("DHO_MCMC_chains.png", dpi=300)

plt.show()
# Discard first 2000 steps
# Thin the chain by taking every 10th step
flat_samples = sampler.get_chain(
    discard=2000,
    thin=10,
    flat=True
)

# Create corner plot
fig = corner.corner(
    flat_samples,
    labels=parameter_names,
    truths=mle_params,
    show_titles=True,
    title_fmt=".5f"
)

plt.savefig("DHO_corner_plot.png", dpi=300)

plt.show()