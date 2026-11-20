import numpy as np
from DHO_values import DHO


# Calculate chi-squared for a given set of parameters
def chi_squared(params, time, data, inv_cov):

    A, omega, gamma, phi = params

    model = DHO(A, omega, gamma, phi, time)

    residual = data - model

    chi2 = residual.T @ inv_cov @ residual

    return chi2


# Calculate negative log-likelihood
def negative_log_likelihood(params, time, data, inv_cov):

    chi2 = chi_squared(params, time, data, inv_cov)

    return 0.5 * chi2
