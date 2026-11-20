import numpy as np

def DHO(A, omega, gamma, phi, t):
    omega_prime = np.sqrt(omega**2 - gamma**2)
    return A*np.exp(-gamma*t)*np.cos(omega_prime*t + phi)