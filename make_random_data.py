import numpy as np
from DHO_values import DHO

np.random.seed(123)

A = 0.05
gamma = 0.01
omega = np.sqrt(1000)
omega_prime = np.sqrt(omega**2 - gamma**2)
phi = 0.0
T = 2*np.pi/omega

time = np.random.uniform(0, 3*T, 50)
np.savetxt("DHO_time.txt", time)

mu = DHO(A, omega, gamma, phi, time)

C = np.loadtxt("oscillator_covariance.txt")
L = np.linalg.cholesky(C)

N_samples = 1000
N_data = 50

for i in range(N_samples):
    n = np.random.normal(0.0, 1.0, N_data)
    x = mu + np.dot(L, n)    
    with open(f"DHO_sample_{i+1}.txt","w") as file:
        for x in x:
            file.write(f"{x}\n")    
