import numpy as np
from DHO_values import DHO
import matplotlib.pyplot as plt

np.random.seed(123)

# Set the true values of model parameters...
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate the fake data to work with...
A = 0.05
gamma = 0.01
omega = np.sqrt(1000)
omega_prime = np.sqrt(omega**2 - gamma**2)
phi = 0.0
T = 2*np.pi/omega
N = 50 # Number of data points
    
time = np.random.uniform(0, 3*T, N)

x = DHO(time, A, omega, gamma, phi)

yerr = 0.1 + 0.5*np.random.rand(N)
y = m_true*x + b_true
y += np.abs(f_true*y) * np.random.normal(0.0, 1.0, N)
y += yerr*np.random.normal(0.0, 1.0, N)

plt.errorbar(x, y, yerr=yerr, fmt=".", capsize=2, label="Data", color="#6CA2FF")
x0 = np.linspace(0, 10, 1000)
plt.plot(x0, m_true*x0 + b_true, label="Truth", color="#FF6600")
plt.xlim(0,10)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.savefig("data.png", bbox_inches="tight", dpi=300)
plt.show()