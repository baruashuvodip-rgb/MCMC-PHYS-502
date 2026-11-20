#%config InlineBackend.figure_format = "retina"

import numpy as np
import matplotlib.pyplot as plt
C = np.loadtxt("oscillator_covariance.txt")
np.random.seed(123)

# Set the true values of model parameters...
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate the fake data to work with...
N = 50 # Number of data points
x = np.sort(10 * np.random.rand(N)) # x-values between 0 and 10
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