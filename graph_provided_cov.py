import numpy as np
import matplotlib.pyplot as plt

cov_matrix = np.loadtxt("oscillator_covariance.txt")

plt.imshow(cov_matrix, cmap='coolwarm', interpolation='nearest')
plt.grid(True)
plt.colorbar(label='Covariance')
plt.xlabel('Sample file')
plt.ylabel('Sample file')
plt.title('Covariance Matrix')
plt.show()  