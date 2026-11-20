import numpy as np
import matplotlib.pyplot as plt
from make_cov_matrix import make_cov_mat

cov_matrix = make_cov_mat()

np.savetxt("DHO_covariance.txt", cov_matrix, fmt="%.8e")

plt.imshow(cov_matrix, cmap='coolwarm', interpolation='nearest')
plt.grid(True)
plt.colorbar(label='Covariance')
plt.xlabel('Sample file')
plt.ylabel('Sample file')
plt.title('Sample Covariance (covariance matrix from 1000 samples)')
plt.savefig("Sample covariance (covariance matrix from 1000 samples).pdf")
plt.show()

cov_provided = np.loadtxt("oscillator_covariance.txt")

plt.imshow(cov_provided, cmap='coolwarm', interpolation='nearest')
plt.grid(True)
plt.colorbar(label='Covariance')
plt.xlabel('Sample file')
plt.ylabel('Sample file')
plt.title('Provided covariance matrix')
plt.savefig("Plot of provided covariance matrix.pdf")
plt.show()