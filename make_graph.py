import numpy as np
import matplotlib.pyplot as plt
from make_cov_matrix import make_cov_mat

cov_matrix = make_cov_mat()

np.savetxt("DHO_covariance.txt", cov_matrix, fmt="%.8e")

plt.imshow(cov_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Covariance')
plt.xlabel('Sample file')
plt.ylabel('Sample file')
plt.title('Covariance Matrix')
plt.show()  