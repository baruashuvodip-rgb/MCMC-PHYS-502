import numpy as np
A = 0.05
gamma = 0.01
omega = np.sqrt(1000)
T = 2*np.pi/omega
#cov_matrix = np.loadtxt("oscillator_covariance.txt")

#L = np.linalg.cholesky(cov_matrix)
#print("L:\n", L)

def DHO(t):
    omega_prime = np.sqrt(omega**2 - gamma**2)
    phi = 0

    return A*np.exp(-gamma*t)*np.cos(omega_prime*t + phi)

for i in range(1000):
    
    time = np.random.uniform(0, 3*T, 50)

    x = DHO(time)
    with open(f"DHO_sample_{i+1}.txt","w") as file:
        for x in x:
            file.write(f"{x}\n")    
