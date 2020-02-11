import numpy as np

n = 10
mean_array = np.zeros(shape=(n, 2))
covariance_array = np.ones(shape=(n, 2))
gaussian_parameters = np.concatenate((mean_array, covariance_array), axis=-1)

print(gaussian_parameters)