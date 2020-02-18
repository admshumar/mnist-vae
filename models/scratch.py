from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
restriction_indices = np.where(np.isin(y_train, [6, 9]))
print(x_train[restriction_indices])