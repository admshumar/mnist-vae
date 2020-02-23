import numpy as np
import os
from time import time


class MNISTLoader:
    def __init__(self, data_directory):
        self.directory = os.path.abspath(os.path.join(os.getcwd(), '../data/mnist', data_directory))

    def get_file(self, digit, number_of_rotations, angle_of_rotation, label=False):
        if label:
            file = f'labels=[{digit}]_n_rot={number_of_rotations}_angle={angle_of_rotation}.npy'
        else:
            file = f'digits=[{digit}]_n_rot={number_of_rotations}_angle={angle_of_rotation}.npy'
        file_path = os.path.join(self.directory, file)
        if os.path.exists(file_path):
            return np.load(file_path)
        else:
            raise FileNotFoundError(f'Dataset with parameters {digit}, '
                                    f'{number_of_rotations}, '
                                    f'{angle_of_rotation} not found!')

    def load(self, list_of_digits, number_of_rotations, angle_of_rotation, label=False):
        t0 = time()
        print("Loading data.")
        initial_digit = list_of_digits[0]
        data = self.get_file(initial_digit, number_of_rotations, angle_of_rotation, label=label)
        for digit in list_of_digits:
            new_data = self.get_file(digit, number_of_rotations, angle_of_rotation, label=label)
            data = np.concatenate((data, new_data))
        t1 = time()
        t = t1 - t0
        print(f"Data loaded in {t} seconds.")
        return data
