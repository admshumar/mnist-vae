from tensorflow.keras.datasets import mnist
from time import time, gmtime
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import os


class Rotator():
    @classmethod
    def view_image(cls, image):
        plt.imshow(image)
        plt.show()

    def __init__(self, data, number_of_rotations=2, angle_of_rotation=30):
        """
        A rotator comes equipped with a data set, a directory to which augmented data sets are written, and augmentation
        parameters that determine the size and character of the augmented data set.
        :param data:
        :param directory:
        :param number_of_rotations:
        :param angle_of_rotation:
        """
        self.directory = os.path.join(os.getcwd(), '../data/mnist')
        if data is None:
            print("No data specified. Loading MNIST.")
            (x_train, _), (_, _) = mnist.load_data()
            self.data = x_train
            self.data_length = len(x_train)
        else:
            self.data = data
            self.data_length = len(data)
        self.number_of_rotations = number_of_rotations
        self.angle_of_rotation = angle_of_rotation

    def append_rotated_images(self):
        print(f"Augmenting data set of shape {self.data.shape}.")
        t0 = time()
        if self.number_of_rotations is None:
            self.number_of_rotations = 2
        augmented_data = self.data
        for k in range(self.data_length):
            print(f"Augmenting data point {k}")
            for j in range(1, self.number_of_rotations + 1):
                image = self.data[k]
                image = ndimage.rotate(image, j * self.angle_of_rotation, reshape=False)
                image = np.reshape(image, (1, 28, 28))
                augmented_data = np.append(augmented_data, image, axis=0)
        np.random.shuffle(augmented_data)
        t1 = time()
        t = t1 - t0
        print(f"Completed in {t} seconds.")
        return augmented_data

    def save(self, list_of_digits):
        filename = f'digits={str(list_of_digits)}_n_rot={self.number_of_rotations}_angle={self.angle_of_rotation}.npy'
        filepath = os.path.abspath(os.path.join(self.directory, filename))
        augmented_data = self.append_rotated_images()
        np.save(filepath, augmented_data)

    def demo(self):
        digit_array = []
        for i in (4, 13, 18, 19):
            digit_array.append(self.data[i])
        digit_array = np.asarray(digit_array)

        rot = Rotator(digit_array)
        data = rot.append_rotated_images()

        for i in range(len(data)):
            self.view_image(data[i])

class MNISTRotator(Rotator):
    def __init__(self, list_of_digits, number_of_rotations=2, angle_of_rotation=30):
        (x_train, y_train), (_, _) = mnist.load_data()
        class_indices = np.where(np.isin(y_train, list_of_digits))
        data = x_train[class_indices]
        self.list_of_digits = list_of_digits
        super(MNISTRotator, self).__init__(data,
                                           number_of_rotations=number_of_rotations,
                                           angle_of_rotation=angle_of_rotation)
    def augment(self):
        self.save(self.list_of_digits)

rotator = MNISTRotator([5,6], number_of_rotations=11)
rotator.augment()