from tensorflow.keras.datasets import mnist
from time import time
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import os


class Rotator():
    @classmethod
    def view_image(cls, image):
        plt.imshow(image)
        plt.show()

    def __init__(self, images, labels, number_of_rotations, angle_of_rotation, partition='train'):
        """
        A rotator comes equipped with a data set, a directory to which augmented data sets are written, and augmentation
        parameters that determine the size and character of the augmented data set.
        :param images:
        :param directory:
        :param number_of_rotations:
        :param angle_of_rotation:
        """
        directory = os.path.abspath(os.path.join(os.getcwd(), '../data/mnist', partition))
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.directory = directory
        if images is None:
            print("No data specified. Loading MNIST training set.")
            (x, y), (_, _) = mnist.load_data()
            self.images = x
            self.labels = y
            self.data_length = len(x)
        else:
            self.images = images
            self.labels = labels
            self.data_length = len(images)
        self.number_of_rotations = number_of_rotations
        self.angle_of_rotation = angle_of_rotation

    def append_rotated_images(self):
        print(f"Augmenting data set of shape {self.images.shape}.")
        t0 = time()
        if self.number_of_rotations is None:
            self.number_of_rotations = 2
        augmented_labels = np.tile(self.labels, self.number_of_rotations + 1)
        augmentation_list = [self.images]
        for j in range(1, self.number_of_rotations + 1):
            theta = j * self.angle_of_rotation
            print(f"Rotating by angle {theta}")
            new_images = ndimage.rotate(self.images, j * self.angle_of_rotation, axes=(2, 1), reshape=False)
            augmentation_list.append(new_images)
        augmentation_tuple = tuple(augmentation_list)
        augmented_data = np.concatenate(augmentation_tuple)
        t1 = time()
        t = t1 - t0
        print(f"Completed in {t} seconds.")
        return augmented_data, augmented_labels

    def save(self, digit):
        images_filename = f'digits={str(digit)}_n_rot={self.number_of_rotations}_angle={self.angle_of_rotation}.npy'
        labels_filename = f'labels={str(digit)}_n_rot={self.number_of_rotations}_angle={self.angle_of_rotation}.npy'
        images_filepath = os.path.abspath(os.path.join(self.directory, images_filename))
        labels_filepath = os.path.abspath(os.path.join(self.directory, labels_filename))
        augmented_data, augmented_labels = self.append_rotated_images()
        np.save(images_filepath, augmented_data)
        np.save(labels_filepath, augmented_labels)

    def demo(self):
        digit_array = []
        for i in (4, 13, 18, 19):
            digit_array.append(self.images[i])
        digit_array = np.asarray(digit_array)

        rot = Rotator(digit_array)
        data = rot.append_rotated_images()

        for i in range(len(data)):
            self.view_image(data[i])


class MNISTRotator(Rotator):
    def __init__(self, digit, number_of_rotations, angle_of_rotation, partition='train'):
        if partition == 'test':
            print("Using MNIST test data.")
            (_, _), (x, y) = mnist.load_data()
        else:
            print("Using MNIST training data.")
            (x, y), (_, _) = mnist.load_data()
        class_indices = np.where(y == digit)
        data = x[class_indices]
        labels = y[class_indices]
        self.digit = digit
        super(MNISTRotator, self).__init__(data, labels, number_of_rotations, angle_of_rotation, partition=partition)

    def augment(self):
        self.save(self.digit)


classes = list(range(10))
for c in classes:
    rotator = MNISTRotator([c], 11, 30, partition='test')
    rotator.augment()


