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

    def __init__(self,
                 images,
                 labels,
                 number_of_rotations=2,
                 angle_of_rotation=30,
                 partition='train',
                 angle_threshold=180):
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
        self.angle_of_rotation = min(angle_of_rotation, 180)
        self.angle_threshold = angle_threshold
        self.angle_set = {theta for theta in range(0, (number_of_rotations + 1) * angle_of_rotation, angle_of_rotation)
                          if theta <= angle_threshold or theta >= 360 - angle_threshold}

    def rotate(self):
        augmentation_tuple = tuple(ndimage.rotate(self.images, theta, axes=(2, 1), reshape=False)
                                   for theta in self.angle_set)
        return np.concatenate(augmentation_tuple)

    def append_rotated_images(self):
        print(f"Augmenting data set of shape {self.images.shape}.")
        t0 = time()
        augmented_labels = np.tile(self.labels, len(self.angle_set) + 1)
        augmented_data = self.rotate()
        t1 = time()
        t = t1 - t0
        print(f"Completed in {t} seconds.")
        return augmented_data, augmented_labels

    def save(self, list_of_digits):
        images_filename = f'digits={str(list_of_digits)}_n_rot={self.number_of_rotations}_angle={self.angle_of_rotation}'
        labels_filename = f'labels={str(list_of_digits)}_n_rot={self.number_of_rotations}_angle={self.angle_of_rotation}'
        if self.angle_threshold < 180:
            images_filename += f'_threshold={self.angle_threshold}'
            labels_filename += f'_threshold={self.angle_threshold}'
        images_filename += '.npy'
        labels_filename += '.npy'

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
    def __init__(self, list_of_digits, number_of_rotations, angle_of_rotation, partition='train', angle_threshold=None):
        if partition == 'test':
            print("Using MNIST test data.")
            (_, _), (x, y) = mnist.load_data()
        else:
            print("Using MNIST training data.")
            (x, y), (_, _) = mnist.load_data()
        class_indices = np.where(np.isin(y, list_of_digits))
        data = x[class_indices]
        labels = y[class_indices]
        self.list_of_digits = list_of_digits
        super(MNISTRotator, self).__init__(data, labels,
                                           number_of_rotations=number_of_rotations,
                                           angle_of_rotation=angle_of_rotation,
                                           partition=partition,
                                           angle_threshold=angle_threshold)

    def augment(self):
        self.save(self.list_of_digits)

for i in range(10):
    rot = MNISTRotator([i], 11, 30, angle_threshold=90)
    rot.augment()
