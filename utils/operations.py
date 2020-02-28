import numpy as np


def standardize(target_matrix, reference_matrix):
    """
    Standardize the columns of a NumPy array.
    :param target_matrix: A NumPy array to be standardized.
    :param reference_matrix: A NumPy array whose parameters are applied in the standardization.
    :return: A standardized NumPy array.
    """
    mean_matrix = np.mean(reference_matrix, axis=0)
    standard_deviation_matrix = np.std(reference_matrix, axis=0)
    standardized_matrix = (target_matrix - mean_matrix) / standard_deviation_matrix
    return standardized_matrix


def get_gaussian_parameters(data):
    """
    Grab the TensorFlow Keras incarnation of the MNIST data set.
    :return: A NumPy array of MNIST training and test sets.
    """
    mean_array = np.zeros(shape=(len(data), 2))
    covariance_array = np.ones(shape=(len(data), 2))
    gaussian_parameters = np.concatenate((mean_array, covariance_array), axis=-1)
    return gaussian_parameters


def split_gaussian_parameters(gaussian_parameters):
    """
    Grab the TensorFlow Keras incarnation of the MNIST data set.
    :return: A NumPy array of MNIST training and test sets.
    """
    dimension = gaussian_parameters.shape[1] // 2
    mean, logarithmic_covariance = gaussian_parameters[:, 0:dimension], gaussian_parameters[:, dimension:]
    return mean, logarithmic_covariance


def normalize(data):
    data = data / 255.
    return data


def reshape_for_convolution(x_train, x_test):
    """
    Keras's convolutional layers want as inputs 4D tensors with shape: (samples, channels, rows, cols),
    if data_format='channels_first' or 4D tensors with shape: (samples, rows, cols, channels) if
    data_format='channels_last'. The default is data_format='channels_last'.
    :param x_train:
    :param x_test:
    :param dimension:
    :return:
    """
    x_train = np.reshape(x_train, (-1, x_train.shape[1], x_train.shape[2], 1))
    x_test = np.reshape(x_test, (-1, x_test.shape[1], x_test.shape[1], 1))
    return x_train, x_test


def reshape_for_dense_map(x_train, x_test, dimension):
    """

    :param x_train:
    :param x_test:
    :param dimension:
    :return:
    """
    x_train = np.reshape(x_train, (-1, 1, dimension * dimension))
    x_test = np.reshape(x_test, (-1, 1, dimension * dimension))
    return x_train, x_test


def aggregate_labels(list_of_labels):
    """
    Construct the class distribution of labels.
    :param list_of_labels: A NumPy array of class labels.
    :return: A dictionary whose keys are classes and whose values are class probabilities.
    """
    class_distribution = {}
    unique_labels = np.unique(list_of_labels)

    for label in unique_labels:
        label_frequency = len(list_of_labels[list_of_labels == label])
        label_proportion = label_frequency / len(list_of_labels)
        class_distribution[label] = label_proportion

    return class_distribution


def get_power_sequence(data_dimension, exponent):
    """
    Given an integer, construct a list of integers starting with the given integer, then all positive
    powers of two that are less than the given integer.
    :param data_dimension: An integer that represents the dimension of an input data set.
    :param exponent: The exponent of the dimension of the latent space representation (which is expressed as a
        power of two.)
    :return: A list of integers, which are the dimensions of the feature space representations in the autoencoder.
    """
    k = len(bin(data_dimension)) - 3
    sequence = [2 ** i for i in range(exponent, k + 1)]
    if sequence[-1] == data_dimension:
        sequence = sequence[:-1]
    sequence.append(data_dimension)
    return sequence[::-1]


def restrict_data_by_label(data, labels, restriction_array):
    condition = np.isin(labels, restriction_array)
    restriction_indices = np.where(condition)
    return data[restriction_indices], labels[restriction_indices]
