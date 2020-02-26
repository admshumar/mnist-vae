import numpy as np
from scipy.stats import multivariate_normal


class OneHotEncoder:
    """
    Takes in a list of labels and returns a NumPy array of one-hot encodings.
    """
    def __init__(self, labels):
        classes = np.unique(labels)
        self.labels = labels
        self.classes = classes
        self.number_of_samples = len(labels)
        self.number_of_classes = len(classes)
        self.distribution_array = np.ones(shape=(len(labels), len(classes)))

    def encode(self):
        one_hot_encoding = 0 * self.distribution_array
        for sample_index in range(self.number_of_samples):
            label = self.labels[sample_index]
            class_index = np.where(self.classes == label)
            one_hot_encoding[sample_index, class_index] += 1
        return one_hot_encoding


class Smoother(OneHotEncoder):
    """
    Perform label smoothing when given a list of labels for a data set.
    """
    def __init__(self, labels, alpha=0.5):
        super(Smoother, self).__init__(labels)
        self.alpha = float(alpha)

    def smooth_uniform(self):
        true_distribution = (1-self.alpha) * self.encode()
        uniform_distribution = (self.alpha / self.number_of_classes) * self.distribution_array
        return np.add(true_distribution, uniform_distribution)


class GaussianSoftLabels:
    """
    Uses a Gaussian mixture model to construct soft labels for data points in Euclidean space.
    """
    def __init__(self, data, means, covariances, labels=None):
        self.data = data
        self.means = means
        self.covariances = covariances
        self.labels = labels

    def get_class_distribution(self, point):
        class_densities = [multivariate_normal.pdf(point,
                                                   self.means[j],
                                                   self.covariances[j]) for j in range(len(self.means))]
        class_distribution = np.asarray(class_densities)
        class_distribution *= 1 / np.sum(class_distribution)
        return np.asarray([class_distribution])

    def get_soft_labels(self):
        class_distribution_tuple = tuple(self.get_class_distribution(self.data[i]) for i in range(len(self.data)))
        soft_labels = np.concatenate(class_distribution_tuple)
        return soft_labels

    def smooth_gaussian_mixture(self, alpha=0.1):
        if self.labels:
            soft_labels = self.get_soft_labels()
            assert self.labels.shape == soft_labels, "Labels have incompatible shapes."
            true_distribution = (1-alpha) * self.labels
            soft_distribution = alpha * soft_labels
            return true_distribution + soft_distribution
        else:
            print('No labels provided!')
            return None
