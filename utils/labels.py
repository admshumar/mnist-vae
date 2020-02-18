import numpy as np


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

    def smooth(self):
        true_distribution = (1-self.alpha) * self.encode()
        uniform_distribution = (self.alpha / self.number_of_classes) * self.distribution_array
        return np.add(true_distribution, uniform_distribution)
