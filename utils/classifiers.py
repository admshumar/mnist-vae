import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression


def logistically_regress_on_latent_space(data, labels):
    logistic_regressor = LogisticRegression(multi_class='multinomial')
    logistic_regressor.fit(data, labels)
    return logistic_regressor


def fit_mixture_model_on_latent_space(data, labels):
    unique_labels = np.unique(labels)
    number_of_classes = len(unique_labels)
    mixture_model = GaussianMixture(n_components=number_of_classes, n_init=10)
    mixture_model.fit(data, labels)
    return mixture_model
