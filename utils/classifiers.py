import numpy as np
from sklearn import svm
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


def sv_classify_on_latent_space(data, labels, kernel='linear'):
    support_vector_classifier = svm.SVC(kernel=kernel)
    support_vector_classifier.fit(data, labels)
    return support_vector_classifier
