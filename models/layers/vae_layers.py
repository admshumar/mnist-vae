import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras.layers import *


class Reparametrization(Layer):
    """
    A deterministic layer for performing Kingma and Welling's "reparametrization trick".
    """
    @classmethod
    def reparametrize(cls, mean, logarithmic_covariance):
        """
        The given mean and logarithmic covariance are regarded as parameters for an isotropic Gaussian. Sampling from
        this Gaussian is identical to first sampling from a Gaussian with zero mean and identity covariance and
        then scaling and translating the samples. This is the "reparametrization trick" of Kingma and Welling, found in
        their paper "Autoencoding Variational Bayes". Let N be the dimension of a Euclidean space.
        :param mean: A Keras tensor of shape (?, N) consisting of means of Gaussian distributions.
        :param logarithmic_covariance: A Keras tensor of shape (?, N) consisting of logarithmic covariances of
            Gaussian distributions.
        :return: A Keras tensor of shape (?, N) representing a set of samples drawn from a reparametrized
            isotropic Gaussian.
        """
        shape = tf.shape(mean)
        standard_deviation = k.exp(0.5 * logarithmic_covariance)
        epsilon = k.random_normal(shape)
        reparametrization = mean + epsilon * standard_deviation
        return reparametrization

    def __init__(self, **kwargs):
        super(Reparametrization, self).__init__(**kwargs)

    def call(self, gaussian_parameters, **kwargs):
        """
        Take in Gaussian parameters, separate their means and covariances, and sample from these Gaussians using Kingma
        and Welling's reparametrization trick.
        :param gaussian_parameters: A Keras tensor of shape (?, 2N) consisting of parameters of Gaussian distributions.
            Note that the covariances are expressed using vectors instead of symmetric matrices, and that these are
            logarithmic covariances instead of ordinary covariances.
        :param kwargs: Keyword arguments.
        :return: A Keras tensor of shape (?, N) consisting of samples from Gaussians.
        """
        dimension = gaussian_parameters.shape[1] // 2
        mean = gaussian_parameters[:, 0:dimension]
        logarithmic_covariance = gaussian_parameters[:, dimension:]
        return Reparametrization.reparametrize(mean, logarithmic_covariance)
