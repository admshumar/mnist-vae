import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras.layers import Layer, Lambda


class Reparametrization(Layer):

    def reparametrize(self, mean, logarithmic_covariance):
        shape = tf.shape(mean)
        standard_deviation = k.exp(0.5 * logarithmic_covariance)
        epsilon = k.random_normal(shape)
        reparametrization = mean + epsilon * standard_deviation
        return reparametrization

    def __init__(self):
        super(Reparametrization, self).__init__()

    def call(self, gaussian_parameters, **kwargs):
        dimension = gaussian_parameters.shape[1] // 2
        mean = gaussian_parameters[:, 0:dimension]
        logarithmic_covariance = gaussian_parameters[:, dimension:]
        return self.reparametrize(mean, logarithmic_covariance)
