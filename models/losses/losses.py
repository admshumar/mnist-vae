import tensorflow.keras.backend as k
from tensorflow.keras.losses import Loss


class EncodingLoss(Loss):
    def call(self, y_true, y_pred):
        """
        Compute the Kullback-Leibler divergence between an arbitrary multivariate Gaussian and the Gaussian with zero
        mean and identity covariance. (In the future, this should be extended to KL for two arbitrary Gaussians.)
        :param y_true: A dummy Keras tensor.
        :param y_pred: A Keras tensor consisting of the parameters of an arbitrary multivariate Gaussian, which is
            parametrized by the encoder model and thus represents the approximate posterior of the variational
            autoencoder's generative model.
        :return: A float indicating the encoding loss.
        """
        mean = y_pred[:, 0:2]
        logarithmic_covariance = y_pred[:, 2:]
        kullback_leibler_divergence_vector = \
            -1 - logarithmic_covariance \
            + k.square(mean) + k.exp(logarithmic_covariance)
        kullback_leibler_divergence = k.sum(kullback_leibler_divergence_vector, axis=-1)
        kullback_leibler_divergence = 0.5 * kullback_leibler_divergence
        return kullback_leibler_divergence


class ReconstructionLoss(Loss):
    def call(self, y_true, y_pred):
        """
        Compute the mean squared error approximation to the reconstruction loss.
        :param y_true: A Keras tensor.
        :param y_pred: A Keras tensor of data synthesized by the approximate posterior.
        :return: A float indicating the approximate reconstruction loss.
        """
        loss = y_true - y_pred
        loss = k.flatten(loss)
        loss = k.square(loss)
        loss = k.mean(loss)
        return loss
