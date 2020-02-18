import tensorflow.keras.backend as k
from tensorflow.keras.losses import Loss


class EncodingLoss(Loss):
    def __init__(self):
        super(EncodingLoss, self).__init__()

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
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

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


class ICMILoss(Loss):
    def __init__(self):
        super(ICMILoss, self).__init__()

    def call(self, y_true, y_pred):
        """
        Compute the index term mutual information loss in the total correlation decomposition of the evidence lower
        bound. This is the first of three terms in the ELBO decomposition for a TC-VAE. For details, see "Isolating
        sources of disentanglement" by Chen et al.
        :param y_true:
        :param y_pred:
        :return:
        """
        return y_true


class TCLoss(Loss):
    def __init__(self):
        super(TCLoss, self).__init__()

    def call(self, y_true, y_pred):
        """
        Compute the total correlation loss (dependence loss) in the total correlation decomposition of the evidence
        lower bound. This is the second of three terms in the ELBO decomposition for a TC-VAE. For details, see
        "Isolating sources of disentanglement" by Chen et al.
        :param y_true:
        :param y_pred:
        :return:
        """
        return y_true


class ComponentKLLoss(Loss):
    def __init__(self):
        super(ComponentKLLoss, self).__init__()

    def call(self, y_true, y_pred):
        """
        Compute the coordinate-wise KL divergence  in the total correlation decomposition of the evidence lower
        bound. This is the third of three terms in the ELBO decomposition for a TC-VAE. For details, see "Isolating
        sources of disentanglement" by Chen et al.
        :param y_true:
        :param y_pred:
        :return:
        """
        return y_true


class TCVAELoss(Loss):
    def __init__(self, alpha=1, beta=2, gamma=1):
        super(TCVAELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def call(self, y_true, y_pred):
        """
        Compute the loss for a TC-VAE. For details, see "Isolating  sources of disentanglement" by Chen et al.
        :param y_true:
        :param y_pred:
        :return:
        """
        return self.alpha * ICMILoss()(y_true, y_pred) \
               + self.beta * TCLoss()(y_true, y_pred) \
               + self.gamma * ComponentKLLoss()(y_true, y_pred)
