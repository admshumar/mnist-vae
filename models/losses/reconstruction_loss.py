import tensorflow.keras.backend as k
from tensorflow.keras.losses import Loss


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
