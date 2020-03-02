from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow

import tensorflow.keras.backend as k
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from models.losses.losses import EncodingLoss
from models.vae import VAE
from models.layers.vae_layers import Reparametrization

from utils import plots, logs


class MNISTCNNClassifier(VAE):
    """
    MNISTClassifier inherits from class VAE, but MNISTClassifier is not a variational autoencoder. It is a vanilla
    classifier on MNIST digits.
    """
    def __init__(self,
                 deep=True,
                 enable_activation=True,
                 enable_augmentation=False,
                 enable_batch_normalization=True,
                 enable_dropout=True,
                 enable_early_stopping=False,
                 early_stopping_patience=10,
                 enable_lr_reduction=False,
                 lr_reduction_patience=10,
                 enable_logging=True,
                 enable_label_smoothing=False,
                 enable_rotations=False,
                 enable_stochastic_gradient_descent=False,
                 has_custom_layers=True,
                 has_validation_set=False,
                 validation_size=0.5,
                 is_mnist=True,
                 is_restricted=False,
                 is_standardized=False,
                 show=False,
                 with_mixture_model=False,
                 with_logistic_regression=False,
                 with_svc=False,
                 number_of_clusters=3,
                 restriction_labels=list(range(10)),
                 intermediate_dimension=512,
                 latent_dimension=128,
                 exponent_of_latent_space_dimension=1,
                 augmentation_size=100,
                 covariance_coefficient=0.2,
                 number_of_epochs=5,
                 batch_size=128,
                 learning_rate_initial=1e-5,
                 learning_rate_minimum=1e-6,
                 dropout_rate=0.5,
                 l2_constant=1e-4,
                 early_stopping_delta=1,
                 beta=1,
                 smoothing_alpha=0.5,
                 number_of_rotations=11,
                 angle_of_rotation=30,
                 encoder_activation='relu',
                 decoder_activation='relu',
                 encoder_activation_layer=ReLU(),
                 decoder_activation_layer=ReLU(),
                 final_activation='sigmoid'
                 ):
        model_name = 'mnist_cnn_classifier'
        super(MNISTCNNClassifier, self).__init__(deep=deep,
                                                 enable_activation=enable_activation,
                                                 enable_augmentation=enable_augmentation,
                                                 enable_batch_normalization=enable_batch_normalization,
                                                 enable_dropout=enable_dropout,
                                                 enable_early_stopping=enable_early_stopping,
                                                 early_stopping_patience=early_stopping_patience,
                                                 enable_lr_reduction=enable_lr_reduction,
                                                 lr_reduction_patience=lr_reduction_patience,
                                                 enable_logging=enable_logging,
                                                 enable_label_smoothing=enable_label_smoothing,
                                                 enable_rotations=enable_rotations,
                                                 enable_stochastic_gradient_descent=enable_stochastic_gradient_descent,
                                                 has_custom_layers=has_custom_layers,
                                                 has_validation_set=has_validation_set,
                                                 validation_size=validation_size,
                                                 is_mnist=is_mnist,
                                                 is_restricted=is_restricted,
                                                 is_standardized=is_standardized,
                                                 show=show,
                                                 with_mixture_model=with_mixture_model,
                                                 with_logistic_regression=with_logistic_regression,
                                                 with_svc=with_svc,
                                                 number_of_clusters=number_of_clusters,
                                                 restriction_labels=restriction_labels,
                                                 intermediate_dimension=intermediate_dimension,
                                                 latent_dimension=latent_dimension,
                                                 exponent_of_latent_space_dimension=exponent_of_latent_space_dimension,
                                                 augmentation_size=augmentation_size,
                                                 covariance_coefficient=covariance_coefficient,
                                                 number_of_epochs=number_of_epochs,
                                                 batch_size=batch_size,
                                                 learning_rate_initial=learning_rate_initial,
                                                 learning_rate_minimum=learning_rate_minimum,
                                                 dropout_rate=dropout_rate,
                                                 l2_constant=l2_constant,
                                                 early_stopping_delta=early_stopping_delta,
                                                 beta=beta,
                                                 smoothing_alpha=smoothing_alpha,
                                                 number_of_rotations=number_of_rotations,
                                                 angle_of_rotation=angle_of_rotation,
                                                 encoder_activation=encoder_activation,
                                                 decoder_activation=decoder_activation,
                                                 encoder_activation_layer=encoder_activation_layer,
                                                 decoder_activation_layer=decoder_activation_layer,
                                                 final_activation=final_activation,
                                                 model_name=model_name)

    def define_classifier(self):
        z = self.encoder_mnist_input
        z = Reshape((28, 28, 1))(z)

        z = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(z)
        z = BatchNormalization()(z)
        z = Dropout(rate=self.dropout_rate, seed=17)(z)

        z = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(z)
        z = BatchNormalization()(z)
        z = Dropout(rate=self.dropout_rate, seed=17)(z)

        z = Flatten()(z)

        z = Dense(self.intermediate_dimension, activation='relu')(z)
        if self.enable_batch_normalization:
            z = BatchNormalization()(z)
        if self.enable_dropout:
            z = Dropout(rate=self.dropout_rate, seed=17)(z)

        z = Dense(self.intermediate_dimension // 2, activation='relu')(z)
        if self.enable_batch_normalization:
            z = BatchNormalization()(z)
        if self.enable_dropout:
            z = Dropout(rate=self.dropout_rate, seed=17)(z)

        z = Dense(self.latent_dimension, activation='relu')(z)

        logits_layer = Dense(self.number_of_clusters)(z)

        probability_layer = Softmax()(logits_layer)

        classifier = Model(self.encoder_mnist_input, probability_layer, name='mnist_cnn_classifier')
        classifier.summary()
        plot_model(classifier, to_file=os.path.join(self.image_directory, 'classifier.png'), show_shapes=True)
        classifier.compile(optimizers.Adam(lr=self.learning_rate),
                           loss=CategoricalCrossentropy(),
                           metrics=['accuracy'])

        return classifier

    def get_fit_args(self, x_train_latent=None, x_test_latent=None):
        """
        Define a list of NumPy inputs and NumPy outputs of the Keras model. These are the actual data that flow through
        the Keras model.
        :return: A list of arguments for the fit method of the Keras model.
        """
        model_input = self.x_train
        if x_train_latent and x_test_latent:
            model_target = self.assign_soft_labels(x_train_latent, x_test_latent)
        else:
            model_target = self.y_train_binary
        return [model_input, model_target]

    def get_fit_kwargs(self):
        """
        Construct keyword arguments for fitting the Keras model. This is useful for conditioning the model's training
        on the presence of a validation set.
        :return: A dictionary of keyword arguments for the fit method of the Keras model.
        """
        fit_kwargs = dict()
        fit_kwargs['epochs'] = self.number_of_epochs
        fit_kwargs['batch_size'] = self.batch_size
        if self.has_validation_set and self.enable_early_stopping:
            fit_kwargs['callbacks'] = [self.early_stopping_callback,
                                       self.nan_termination_callback]
        else:
            fit_kwargs['callbacks'] = [self.nan_termination_callback]
        if self.has_validation_set:
            fit_kwargs['validation_data'] = (self.x_val, self.y_val_binary)
        return fit_kwargs

    def fit_classifier(self, weight_directory=None, alpha=0):
        args = self.get_fit_args()
        kwargs = self.get_fit_kwargs()
        classifier = self.define_classifier()
        print("Training classifier.")
        history = classifier.fit(*args, **kwargs)
        print("Classifier trained.\n")
        return classifier, history

    def train_classifier(self, alpha=0, evaluate=True, report=True):
        if self.enable_logging:
            logs.begin_logging(self.experiment_directory)
        classifier, history = self.fit_classifier(alpha=alpha)
        self.print_settings()
        plots.loss(history, self.image_directory)
        plots.accuracy(history, self.image_directory)
        self.save_model_weights(classifier, 'classifier')
        if evaluate:
            print('Evaluation')
            result = classifier.evaluate(self.x_test, self.y_test_binary,
                                         batch_size=self.batch_size, )
            print(result)
        if evaluate and report:
            filepath = os.path.abspath(os.path.join(self.experiment_directory, 'report.txt'))
            report = open(filepath, "w+")
            report.write(f"Loss: {result[0]}\nAccuracy: {result[1]}")
            report.close()
        return classifier
