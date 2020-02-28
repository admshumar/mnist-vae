from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow

import tensorflow.keras.backend as k
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from models.losses.losses import EncodingLoss
from utils import logs, plots
from models.vae import VAE
from models.layers.vae_layers import Reparametrization


class ConvolutionalVAE(VAE):
    def __init__(self,
                 deep=True,
                 enable_activation=True,
                 enable_augmentation=False,
                 enable_batch_normalization=True,
                 enable_dropout=True,
                 enable_early_stopping=False,
                 enable_logging=True,
                 enable_label_smoothing=False,
                 enable_rotations=False,
                 enable_stochastic_gradient_descent=False,
                 has_custom_layers=True,
                 has_validation_set=False,
                 is_mnist=True,
                 is_restricted=False,
                 is_standardized=False,
                 show=False,
                 number_of_clusters=3,
                 restriction_labels=[1, 2, 3],
                 intermediate_dimension=512,
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
                 number_of_rotations=2,
                 angle_of_rotation=30,
                 encoder_activation='relu',
                 decoder_activation='relu',
                 final_activation='sigmoid',
                 ):
        model_name = 'vae_conv'
        super(ConvolutionalVAE, self).__init__(deep=deep,
                                               enable_activation=enable_activation,
                                               enable_augmentation=enable_augmentation,
                                               enable_batch_normalization=enable_batch_normalization,
                                               enable_dropout=enable_dropout,
                                               enable_early_stopping=enable_early_stopping,
                                               enable_logging=enable_logging,
                                               enable_label_smoothing=enable_label_smoothing,
                                               enable_rotations=enable_rotations,
                                               enable_stochastic_gradient_descent=enable_stochastic_gradient_descent,
                                               has_custom_layers=has_custom_layers,
                                               has_validation_set=has_validation_set,
                                               is_mnist=is_mnist,
                                               is_restricted=is_restricted,
                                               is_standardized=is_standardized,
                                               show=show,
                                               number_of_clusters=number_of_clusters,
                                               restriction_labels=restriction_labels,
                                               intermediate_dimension=intermediate_dimension,
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
                                               final_activation=final_activation,
                                               model_name=model_name)

    def define_encoder(self):
        z = self.encoder_mnist_input
        z = Reshape((28, 28, 1))(z)

        z = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=self.encoder_activation)(z)
        if self.enable_batch_normalization:
            z = BatchNormalization()(z)
        if self.enable_dropout:
            z = Dropout(rate=self.dropout_rate, seed=17)(z)

        z = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=self.encoder_activation)(z)
        if self.enable_batch_normalization:
            z = BatchNormalization()(z)
        if self.enable_dropout:
            z = Dropout(rate=self.dropout_rate, seed=17)(z)

        z = Flatten()(z)
        z_gaussian = Dense(self.gaussian_dimension, name="gaussian")(z)
        z = Reparametrization(name="latent_samples")(z_gaussian)
        encoder_output = [z_gaussian, z]

        encoder = Model([self.encoder_gaussian, self.encoder_mnist_input], encoder_output, name='encoder')
        encoder.summary()
        plot_model(encoder, to_file=os.path.join(self.image_directory, 'encoder.png'), show_shapes=True)

        return encoder, [z_gaussian, z]

    def define_decoder(self, encoder_output):
        decoder_gaussian_input = Input(shape=encoder_output[0].shape[1:], name='gaussian_input')
        decoder_latent_input = Input(shape=encoder_output[1].shape[1:], name='latent_input')
        x = decoder_latent_input
        gaussian = decoder_gaussian_input
        convolution_dimension = 784

        # Needed to prevent Keras from complaining that nothing was done to this tensor:
        identity_lambda = Lambda(lambda w: w, name="dec_identity_lambda")
        gaussian = identity_lambda(gaussian)

        x = Dense(convolution_dimension, activation=self.decoder_activation)(x)
        x = Reshape((7, 7, 16))(x)
        x = Conv2DTranspose(8,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            padding='same',
                            activation=self.decoder_activation)(x)
        if self.enable_batch_normalization:
            x = BatchNormalization()(x)
        if self.enable_dropout:
            x = Dropout(rate=self.dropout_rate, seed=17)(x)

        x = Conv2DTranspose(1,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            padding='same',
                            activation=self.decoder_activation)(x)
        if self.enable_batch_normalization:
            x = BatchNormalization()(x)
        if self.enable_dropout:
            x = Dropout(rate=self.dropout_rate, seed=17)(x)

        x = Reshape((28, 28))(x)

        decoder_output = [gaussian, x]
        decoder = Model([decoder_gaussian_input, decoder_latent_input], decoder_output, name='decoder')
        decoder.summary()
        plot_model(decoder, to_file=os.path.join(self.image_directory, 'decoder.png'), show_shapes=True)

        return decoder

    def define_autoencoder(self):
        encoder, z = self.define_encoder()
        decoder = self.define_decoder(z)

        auto_encoder_input = [self.auto_encoder_gaussian, self.auto_encoder_mnist_input]
        latent_space_input = encoder(auto_encoder_input)
        auto_encoder_output = decoder(latent_space_input)
        auto_encoder = Model(auto_encoder_input, auto_encoder_output, name='variational_auto_encoder')
        encoding_loss = EncodingLoss()
        reconstruction_loss = tensorflow.keras.losses.MeanSquaredError()
        auto_encoder.summary()
        plot_model(auto_encoder, to_file=os.path.join(self.image_directory, 'auto_encoder.png'), show_shapes=True)
        auto_encoder.compile(optimizers.Adam(lr=self.learning_rate),
                             loss=[encoding_loss, reconstruction_loss],
                             loss_weights=[self.beta, 764])
        return auto_encoder, encoder, decoder

    def get_fit_args(self):
        """
        Define a list of NumPy inputs and NumPy outputs of the Keras model. These are the actual data that flow through
        the Keras model.
        :return: A list of arguments for the fit method of the Keras model.
        """
        return [[self.gaussian_train, self.x_train], [self.gaussian_train, self.x_train]]

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
            fit_kwargs['callbacks'] = [self.early_stopping_callback, self.nan_termination_callback]
        else:
            fit_kwargs['callbacks'] = [self.nan_termination_callback]
        if self.has_validation_set:
            fit_kwargs['validation_data'] = ([self.gaussian_test, self.x_val], [self.gaussian_test, self.x_val])
        return fit_kwargs

    def fit_autoencoder(self):
        """
        Fit the autoencoder to the data.
        :return: A 4-tuple consisting of the autoencoder, encoder, and decoder Keras models, along with the history of
            the autoencoder, which stores training and validation metrics.
        """
        args = self.get_fit_args()
        kwargs = self.get_fit_kwargs()
        auto_encoder, encoder, decoder = self.define_autoencoder()
        history = auto_encoder.fit(*args, **kwargs)
        print("Variational autoencoder trained.\n")
        return auto_encoder, encoder, decoder, history

    def train(self):
        """
        Begin logging, train the autoencoder, use the autoencoder's history to plot loss curves, and save the parameters of the autoencoder, encoder, and decoder (respectively) to .h5 files.
        :return: None
        """
        if self.enable_logging:
            logs.begin_logging(self.experiment_directory)

        auto_encoder, encoder, decoder, history = self.fit_autoencoder()

        plots.loss(history, self.image_directory)

        self.save_model_weights(auto_encoder, encoder, decoder)

        self.plot_results((encoder, decoder))

    def predict(self, model, data=None):
        """
        Run a prediction on the given data set.
        :param model: A Keras model. In this case, either the autoencoder, the encoder, or the decoder.
        :param data: The data on which to predict. Default is None. If None, then data is set to the training data.
        :return: The model's prediction of the data.
        """
        if data is None:
            data = self.x_train
        return model.predict(data)

    def generate(self, decoder, number_of_samples=1):
        """
        Generate samples using the decoder of the learned autoencoder's generative model.
        :param decoder: A Keras model. Here's it's a decoder learned by training a VAE.
        :param number_of_samples: An integer denoting the number of samples to generate. Default is 1.
        :return: A NumPy array of data produced by the generative model.
        """
        # data = samples in the latent space.
        # return self.predict(decoder, data)


vae = ConvolutionalVAE(number_of_epochs=50,
                       is_restricted=True,
                       restriction_labels=[2],
                       enable_logging=True,
                       enable_rotations=True,
                       number_of_rotations=11,
                       angle_of_rotation=30,
                       enable_stochastic_gradient_descent=True,
                       encoder_activation='relu',
                       decoder_activation='relu',
                       final_activation='sigmoid',
                       learning_rate_initial=1e-2,
                       beta=1.5)
vae.train()
del vae
