from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow

import tensorflow.keras.backend as k
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from models.layers.vae_layers import Reparametrization
from models.losses.losses import EncodingLoss
from utils import logs, operations, plots, directories, labels

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


class ConvolutionalVAE:

    @classmethod
    def get_split_mnist_data(cls, val_size=0.5):
        """
        Grab the TensorFlow Keras incarnation of the MNIST data set, then split the training set into a training subset
        and a validation subset.
        :return: NumPy arrays of MNIST training, validation, and test sets.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                        test_size=val_size,
                                                        random_state=37,
                                                        stratify=y_test)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def __init__(self,
                 deep=True,
                 is_mnist=True,
                 number_of_clusters=3,
                 is_restricted=False,
                 is_standardized=False,
                 intermediate_dimension=512,
                 enable_stochastic_gradient_descent=False,
                 has_custom_layers=True,
                 has_validation_set=False,
                 exponent_of_latent_space_dimension=1,
                 enable_augmentation=False,
                 augmentation_size=100,
                 covariance_coefficient=0.2,
                 show=False,
                 number_of_epochs=5,
                 batch_size=128,
                 learning_rate_initial=1e-5,
                 learning_rate_minimum=1e-6,
                 restriction_labels=None,
                 enable_label_smoothing=False,
                 smoothing_alpha=0.5,
                 enable_batch_normalization=True,
                 enable_dropout=True,
                 enable_activation=True,
                 encoder_activation='relu',  # 'relu', 'tanh', 'elu', 'softmax', 'sigmoid'
                 decoder_activation='relu',
                 final_activation='sigmoid',
                 dropout_rate=0.2,
                 l2_constant=1e-4,
                 early_stopping_delta=1,
                 beta=1,
                 enable_logging=True
                 ):
        """
        For an MNIST variational autoencoder, we have the usual options that control network hyperparameters. In
        addition, . . .
        :param deep: A boolean indicating whether the autoencoder has more than one hidden layer.
        :param is_mnist: A boolean indicating whether the data set is MNIST.
        :param number_of_clusters: An integer indicating the number of clusters to be produced by clustering algorithms.
        :param is_restricted: A boolean indicating whether at least one class label is to be ignored.
        :param restriction_labels: A list of integers that indicate the class labels to be retained in the data set.
        :param is_standardized: A boolean indicating whether the train_contrastive_mlp and test sets are standardized before being
            input into the network.
        :param enable_stochastic_gradient_descent: A boolean indicating whether SGD is performed during training.
        :param has_custom_layers: A boolean indicating the layer structure of the network.
        :param exponent_of_latent_space_dimension: An integer indicating the size of the latent space.
        :param enable_augmentation: A boolean indicating whether data augmentation is to be performed.
        :param augmentation_size: An integer indicating how much data are to be sampled for each existing data point.
        :param covariance_coefficient: A float indicating the scalar multiple of the identity covariance matrix for the
            Gaussians that are used to augment the data.
        :param show: A boolean indicating whether matplotlib.pyplot.show is invoked after inference.
            By default this is False.
        :param number_of_epochs: An integer indicating the number of training epochs.
        :param batch_size: An integer indicating the batch size.
        :param learning_rate_initial: A float indicating the initial learning rate.
        :param learning_rate_minimum: A float indicating the minimum learning rate (for a learning rate scheduler).
        :param enable_batch_normalization: A boolean indicating whether batch normalization is performed.
        :param enable_dropout: A boolean indicating whether dropout is performed during training.
        :param enable_activation: A boolean indicating whether activation functions are used during training. In the
            case of an autoencoder, removing network activations will give us an algorithm similar to PCA.
        :param encoder_activation: A boolean indicating the activation function to be used in the encoder layers.
        :param decoder_activation: A boolean indicating the activation function to be used in the decoder layers.
        :param dropout_rate: A float indicating the proportion of neurons to be deactivated.
        :param l2_constant: A float indicating the amount of L2 regularization.
        :param early_stopping_delta: A float indicating the number of epochs before training is halted due to an
            insufficient change in the validation loss.
        :param beta: A float indicating the beta hyperparameter for a beta-variational autoencoder. Default is 0.
        """
        self.model_name = "vae_conv"
        self.enable_logging = enable_logging
        self.enable_label_smoothing = enable_label_smoothing
        self.deep = deep
        self.is_mnist = is_mnist
        self.is_restricted = is_restricted
        self.restriction_labels = restriction_labels
        if self.is_restricted:
            self.number_of_clusters = len(self.restriction_labels)
        else:
            self.number_of_clusters = number_of_clusters

        self.is_standardized = is_standardized
        self.enable_stochastic_gradient_descent = enable_stochastic_gradient_descent
        self.has_custom_layers = has_custom_layers
        self.exponent_of_latent_space_dimension = exponent_of_latent_space_dimension
        self.enable_augmentation = enable_augmentation
        self.augmentation_size = augmentation_size
        self.covariance_coefficient = covariance_coefficient
        self.show = show

        self.has_validation_set = has_validation_set
        if self.is_mnist:
            if self.has_validation_set:
                x_train, y_train, x_val, y_val, x_test, y_test = ConvolutionalVAE.get_split_mnist_data()

                if self.is_restricted:
                    x_train, y_train = operations.restrict_data_by_label(x_train, y_train, restriction_labels)
                    x_val, y_val = operations.restrict_data_by_label(x_val, y_val, restriction_labels)
                    x_test, y_test = operations.restrict_data_by_label(x_test, y_test, restriction_labels)

                self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test \
                    = x_train, y_train, x_val, y_val, x_test, y_test

                if self.enable_label_smoothing:
                    self.y_train_smooth = labels.Smoother(y_train, alpha=smoothing_alpha).smooth()
                    self.y_val_smooth = labels.Smoother(y_val, alpha=smoothing_alpha).smooth()
                    self.y_test_smooth = labels.Smoother(y_test, alpha=smoothing_alpha).smooth()
            else:
                (x_train, y_train), (x_test, y_test) = mnist.load_data()

                if self.is_restricted:
                    x_train, y_train = operations.restrict_data_by_label(x_train, y_train, restriction_labels)
                    x_test, y_test = operations.restrict_data_by_label(x_test, y_test, restriction_labels)

                self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test

                if self.enable_label_smoothing:
                    self.y_train_smooth = labels.Smoother(y_train, alpha=smoothing_alpha).smooth()
                    self.y_test_smooth = labels.Smoother(y_test, alpha=smoothing_alpha).smooth()

            self.data_width, self.data_height = self.x_train.shape[1], self.x_train.shape[2]
            self.data_dimension = self.data_width * self.data_height
            self.intermediate_dimension = intermediate_dimension

            self.x_train = operations.normalize(self.x_train)
            self.x_val = operations.normalize(self.x_val)
            self.x_test = operations.normalize(self.x_test)

            self.gaussian_train = operations.get_gaussian_parameters(self.x_train)
            self.gaussian_val = operations.get_gaussian_parameters(self.x_test)
            self.restriction_labels = restriction_labels

        self.x_train_length = len(self.x_train)
        self.x_test_length = len(self.x_test)

        """
        Hyperparameters for the neural network.
        """
        self.number_of_epochs = number_of_epochs

        if self.enable_stochastic_gradient_descent:
            self.batch_size = batch_size
        else:
            self.batch_size = len(self.x_train)

        self.learning_rate = learning_rate_initial
        self.learning_rate_minimum = learning_rate_minimum
        self.enable_batch_normalization = enable_batch_normalization
        self.enable_dropout = enable_dropout
        self.enable_activation = enable_activation
        self.encoder_activation = encoder_activation  # 'relu', 'tanh', 'elu', 'softmax', 'sigmoid'
        self.decoder_activation = decoder_activation
        self.final_activation = final_activation
        self.dropout_rate = dropout_rate
        self.l2_constant = l2_constant
        self.patience_limit = self.number_of_epochs // 5
        self.early_stopping_delta = early_stopping_delta

        self.latent_dim = 2
        self.gaussian_dimension = 2 * self.latent_dim

        self.beta = max(beta, 1)
        if self.beta > 1:
            self.enable_beta = True
        else:
            self.enable_beta = False

        self.hyper_parameter_list = [self.number_of_epochs,
                                     self.batch_size,
                                     self.learning_rate,
                                     self.encoder_activation,
                                     self.decoder_activation,
                                     self.enable_batch_normalization,
                                     self.enable_dropout,
                                     self.dropout_rate,
                                     self.enable_beta,
                                     self.beta,
                                     self.l2_constant,
                                     self.patience_limit,
                                     self.early_stopping_delta,
                                     self.latent_dim]

        if self.is_mnist:
            self.hyper_parameter_list.append("mnist")

        if self.is_restricted:
            restriction_label_string = ''
            for label in restriction_labels:
                restriction_label_string += str(label)
                self.hyper_parameter_list.append("restricted_{}".format(restriction_label_string))

        if self.enable_augmentation:
            augmentation_string = "_".join(["augmented", str(covariance_coefficient), str(augmentation_size)])
            self.hyper_parameter_list.append(augmentation_string)

        if not self.enable_activation:
            self.hyper_parameter_list.append("PCA")

        self.hyper_parameter_string = '_'.join([str(i) for i in self.hyper_parameter_list])

        self.directory_counter = directories.DirectoryCounter(self.hyper_parameter_string)
        self.directory_number = self.directory_counter.count()
        self.hyper_parameter_string = '_'.join([self.hyper_parameter_string, 'x{:02d}'.format(self.directory_number)])
        self.directory = directories.DirectoryCounter.make_output_directory(self.hyper_parameter_string,
                                                                            self.model_name)
        self.image_directory = os.path.join('images', self.directory)

        """
        Tensorflow Input instances for declaring model inputs.
        """
        self.mnist_shape = self.x_train.shape[1:]
        self.gaussian_shape = 2*self.latent_dim
        self.encoder_gaussian = Input(shape=self.gaussian_shape, name='enc_gaussian')
        self.encoder_mnist_input = Input(shape=self.mnist_shape, name='enc_mnist')
        self.auto_encoder_gaussian = Input(shape=self.gaussian_shape, name='ae_gaussian')
        self.auto_encoder_mnist_input = Input(shape=self.mnist_shape, name='ae_mnist')

        """
        Callbacks to TensorBoard for observing the model structure and network training curves.
        """
        self.tensorboard_callback = TensorBoard(log_dir=os.path.join(self.directory, 'tensorboard_logs'),
                                                histogram_freq=2,
                                                write_graph=True,
                                                write_images=True)

        self.early_stopping_callback = EarlyStopping(monitor='val_loss',
                                                     min_delta=self.early_stopping_delta,
                                                     patience=self.patience_limit,
                                                     mode='auto',
                                                     restore_best_weights=True)

        self.learning_rate_callback = ReduceLROnPlateau(monitor='val_loss',
                                                        factor=0.1,
                                                        patience=50,
                                                        min_lr=self.learning_rate_minimum)

        self.nan_termination_callback = TerminateOnNaN()

        self.colors = ['#00B7BA', '#FFB86F', '#5E6572', '#6B0504', '#BA5C12']

    # def compute_convolution_dimension(self, number_of_convolutions, kernel_size, strides):
        # Returns the product of the shape of a feature map after all convolution operations have been performed.

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

    def get_generic_fit_kwargs(self):
        """
        Get arguments for fitting an arbitrary Keras model.
        :return:
        """
        fit_kwargs = dict()
        fit_kwargs['epochs'] = self.number_of_epochs
        fit_kwargs['batch_size'] = self.batch_size
        fit_kwargs['callbacks'] = [self.early_stopping_callback, self.nan_termination_callback]
        return fit_kwargs

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
        fit_kwargs = self.get_generic_fit_kwargs()
        if self.has_validation_set:
            fit_kwargs['validation_data'] = ([self.gaussian_val, self.x_val], [self.gaussian_val, self.x_val])
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

    def fit_contrastive_mlp(self):
        args = self.get_contrastive_mlp_fit_args()
        kwargs = self.get_generic_fit_kwargs()
        contrastive_mlp, _ = self.define_contrastive_mlp()
        history = contrastive_mlp.fit(*args, **kwargs)
        print("Contrastive MLP trained.\n")
        return contrastive_mlp, history

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

    def save_model_weights(self, model, filename):
        """
        Save the weights of the autoencoder, encoder, and decoder (respectively) to .h5 files.
        :param autoencoder: A Keras model, in this case an autoencoder.
        :param encoder: A Keras model, in this case an encoder.
        :param decoder: A Keras model, in this case a decoder.
        :return: None
        """
        model_directory = os.path.join(self.image_directory, 'models')
        model_filepath = os.path.join(model_directory, filename + '.h5')

        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        model.save_weights(model_filepath)

    def plot_results(self, models):
        """Plots labels and MNIST digits as a function of the 2D latent vector

        # Arguments
            models (tuple): encoder and decoder models
            data (tuple): test data and label
            batch_size (int): prediction batch size
            model_name (string): which model is using this function
        """
        encoder, decoder = models
        test_gaussian = operations.get_gaussian_parameters(self.x_test)
        os.makedirs(self.image_directory, exist_ok=True)
        filename = os.path.join(self.image_directory, "vae_mean.png")

        # display a 2D plot of the digit classes in the latent space
        z_gaussian, z_mnist = encoder.predict([test_gaussian, self.x_test], batch_size=self.batch_size)
        z_mean, z_covariance = operations.split_gaussian_parameters(z_gaussian)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=self.y_test)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.savefig(filename)
        if self.show:
            plt.show()

        filename = os.path.join(self.image_directory, "digits_over_latent.png")
        # display a 30x30 2D manifold of digits
        n = 30
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4.5, 3.5, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                dummy_gaussian = np.array([[0, 0, 1, 1]])
                z_sample = np.array([[xi, yi]])
                x_decoded = decoder.predict([dummy_gaussian, z_sample])
                digit = x_decoded[1].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = (n - 1) * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(filename)
        if self.show:
            plt.show()

    def train_autoencoder(self):
        """
        Begin logging, train the autoencoder, use the autoencoder's history to plot loss curves, and save the parameters
        of the autoencoder, encoder, and decoder (respectively) to .h5 files.
        :return: None
        """
        if self.enable_logging:
            logs.begin_logging(self.directory)

        auto_encoder, encoder, decoder, history = self.fit_autoencoder()

        plots.plot_loss_curves(history, self.image_directory)

        self.save_model_weights(auto_encoder, "auto_encoder")
        self.save_model_weights(encoder, "encoder")
        self.save_model_weights(decoder, "decoder")

        self.plot_results((encoder, decoder))


vae = ConvolutionalVAE(number_of_epochs=12,
                       is_restricted=True,
                       restriction_labels=[6, 9],
                       enable_dropout=True,
                       enable_label_smoothing=False,
                       smoothing_alpha=.5,
                       enable_logging=True,
                       enable_batch_normalization=True,
                       enable_stochastic_gradient_descent=True,
                       encoder_activation='relu',
                       decoder_activation='relu',
                       final_activation='sigmoid',
                       learning_rate_initial=1e-2,
                       has_validation_set=True,
                       beta=2)
vae.train_contrastive_mlp()
del vae
