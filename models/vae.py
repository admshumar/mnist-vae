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

from models.layers import vae_layers
from models.losses.losses import EncodingLoss
from utils import logs, operations, plots, directories, labels
from utils.loaders import MNISTLoader

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


class VAE:
    """
    Base class for variational autoencoders, from which all autoencoder models inherit.
    """

    @classmethod
    def get_kwargs(cls):
        """
        Factory method for the default keyword arguments for the VAE class constructor.
        :return: A dictionary of boolean valued keyword arguments.
        """
        kwargs = dict()

        """
        Boolean valued keyword arguments
        """
        kwargs['deep'] = True
        kwargs['enable_activation'] = True
        kwargs['enable_augmentation'] = False
        kwargs['enable_batch_normalization'] = True
        kwargs['enable_dropout'] = True
        kwargs['enable_early_stopping'] = False
        kwargs['enable_logging'] = True
        kwargs['enable_label_smoothing'] = False
        kwargs['enable_rotations'] = False
        kwargs['enable_stochastic_gradient_descent'] = False
        kwargs['has_custom_layers'] = True
        kwargs['has_validation_set'] = False
        kwargs['is_mnist'] = True
        kwargs['is_restricted'] = False
        kwargs['is_standardized'] = False
        kwargs['show'] = False

        """
        Integer, float, and string valued keyword arguments.
        """
        kwargs['number_of_clusters'] = 3
        kwargs['restriction_labels'] = [1, 2, 3]
        kwargs['intermediate_dimension'] = 512
        kwargs['exponent_of_latent_space_dimension'] = 1
        kwargs['augmentation_size'] = 100
        kwargs['covariance_coefficient'] = 0.2
        kwargs['number_of_epochs'] = 5
        kwargs['batch_size'] = 128
        kwargs['learning_rate_initial'] = 1e-5
        kwargs['learning_rate_minimum'] = 1e-6
        kwargs['encoder_activation'] = 'relu'  # 'relu' 'tanh' 'elu' 'softmax' 'sigmoid'
        kwargs['decoder_activation'] = 'relu'
        kwargs['final_activation'] = 'sigmoid'
        kwargs['dropout_rate'] = 0.5
        kwargs['l2_constant'] = 1e-4
        kwargs['early_stopping_delta'] = 1
        kwargs['beta'] = 1
        kwargs['smoothing_alpha'] = 0.5
        kwargs['number_of_rotations'] = 2
        kwargs['angle_of_rotation'] = 30

        return kwargs

    @classmethod
    def get_split_mnist_data(cls, val_size=0.5):
        """
        Grab the TensorFlow Keras incarnation of the MNIST data set, then split the training set into a training subset
        and a validation subset.
        :return: NumPy arrays of MNIST training, validation, and test sets.
        """
        (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                        test_size=val_size,
                                                        random_state=37,
                                                        stratify=y_test)
        return x_train, y_train, x_val, y_val, x_test, y_test

    @classmethod
    def shuffle(cls, data, labels):
        assert len(data) == len(labels)
        permutation = np.random.permutation(len(data))
        return data[permutation], labels[permutation]


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
                 model_name='vae'):

        self.model_name = model_name
        self.enable_logging = enable_logging
        self.enable_label_smoothing = enable_label_smoothing
        self.deep = deep
        self.is_mnist = is_mnist
        self.is_restricted = is_restricted
        self.restriction_labels = restriction_labels
        self.enable_early_stopping = enable_early_stopping and has_validation_set
        self.enable_rotations = enable_rotations
        self.number_of_rotations = number_of_rotations
        self.angle_of_rotation = angle_of_rotation

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
        self.restriction_labels = restriction_labels

        self.has_validation_set = has_validation_set
        if self.is_mnist:
            if self.has_validation_set:
                x_train, y_train, x_val, y_val, x_test, y_test = VAE.get_split_mnist_data()

                if self.is_restricted:
                    x_train, y_train = operations.restrict_data_by_label(x_train, y_train, restriction_labels)
                    x_val, y_val = operations.restrict_data_by_label(x_val, y_val, restriction_labels)
                    x_test, y_test = operations.restrict_data_by_label(x_test, y_test, restriction_labels)

                if enable_rotations:
                    print("Rotations enabled!")
                    x_train = MNISTLoader('x_train').load(restriction_labels, number_of_rotations, angle_of_rotation)
                    x_val = MNISTLoader('x_val').load(restriction_labels, number_of_rotations, angle_of_rotation)
                    x_test = MNISTLoader('x_test').load(restriction_labels, number_of_rotations, angle_of_rotation)

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

                if enable_rotations:
                    print("Rotations enabled!")
                    x_train = MNISTLoader('train').load(restriction_labels, number_of_rotations,
                                                        angle_of_rotation)
                    y_train = MNISTLoader('train').load(restriction_labels, number_of_rotations,
                                                        angle_of_rotation, label=True)
                    x_train, y_train = VAE.shuffle(x_train, y_train)

                    x_test = MNISTLoader('test').load(restriction_labels, number_of_rotations,
                                                      angle_of_rotation)
                    y_test = MNISTLoader('test').load(restriction_labels, number_of_rotations,
                                                      angle_of_rotation, label=True)
                    x_test, y_test = VAE.shuffle(x_test, y_test)

                self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test

                if self.enable_label_smoothing:
                    self.y_train_smooth = labels.Smoother(y_train, alpha=smoothing_alpha).smooth()
                    self.y_test_smooth = labels.Smoother(y_test, alpha=smoothing_alpha).smooth()

            self.data_width, self.data_height = self.x_train.shape[1], self.x_train.shape[2]
            self.data_dimension = self.data_width * self.data_height
            self.intermediate_dimension = intermediate_dimension

            self.x_train = operations.normalize(self.x_train)
            self.x_test = operations.normalize(self.x_test)
            if self.has_validation_set:
                self.x_val = operations.normalize(self.x_val)

            self.gaussian_train = operations.get_gaussian_parameters(self.x_train)
            self.gaussian_val = operations.get_gaussian_parameters(self.x_test)

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

        self.hyper_parameter_list = [self.number_of_epochs,
                                     self.batch_size,
                                     self.learning_rate,
                                     self.encoder_activation,
                                     self.decoder_activation,
                                     self.enable_batch_normalization,
                                     self.enable_dropout,
                                     self.dropout_rate,
                                     self.l2_constant,
                                     self.patience_limit,
                                     self.early_stopping_delta,
                                     self.latent_dim]

        if self.is_mnist:
            self.hyper_parameter_list.append("mnist")

        if self.is_restricted:
            self.hyper_parameter_list.append(f"restricted_{str(restriction_labels)}")

        if self.enable_augmentation:
            augmentation_string = "_".join(["augmented", str(covariance_coefficient), str(augmentation_size)])
            self.hyper_parameter_list.append(augmentation_string)

        if not self.enable_activation:
            self.hyper_parameter_list.append("PCA")

        if self.enable_rotations:
            self.hyper_parameter_list.append(f"rotated_{number_of_rotations},{angle_of_rotation}")

        if beta > 1:
            self.hyper_parameter_list.append(f"beta_{beta}")

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
        self.gaussian_shape = 2 * self.latent_dim
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

    def assign_soft_labels(self, x_train_latent, x_test_latent):
        """
        Fit a Gaussian mixture model to a latent representation of training data, then use the components in the
        mixture model to return a collection of class probabilities.
        class probabilities for a latent representation of test data.
        :param x_train_latent: A NumPy data array.
        :param x_test_latent: A NumPy data array.
        :return: A NumPy data array of soft class probabilities.
        """
        mixture_model = GaussianMixture(self.number_of_clusters)
        mixture_model.fit(x_train_latent)
        # Get the parameters for each Gaussian density in the mixture model, then fire up SciPy to compute the density
        # values for each data point in the latent representation.
        soft_labels = None  # Here, the soft labels are gotten from the densities of each Gaussian
        print(soft_labels)
        return soft_labels

    def save_model_weights(self, autoencoder, encoder, decoder):
        """
        Save the weights of the autoencoder, encoder, and decoder (respectively) to .h5 files.
        :param autoencoder: A Keras model, in this case an autoencoder.
        :param encoder: A Keras model, in this case an encoder.
        :param decoder: A Keras model, in this case a decoder.
        :return: None
        """
        model_directory = os.path.join(self.image_directory, 'models')
        auto_encoder_filepath = os.path.join(model_directory, 'autoencoder.h5')
        encoder_filepath = os.path.join(model_directory, 'encoder.h5')
        decoder_filepath = os.path.join(model_directory, 'decoder.h5')

        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        autoencoder.save_weights(auto_encoder_filepath)
        encoder.save_weights(encoder_filepath)
        decoder.save_weights(decoder_filepath)

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
