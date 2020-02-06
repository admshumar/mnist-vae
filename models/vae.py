from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import sys

import numpy as np

import tensorflow
import tensorflow.keras.backend as k
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.utils import plot_model

from data import image_directory_counter

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from scipy.special import comb

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm


class VariationalAutoencoder:

    @classmethod
    def plot_loss_curves(cls, model_history, directory):
        """
        Plot loss curves for a Keras model using MatPlotLib. (NOTE: This plots only *after* training completes. It would
        be nice to plot concurrently with model training just in case something goes wrong, otherwise you'll get no
        loss curves.)
        :param model_history: A dictionary of evidence_lower_bound values.
        :param directory: A string indicating the directory to which the evidence_lower_bound image is written.
        :return: None
        """
        filename = os.path.join(directory, 'losses.png')
        model_losses = {'loss'}.intersection(set(model_history.history.keys()))

        fig = plt.figure(dpi=200)
        for loss in model_losses:
            plt.plot(model_history.history[loss])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        fig.savefig(filename)
        plt.close(fig)

    @classmethod
    def standardize(cls, target_matrix, reference_matrix):
        """
        Standardize the columns of a NumPy array.
        :param target_matrix: A NumPy array to be standardized.
        :param reference_matrix: A NumPy array whose parameters are applied in the standardization.
        :return: A standardized NumPy array.
        """
        mean_matrix = np.mean(reference_matrix, axis=0)
        standard_deviation_matrix = np.std(reference_matrix, axis=0)
        standardized_matrix = (target_matrix - mean_matrix) / standard_deviation_matrix
        return standardized_matrix

    @classmethod
    def reparametrize_and_sample(cls, gaussian_parameters):
        """
        The given mean and logarithmic covariance are regarded as parameters for an isotropic Gaussian. Sampling from
        this Gaussian is identical to first sampling from a Gaussian with zero mean and identity covariance and
        then scaling and translating the samples. This is the "reparametrization trick" of Kingma and Welling, found in
        their paper "Autoencoding Variational Bayes".
        :param gaussian_parameters: A List of NumPy arrays, consisting of the mean and covariance of a Gaussian.
        :return: A NumPy array representing a set of samples drawn from a reparametrized isotropic Gaussian.
        """
        mean, logarithmic_covariance = gaussian_parameters[0], gaussian_parameters[1]
        standard_deviation = tensorflow.keras.backend.exp(0.5 * logarithmic_covariance)
        shape = tensorflow.shape(mean)  # NOT mean.shape
        epsilon = tensorflow.keras.backend.random_normal(shape)
        return mean + epsilon * standard_deviation

    @classmethod
    @tensorflow.function
    def reconstruction_loss(cls,
                            inputs,
                            outputs,
                            # x_mean,
                            # x_logarithmic_covariance,
                            approximate_reconstruction_loss=True):
        batch_size, width, height = inputs.shape[0], inputs.shape[1], inputs.shape[2]
        inputs = Reshape((width * height,))(inputs)
        outputs = Reshape((width * height,))(outputs)
        if approximate_reconstruction_loss:
            loss_vector = mean_squared_error(inputs, outputs)
            loss = tensorflow.keras.backend.mean(loss_vector)
            return loss
        # else:
            # implement later

    @classmethod
    @tensorflow.function
    def encoding_loss(cls, z_mean, z_logarithmic_covariance):
        kullback_leibler_divergence_vector = \
            -1 - z_logarithmic_covariance \
            + tensorflow.keras.backend.square(z_mean) + tensorflow.keras.backend.exp(z_logarithmic_covariance)
        kullback_leibler_divergence = tensorflow.keras.backend.sum(kullback_leibler_divergence_vector) # 0.5 *
        return kullback_leibler_divergence

    @classmethod
    @tensorflow.function
    def evidence_lower_bound(cls, inputs, outputs, z_mean, z_logarithmic_covariance,
                             approximate_reconstruction_loss=True):
        """
        Compute the evidence lower bound of a variational autoencoder (more generally a beta-VAE for beta > 1).
        :param inputs:
        :param outputs:
        :param z_mean:
        :param z_logarithmic_covariance:
        :param approximate_reconstruction_loss:
        :param beta:
        :return:
        """
        reconstruction_loss = VariationalAutoencoder.reconstruction_loss(inputs, outputs,
                                                                         # x_mean,
                                                                         # x_logarithmic_covariance,
                                                                         approximate_reconstruction_loss)
        kullback_leibler_divergence = VariationalAutoencoder.encoding_loss(z_mean,
                                                                           z_logarithmic_covariance)
        return reconstruction_loss - kullback_leibler_divergence

    @classmethod
    def normalize(cls, x_train, x_test):
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        return x_train, x_test

    @classmethod
    def reshape_for_convolution(cls, x_train, x_test):
        """
        Keras's convolutional layers want as inputs 4D tensors with shape: (samples, channels, rows, cols),
        if data_format='channels_first' or 4D tensors with shape: (samples, rows, cols, channels) if
        data_format='channels_last'. The default is data_format='channels_last'.
        :param x_train:
        :param x_test:
        :param dimension:
        :return:
        """
        x_train = np.reshape(x_train, (-1, x_train.shape[1], x_train.shape[2], 1))
        x_test = np.reshape(x_test, (-1, x_test.shape[1], x_test.shape[1], 1))
        return x_train, x_test

    @classmethod
    def reshape_for_dense_map(cls, x_train, x_test, dimension):

        x_train = np.reshape(x_train, (-1, 1, dimension * dimension))
        x_test = np.reshape(x_test, (-1, 1, dimension * dimension))
        return x_train, x_test

    @classmethod
    def aggregate_labels(cls, list_of_labels):
        """
        Construct the class distribution of labels.
        :param list_of_labels: A NumPy array of class labels.
        :return: A dictionary whose keys are classes and whose values are class probabilities.
        """
        class_distribution = {}
        unique_labels = np.unique(list_of_labels)

        for label in unique_labels:
            label_frequency = len(list_of_labels[list_of_labels == label])
            label_proportion = label_frequency / len(list_of_labels)
            class_distribution[label] = label_proportion

        return class_distribution

    @classmethod
    def make_output_directory(cls, hyper_parameter_string):
        """
        Make an output directory indexed by a set of hyperparameters.
        :return: A string corresponding to the output directory.
        """
        output_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'images', hyper_parameter_string))
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        return output_directory

    @classmethod
    def get_power_sequence(cls, data_dimension, exponent):
        """
        Given an integer, construct a list of integers starting with the given integer, then all positive
        powers of two that are less than the given integer.
        :param data_dimension: An integer that represents the dimension of an input data set.
        :param exponent: The exponent of the dimension of the latent space representation (which is expressed as a
            power of two.)
        :return: A list of integers, which are the dimensions of the feature space representations in the autoencoder.
        """
        k = len(bin(data_dimension)) - 3
        sequence = [2 ** i for i in range(exponent, k + 1)]
        if sequence[-1] == data_dimension:
            sequence = sequence[:-1]
        sequence.append(data_dimension)
        return sequence[::-1]

    @classmethod
    def get_mnist_data(cls):
        """
        Grab the TensorFlow Keras incarnation of the MNIST data set.
        :return: A NumPy array of MNIST training and test sets.
        """
        (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
        return x_train, y_train, x_test, y_test

    @classmethod
    def split_data(cls, data,
                   label_index,
                   random_state=17,
                   test_size=0.30,
                   is_stratified=False,
                   enable_cross_validation=False,
                   number_of_folds=5):

        if enable_cross_validation:
            skf = StratifiedKFold(n_splits=number_of_folds,
                                  random_state=random_state)
            skf.get_n_splits()
            # Determine the construction of train and test

        elif is_stratified:
            train, test = train_test_split(data,
                                           test_size=test_size,
                                           random_state=random_state,
                                           stratify=data[:, label_index])

        else:
            train, test = train_test_split(data,
                                           test_size=test_size,
                                           random_state=random_state)

        return train, test

    @classmethod
    def approximate_reconstruction_loss(cls, y_true, y_pred):
        loss = y_true-y_pred
        loss = k.flatten(loss)
        loss = k.square(loss)
        loss = k.mean(loss)
        return loss

    @classmethod
    def encoding_loss(cls, y_true, y_pred):
        """
        Compute the Kullback-Leibler divergence between an arbitrary multivariate Gaussian and the Gaussian with zero
        mean and identity covariance. (In the future, this should be extended to KL for two arbitrary Gaussians.)
        :param y_true: A dummy Keras tensor.
        :param y_pred: A Keras tensor consisting of the parameters of an arbitrary multivariate Gaussian, which is
            parametrized by the encoder model and thus represents the approximate posterior of the variational
            autoencoder's generative model.
        :return: A float indicating the loss value.
        """
        mean = y_pred[0]
        logarithmic_covariance = y_pred[1]
        kullback_leibler_divergence_vector = \
            -1 - logarithmic_covariance \
            + tensorflow.keras.backend.square(mean) + tensorflow.keras.backend.exp(logarithmic_covariance)
        kullback_leibler_divergence = tensorflow.keras.backend.sum(kullback_leibler_divergence_vector)
        kullback_leibler_divergence = 0.5 * kullback_leibler_divergence
        return kullback_leibler_divergence

    def __init__(self,
                 deep=True,
                 is_mnist=True,
                 number_of_clusters=3,
                 is_restricted=False,
                 is_standardized=False,
                 restriction_labels=[1, 2, 3],
                 enable_stochastic_gradient_descent=False,
                 has_custom_layers=True,
                 exponent_of_latent_space_dimension=1,
                 enable_augmentation=False,
                 augmentation_size=100,
                 covariance_coefficient=0.2,
                 show_representations=False,
                 number_of_epochs=5,
                 batch_size=128,
                 learning_rate_initial=1e-3,
                 learning_rate_minimum=1e-6,
                 enable_batch_normalization=True,
                 enable_dropout=True,
                 enable_activation=True,
                 encoder_activation='relu',  # 'relu', 'tanh', 'elu', 'softmax', 'sigmoid'
                 decoder_activation='tanh',
                 dropout_rate=0.5,
                 l2_constant=1e-4,
                 early_stopping_delta=0.1,
                 beta=1
                 ):
        """
        For an MNIST variational autoencoder, we have the usual options that control network hyperparameters. In
        addition, . . .
        :param deep: A boolean indicating whether the autoencoder has more than one hidden layer.
        :param is_mnist: A boolean indicating whether the data set is MNIST.
        :param number_of_clusters: An integer indicating the number of clusters to be produced by clustering algorithms.
        :param is_restricted: A boolean indicating whether at least one class label is to be ignored.
        :param restriction_labels: A list of integers that indicate the class labels to be retained in the data set.
        :param is_standardized: A boolean indicating whether the train and test sets are standardized before being
            input into the network.
        :param enable_stochastic_gradient_descent: A boolean indicating whether SGD is performed during training.
        :param has_custom_layers: A boolean indicating the layer structure of the network.
        :param exponent_of_latent_space_dimension: An integer indicating the size of the latent space.
        :param enable_augmentation: A boolean indicating whether data augmentation is to be performed.
        :param augmentation_size: An integer indicating how much data are to be sampled for each existing data point.
        :param covariance_coefficient: A float indicating the scalar multiple of the identity covariance matrix for the
            Gaussians that are used to augment the data.
        :param show_representations: A boolean indicating whether matplotlib.pyplot.show is invoked after inference.
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
            insufficient change in the training evidence_lower_bound.
        :param beta: A float indicating the beta hyperparameter for a beta-variational autoencoder. Default is 0.
        """
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
        self.show_representations = show_representations
        self.restriction_labels = restriction_labels

        if self.is_mnist:
            self.x_train, self.y_train, self.x_test, self.y_test = VariationalAutoencoder.get_mnist_data()
            self.data_width, self.data_height = self.x_train.shape[1], self.x_train.shape[2]
            self.data_dimension = self.data_width * self.data_height
            self.x_train, self.x_test = VariationalAutoencoder.normalize(self.x_train, self.x_test)
            # self.x_train, self.x_test = VariationalAutoencoder.reshape_for_convolution(self.x_train, self.x_test,)
        # else: get other data . . . implement later

        self.x_train_length = len(self.x_train)
        self.x_test_length = len(self.x_test)

        if self.is_mnist:
            self.w_train = self.x_train
            self.w_test = self.x_test
        # else:

        """
        Hyperparameters for the neural network.
        """
        self.encoder_input_shape = self.x_train.shape[1:]
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
        self.dropout_rate = dropout_rate
        self.l2_constant = l2_constant
        self.patience_limit = self.number_of_epochs // 10
        self.early_stopping_delta = early_stopping_delta

        if self.has_custom_layers:
            self.power_sequence = [self.data_dimension, 48, 24, 4]
        else:
            self.power_sequence = self.get_power_sequence(self.data_dimension, self.exponent_of_latent_space_dimension)

        self.latent_dim = self.power_sequence[-1]

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

        self.directory_counter = image_directory_counter.DirectoryCounter(self.hyper_parameter_string)
        self.directory_number = self.directory_counter.count()
        self.hyper_parameter_string = '_'.join([self.hyper_parameter_string, 'x{:02d}'.format(self.directory_number)])
        self.directory = VariationalAutoencoder.make_output_directory(self.hyper_parameter_string)
        self.image_directory = os.path.join('images', self.directory)

        """
        Tensorflow Input tensor for input into the encoder model.
        """
        self.encoder_input = Input(shape=self.encoder_input_shape, name='encoder_input')

        """
        Callbacks to TensorBoard for observing the model structure and network training curves.
        """
        self.tensorboard_callback = TensorBoard(log_dir=os.path.join(self.directory, 'tensorboard_logs'),
                                                histogram_freq=1,
                                                write_graph=False,
                                                write_images=True)
        """
        self.early_stopping_callback = EarlyStopping(monitor='val_loss',
                                                     min_delta=self.early_stopping_delta,
                                                     patience=self.patience_limit,
                                                     mode='auto',
                                                     restore_best_weights=True)
        """

        self.learning_rate_callback = ReduceLROnPlateau(monitor='val_loss',
                                                        factor=0.1,
                                                        patience=50,
                                                        min_lr=self.learning_rate_minimum)

        self.colors = ['#00B7BA', '#FFB86F', '#5E6572', '#6B0504', '#BA5C12']

    def begin_logging(self):
        log_filename = os.path.join(self.directory, 'experiment.log')
        log_err_filename = os.path.join(self.directory, 'error.log')
        sys.stdout = open(log_filename, "w")
        sys.stderr = open(log_err_filename, "w")

    def end_logging(self):
        sys.stdout.close()

    def define_encoder(self):
        z = self.encoder_input
        z = Reshape((28, 28, 1,))(z)

        z = Conv2D(filters=8,
                   kernel_size=(3, 3),
                   padding="same",
                   strides=(2, 2),
                   activation=self.encoder_activation,
                   input_shape=self.encoder_input_shape)(z)
        z = BatchNormalization(axis=-1, epsilon=1e-5)(z)

        z = Conv2D(filters=16,
                   kernel_size=(3, 3),
                   padding="same",
                   strides=(2, 2),
                   activation=self.encoder_activation,
                   input_shape=self.encoder_input_shape)(z)
        z = BatchNormalization(axis=-1, epsilon=1e-5)(z)

        z = Flatten()(z)
        z = Dense(20, activation=self.encoder_activation)(z)
        z = BatchNormalization(epsilon=1e-5)(z)

        z_mean = Dense(2, activation=self.encoder_activation)(z)
        z_logarithmic_covariance = Dense(2, activation=self.encoder_activation)(z)
        z = Lambda(VariationalAutoencoder.reparametrize_and_sample)([z_mean, z_logarithmic_covariance])
        encoder_output = [z_mean, z_logarithmic_covariance, z]

        encoder = Model(self.encoder_input, encoder_output, name='encoder')
        encoder.summary()
        plot_model(encoder, to_file=os.path.join(self.image_directory, 'encoder.png'), show_shapes=True)

        return encoder, z

    def define_decoder(self, z):
        decoder_input = Input(shape=z.shape[1:], name='decoder_input')
        x = decoder_input

        x = Dense(784, activation=self.decoder_activation)(x)
        x = BatchNormalization(epsilon=1e-5)(x)
        x = Reshape((7, 7, 16))(x)

        x = Conv2DTranspose(filters=8,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            padding="same",
                            activation=self.decoder_activation)(x)
        x = BatchNormalization(epsilon=1e-5)(x)

        x = Conv2DTranspose(filters=8,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            padding="same",
                            activation=self.decoder_activation)(x)
        x = BatchNormalization(epsilon=1e-5)(x)
        x = Conv2DTranspose(filters=1,
                            kernel_size=(3, 3),
                            padding="same",
                            activation=self.decoder_activation,
                            name='decoder_output')(x)
        x = Reshape((28, 28))(x)
        decoder_output = x
        decoder = Model(decoder_input, decoder_output, name='decoder')
        decoder.summary()
        plot_model(decoder, to_file=os.path.join(self.image_directory, 'decoder.png'), show_shapes=True)

        return decoder
        #small project, red cross, developing projects for health, on 17+-2days in March, workshop
        #

    def define_autoencoder(self):
        encoder, z = self.define_encoder()
        decoder = self.define_decoder(z)

        auto_encoder_input = self.encoder_input
        latent_space_input = encoder(auto_encoder_input)[2]
        auto_encoder_output = decoder(latent_space_input)#, tensorflow.expand_dims(encoder_output, -1)
        auto_encoder = Model(auto_encoder_input, auto_encoder_output, name='variational_auto_encoder')

        # reconstruction_loss = MeanSquaredError(auto_encoder_input, auto_encoder_output)
        # encoding_loss = VariationalAutoencoder.encoding_loss(encoder_output[0], encoder_output[1])

        auto_encoder.summary()
        plot_model(auto_encoder, to_file=os.path.join(self.image_directory, 'auto_encoder.png'), show_shapes=True)
        auto_encoder.compile(loss=VariationalAutoencoder.approximate_reconstruction_loss,
                             # loss_weights=[1, -self.beta],
                             optimizer=optimizers.Adam(lr=self.learning_rate))
        return auto_encoder, encoder, decoder

    def fit_autoencoder(self):
        """
        Data normalizations.
        """
        if self.is_standardized:
            x_train = VariationalAutoencoder.standardize(self.x_train, self.x_train)
            x_test = VariationalAutoencoder.standardize(self.x_test, self.x_train)
        else:
            x_train, x_test = self.x_train, self.x_test

        auto_encoder, encoder, decoder = self.define_autoencoder()
        history = auto_encoder.fit(x_train,
                                   epochs=self.number_of_epochs,
                                   batch_size=self.batch_size,
                                   callbacks=[self.tensorboard_callback])

        print("Variational autoencoder trained.\n")
        return auto_encoder, encoder, decoder, history

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

    def save_all_models(self, autoencoder, encoder, decoder):
        """
        Save the parameters of the autoencoder, encoder, and decoder (respectively) to .h5 files.
        :param autoencoder: A Keras model, in this case an autoencoder.
        :param encoder: A Keras model, in this case an encoder.
        :param decoder: A Keras model, in this case a decoder.
        :return: None
        """
        model_directory = os.path.join(self.image_directory, 'models')
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        save_model(autoencoder, os.path.join(model_directory, 'autoencoder.h5'))
        save_model(encoder, os.path.join(model_directory, 'encoder.h5'))
        save_model(decoder, os.path.join(model_directory, 'decoder.h5'))

    def train(self):
        """
        Train the autoencoder, use its history to plot loss curves, and save the parameters of the autoencoder, encoder,
        and decoder (respectively) to .h5 files.
        :return: None
        """
        self.begin_logging()

        auto_encoder, encoder, decoder, history = self.fit_autoencoder()

        VariationalAutoencoder.plot_loss_curves(history, self.image_directory)

        self.save_all_models(auto_encoder, encoder, decoder)


vae = VariationalAutoencoder(number_of_epochs=5).train()
