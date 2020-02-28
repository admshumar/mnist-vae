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

from utils import plots


class DenseVAEClassifier(VAE):
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
                 with_mixture_model=False,
                 with_logistic_regression=False,
                 with_svc=False,
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
                 encoder_activation_layer=ReLU(),
                 decoder_activation_layer=ReLU(),
                 final_activation='sigmoid',
                 ):
        model_name = 'vae_dense_with_classifier'
        super(DenseVAEClassifier, self).__init__(deep=deep,
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
                                                 with_mixture_model=with_mixture_model,
                                                 with_logistic_regression=with_logistic_regression,
                                                 with_svc=with_svc,
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
                                                 encoder_activation_layer=encoder_activation_layer,
                                                 decoder_activation_layer=decoder_activation_layer,
                                                 final_activation=final_activation,
                                                 model_name=model_name)

    def define_encoder(self):
        z = self.encoder_mnist_input
        z = Flatten()(z)
        z = Dense(self.intermediate_dimension)(z)

        if self.enable_activation:
            z = self.decoder_activation_layer(z)

        z_gaussian = Dense(self.gaussian_dimension, name="gaussian")(z)
        z = Reparametrization(name="latent_samples")(z_gaussian)
        encoder_output_layer = [z_gaussian, z]

        encoder = Model([self.encoder_gaussian, self.encoder_mnist_input], encoder_output_layer, name='encoder')
        encoder.summary()
        plot_model(encoder, to_file=os.path.join(self.image_directory, 'encoder.png'), show_shapes=True)

        return encoder, encoder_output_layer

    def define_decoder(self, encoder_output):
        decoder_gaussian_input = Input(shape=encoder_output[0].shape[1:], name='gaussian_input')
        decoder_latent_input = Input(shape=encoder_output[1].shape[1:], name='latent_input')
        x = decoder_latent_input
        gaussian = decoder_gaussian_input

        # Needed to prevent Keras from complaining that nothing was done to this tensor:
        identity_lambda = Lambda(lambda w: w, name="dec_identity_lambda")
        gaussian = identity_lambda(gaussian)

        x = Dense(self.intermediate_dimension)(x)

        if self.enable_activation:
            x = self.decoder_activation_layer(x)

        x = Dense(self.data_dimension, activation=self.final_activation)(x)

        x = Reshape((28, 28))(x)

        decoder_output = [gaussian, x]
        decoder = Model([decoder_gaussian_input, decoder_latent_input], decoder_output, name='decoder')
        decoder.summary()
        plot_model(decoder, to_file=os.path.join(self.image_directory, 'decoder.png'), show_shapes=True)

        return decoder

    def define_autoencoder(self):
        """
        Define the encoder and decoder models,
        :return:
        """
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

    def define_projection_head(self, encoder_output_layer):
        latent_input_layer = Input(shape=encoder_output_layer[1].shape[1:], name='projection_head_latent_input')
        z = Dense(latent_input_layer.shape[1], activation='sigmoid')(latent_input_layer)
        z = Dense(self.number_of_clusters, activation='softmax')(z)
        output_layer = z

        projection_head = Model(latent_input_layer, output_layer, name='projection_head')
        projection_head.summary()
        plot_model(projection_head, to_file=os.path.join(self.image_directory, 'projection_head.png'), show_shapes=True)
        return projection_head

    def get_pretrained_encoder(self, weight_directory=None):
        encoder, encoder_output = self.define_encoder()
        if weight_directory is None:
            filepath = os.path.abspath(os.path.join(self.experiment_directory, 'models', 'encoder.h5'))
        else:
            filepath = os.path.abspath(os.path.join(self.experiment_directory, '..',
                                                    weight_directory, 'models', 'encoder.h5'))
        encoder.load_weights(filepath)
        encoder.trainable = False
        return encoder, encoder_output

    def define_encoder_classifier(self, weight_directory=None, alpha=0):
        """
        Define the encoder-classifier. If weight_directory is None, then it is assumed that the current working
        directory corresponds to the model, whose weights don't yet exist. Hence there is a call to self.train().
        :param weight_directory: The hyper-parameter string corresponding to the directory in which the encoder's
        weights are found.
        :return: A Keras Model instance.
        """
        # Network layers
        if weight_directory is None:
            _, encoder_output_layer = self.define_encoder()
            _, encoder, _ = self.train()
        else:
            encoder, encoder_output_layer = self.get_pretrained_encoder(weight_directory)
        projection_head = self.define_projection_head(encoder_output_layer)

        # Network tensors
        ec_gaussian_input_layer = Input(shape=self.gaussian_shape, name='ec_gaussian_input')
        ec_mnist_input_layer = Input(shape=self.mnist_shape, name='ec_mnist_input')
        encoder_classifier_input_layer = [ec_gaussian_input_layer, ec_mnist_input_layer]
        projection_head_input_layer = encoder(encoder_classifier_input_layer)[1]
        class_probabilities = projection_head(projection_head_input_layer)
        ec_output_layer = class_probabilities # [ec_gaussian_input_layer, class_probabilities]

        # Model definition
        encoder_classifier = Model(encoder_classifier_input_layer, ec_output_layer, name='encoder_classifier')
        encoder_classifier.summary()
        plot_model(encoder, to_file=os.path.join(self.image_directory, 'encoder_classifier.png'), show_shapes=True)

        encoder_classifier.compile(optimizers.Adam(lr=self.learning_rate),
                                   loss=CategoricalCrossentropy(name='categorical_cross_entropy',
                                                                label_smoothing=alpha),
                                   metrics=[Accuracy()])
        return encoder_classifier

    def get_fit_args_encoder_classifier(self, use_gaussian_soft_labels=False):
        """
        Define a list of NumPy inputs and NumPy outputs of the Keras model. These are the actual data that flow through
        the Keras model.
        :return: A list of arguments for the fit method of the Keras model.
        """
        model_input = [self.gaussian_train, self.x_train]
        if use_gaussian_soft_labels:
            model_target = [self.assign_soft_labels()]
        else:
            model_target = [self.y_train_binary]
        return [model_input, model_target]

    def fit_encoder_classifier(self, weight_directory=None, alpha=0):
        args = self.get_fit_args_encoder_classifier()
        kwargs = self.get_fit_kwargs()
        encoder_classifier = self.define_encoder_classifier(weight_directory=weight_directory, alpha=alpha)
        print("Training encoder-classifier.")
        history = encoder_classifier.fit(*args, **kwargs)
        print("Encoder classifier trained.\n")
        return encoder_classifier, history

    def train_encoder_classifier(self, alpha=0):
        encoder_classifier, history = self.fit_encoder_classifier(alpha=alpha)
        plots.plot_loss_curves(history, self.image_directory, name="losses_encoder_classifier")
        self.save_model_weights(encoder_classifier, 'encoder_classifier')
        return encoder_classifier

    def evaluate_encoder_classifier(self, model):
        print('Evaluation')
        result = model.evaluate([self.gaussian_test, self.x_test], self.y_test_binary, batch_size=self.batch_size)
        print("Categorical Cross Entropy Loss:", result)
        return result

