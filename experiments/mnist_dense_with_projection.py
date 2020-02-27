from models.projection_head import DenseVAEClassifier
from tensorflow.keras.layers import *
import itertools

"""
What are the accuracies of logistic regression and RBF-SVM on latent representations of standard MNIST and rotated
MNIST? These will be baseline classification models against which a neural network classifier trained on soft-labels
will be benchmarked.
"""
label_list = [[i, j] for i in range(10) for j in range(10) if i < j]
all_labels = list(range(10))
rotation_list = [False, True]
alpha_list = [0, 0.1, 0.2]
beta_list = [1, 1.5, 2]
hyper_parameters = itertools.product(rotation_list, alpha_list, beta_list)

"""
Varying the rotation parameter is obvious. 
Varying alpha corresponds to training with label smoothing.
Varying beta corresponds to enforcing disentanglement.
"""
for h in hyper_parameters:
    vae = DenseVAEClassifier(number_of_epochs=2,
                             is_restricted=False,
                             restriction_labels=all_labels,
                             has_validation_set=True,
                             enable_logging=True,
                             enable_dropout=True,
                             enable_rotations=h[0],
                             with_logistic_regression=True,
                             with_mixture_model=False,
                             with_svc=True,
                             number_of_rotations=11,
                             angle_of_rotation=30,
                             enable_stochastic_gradient_descent=True,
                             encoder_activation_layer=ReLU(),
                             decoder_activation_layer=ReLU(),
                             final_activation='sigmoid',
                             learning_rate_initial=1e-2,
                             beta=h[2],
                             smoothing_alpha=h[1],
                             show=False)
    vae.train_and_evaluate_encoder_classifier()
    del vae

