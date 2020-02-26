from models.vae_dense import DenseVAE
from tensorflow.keras.layers import *

"""
Exploratory Data Analysis: What representations of rotated MNIST are learned by a VAE?
"""
# label_list1 = [[i, j] for i in range(10) for j in range(10) if i < j]
# label_list2 = [list(range(10))]
# label_list = label_list1 + label_list2
label_list = [list(range(10))]
rotation_list = [False]
for labels in label_list:
    for rotations in rotation_list:
        vae = DenseVAE(number_of_epochs=50,
                       is_restricted=True,
                       restriction_labels=labels,
                       has_validation_set=False,
                       enable_logging=True,
                       enable_dropout=True,
                       enable_rotations=rotations,
                       with_logistic_regression=True,
                       with_mixture_model=True,
                       with_svc=True,
                       number_of_rotations=11,
                       angle_of_rotation=30,
                       enable_stochastic_gradient_descent=True,
                       encoder_activation_layer=ReLU(),
                       decoder_activation_layer=ReLU(),
                       final_activation='sigmoid',
                       learning_rate_initial=1e-2,
                       beta=2)
        vae.train()
        del vae
