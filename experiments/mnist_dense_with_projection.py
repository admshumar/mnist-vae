from models.vae_dense_with_projection import DenseVAEClassifier
from tensorflow.keras.layers import *

"""
Exploratory Data Analysis: What representations of rotated MNIST are learned by a VAE?
"""
# label_list1 = [[i, j] for i in range(10) for j in range(10) if i < j]
# label_list2 = [list(range(10))]
# label_list = label_list1 + label_list2
label_list = [list(range(10))]
rotation_list = [False]
alpha_list = [0, 0.2, 0.4, 0.6, 0.8]
vae = DenseVAEClassifier(number_of_epochs=50,
                         is_restricted=False,
                         restriction_labels=[0, 1, 5, 6, 8],
                         has_validation_set=False,
                         enable_logging=True,
                         enable_dropout=True,
                         enable_rotations=False,
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
                         beta=1,
                         smoothing_alpha=0,
                         show=False)

vae.train_and_evaluate_encoder_classifier()
#vae.evaluate_trained_encoder_classifier("10_128_0.01_relu_relu_True_True_0.5_0.0001_2_1_2_mnist_restricted_0,1,5,6,8_beta_2_x01")
pass
"""
for labels in label_list:
    for rotations in rotation_list:
        vae = DenseVAEClassifier(number_of_epochs=50,
                                 is_restricted=True,
                                 restriction_labels=labels,
                                 has_validation_set=False,
                                 enable_logging=False,
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
"""
