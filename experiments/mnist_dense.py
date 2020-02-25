from models.vae_dense import DenseVAE

"""
Exploratory Data Analysis: What representations of rotated MNIST are learned by a VAE?
"""
label_list = [[i, j] for i in range(10) for j in range(10) if i < j]
label_list = [[i] for i in range(10)]
label_list = list(range(10))
vae = DenseVAE(number_of_epochs=100,
               is_restricted=True,
               restriction_labels=label_list,
               enable_logging=True,
               enable_dropout=True,
               enable_rotations=False,
               number_of_rotations=11,
               angle_of_rotation=30,
               enable_stochastic_gradient_descent=True,
               encoder_activation='relu',
               decoder_activation='relu',
               final_activation='sigmoid',
               learning_rate_initial=1e-2,
               beta=2)
vae.train()
del vae
