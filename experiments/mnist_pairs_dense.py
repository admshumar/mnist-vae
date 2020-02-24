from models.vae_dense import DenseVAE

"""

"""
label_list = [[i, j] for i in range(10) for j in range(10) if i < j]
for k in range(len(label_list)):
    vae = DenseVAE(number_of_epochs=100,
                   is_restricted=True,
                   restriction_labels=label_list[k],
                   enable_logging=True,
                   enable_dropout=True,
                   enable_rotations=True,
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
