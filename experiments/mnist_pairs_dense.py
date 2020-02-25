from models.vae_dense import DenseVAE

"""
Hypothesis: A dense variational autoencoder can learn latent representations of rotated MNIST digits in which distinct
classes are separable by submanifolds (i.e. decision boundaries that can be learned by support vector machines, 
logistic regression models, etc.).

Findings: We get great separation between digits that are topologically distinct (e.g. zeros and eights), and some
separation between digits with identical topologies (e.g. ones and sevens, sixes and nines, etc.).
"""
label_list1 = [[i, j] for i in range(10) for j in range(10) if i < j]
label_list2 = [[i] for i in range(10)]
label_list3 = [list(range(10))]

label_list = label_list2 + label_list1 + label_list3
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
