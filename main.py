from models.mnist_cnn_classifier import MNISTCNNClassifier
import argparse

beta = 1
#for beta in [1.2, 1.4, 1.6, 1.8, 2]:
classifier = MNISTCNNClassifier(number_of_epochs=100,
                                is_restricted=True,
                                restriction_labels=[6, 9],
                                has_validation_set=True,
                                validation_size=0.5,
                                enable_early_stopping=True,
                                early_stopping_delta=0.01,
                                early_stopping_patience=10,
                                enable_logging=True,
                                enable_dropout=True,
                                enable_batch_normalization=False,
                                enable_rotations=True,
                                with_logistic_regression=False,
                                with_mixture_model=False,
                                with_svc=False,
                                number_of_rotations=11,
                                angle_of_rotation=30,
                                enable_stochastic_gradient_descent=True,
                                final_activation='sigmoid',
                                learning_rate_initial=1e-3,
                                smoothing_alpha=0.1,
                                beta=beta,
                                show=False)

classifier.train_classifier()
del classifier
