from models.mnist_cnn_classifier import MNISTCNNClassifier

classifier = MNISTCNNClassifier(number_of_epochs=100,
                                is_restricted=False,
                                restriction_labels=[0],
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
                                beta=1,
                                smoothing_alpha=0,
                                show=False)

print(classifier.y_train_binary.shape)
del classifier
