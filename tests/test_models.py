import unittest
import models.mnist_cnn_classifier as nn

class TestModelInstantiation(unittest.TestCase):

    def test_unrotated_label_shape(self):
        model = nn.MNISTCNNClassifier(enable_rotations=False)
        self.assertEqual(len(model.y_train.shape), 1)

    def test_rotated_label_shape(self):
        model = nn.MNISTCNNClassifier(enable_rotations=True)
        self.assertEqual(len(model.y_train.shape), 1)

    def test_unrotated_binarized_label_shape(self):
        model = nn.MNISTCNNClassifier(enable_rotations=False)
        self.assertEqual(model.y_train_binary.shape, (len(model.y_train), model.number_of_clusters))

    def test_rotated_binarized_label_shape(self):
        model = nn.MNISTCNNClassifier(enable_rotations=True)
        self.assertEqual(model.y_train_binary.shape, (len(model.y_train), model.number_of_clusters))