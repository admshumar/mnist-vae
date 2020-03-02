import unittest
import utils.loaders as loaders
import utils.augmenters as augmenters


class TestDataLoaders(unittest.TestCase):

    def test_data_input_shape(self):
        list_of_digits = [0]
        number_of_rotations = 11
        angle_of_rotation = 30
        loader = loaders.MNISTLoader('train')
        data = loader.load(list_of_digits, number_of_rotations, angle_of_rotation)
        self.assertEqual(len(data.shape), 3)

    def test_label_input_shape(self):
        list_of_digits = [0]
        number_of_rotations = 11
        angle_of_rotation = 30
        loader = loaders.MNISTLoader('train')
        labels = loader.load(list_of_digits, number_of_rotations, angle_of_rotation, label=True)
        self.assertEqual(len(labels.shape), 1)


class TestDataAugmentation(unittest.TestCase):

    def test_data_input_shape(self):
        list_of_digits = [0]
        number_of_rotations = 11
        angle_of_rotation = 30
        partition = 'train'
        loader = loaders.MNISTLoader(partition)
        images = loader.load(list_of_digits, number_of_rotations, angle_of_rotation)
        labels = loader.load(list_of_digits, number_of_rotations, angle_of_rotation, label=True)
        augmented_data, _ = augmenters.Rotator(images,
                                               labels,
                                               number_of_rotations=number_of_rotations,
                                               angle_of_rotation=angle_of_rotation,
                                               partition=partition).append_rotated_images()
        self.assertEqual(len(augmented_data.shape), 3)

    def test_label_input_shape(self):
        list_of_digits = [0]
        number_of_rotations = 11
        angle_of_rotation = 30
        partition = 'train'
        loader = loaders.MNISTLoader(partition)
        images = loader.load(list_of_digits, number_of_rotations, angle_of_rotation)
        labels = loader.load(list_of_digits, number_of_rotations, angle_of_rotation, label=True)
        _, augmented_labels = augmenters.Rotator(images,
                                                 labels,
                                                 number_of_rotations=number_of_rotations,
                                                 angle_of_rotation=angle_of_rotation,
                                                 partition=partition).append_rotated_images()
        self.assertEqual(len(augmented_labels.shape), 1)

if __name__ == '__main__':
    unittest.main()
