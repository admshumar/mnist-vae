from utils.augmenters import MNISTRotator

classes = list(range(10))
for c in classes:
    for dir in ['train', 'test']:
        rotator = MNISTRotator([c], 11, 30, partition=dir)
        rotator.augment()
