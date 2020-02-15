from tensorflow.keras.callbacks import *


def tensorboard_callback(directory, histogram_freq=1, write_graph=False, write_images=True):
    tensorboard_callback = TensorBoard(log_dir=os.path.join(self.directory, 'tensorboard_logs'),
                                       histogram_freq=histogram_freq,
                                       write_graph=write_graph,
                                       write_images=write_images)
    return tensorboard_callback


def early_stopping_callback()
self.early_stopping_callback = EarlyStopping(monitor='val_loss',
                                             min_delta=self.early_stopping_delta,
                                             patience=self.patience_limit,
                                             mode='auto',
                                             restore_best_weights=True)

self.learning_rate_callback = ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.1,
                                                patience=50,
                                                min_lr=self.learning_rate_minimum)