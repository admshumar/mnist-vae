from tensorflow.keras.callbacks import *
import os


def tensorboard_callback(directory, histogram_freq=1, write_graph=False, write_images=True):
    callback = TensorBoard(log_dir=os.path.join(directory, 'tensorboard_logs'), histogram_freq=histogram_freq,
                           write_graph=write_graph, write_images=write_images)
    return callback


def early_stopping_callback(early_stopping_delta, patience_limit):
    callback = EarlyStopping(monitor='val_loss', min_delta=early_stopping_delta, patience=patience_limit, mode='auto',
                             restore_best_weights=True)
    return callback


def learning_rate_callback(learning_rate_minimum):
    callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50, min_lr=learning_rate_minimum)
    return callback
