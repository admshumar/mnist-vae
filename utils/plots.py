import matplotlib.pyplot as plt
import numpy as np
import os
from utils import operations


def plot_loss_curves(model_history, directory):
    """
    Plot loss curves for a Keras model using MatPlotLib. (NOTE: This plots only *after* training completes. It would
    be nice to plot concurrently with model training just in case something goes wrong, otherwise you'll get no
    loss curves.)
    :param model_history: A dictionary of evidence_lower_bound values.
    :param directory: A string indicating the directory to which the evidence_lower_bound image is written.
    :return: None
    """
    filepath = os.path.join(directory, 'losses.png')
    model_losses = {'loss', 'val_loss'}.intersection(set(model_history.history.keys()))

    fig = plt.figure(dpi=200)
    for loss in model_losses:
        plt.plot(model_history.history[loss])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    fig.savefig(filepath)
    plt.close(fig)
