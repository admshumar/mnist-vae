import matplotlib.pyplot as plt
import numpy as np
import os
from utils import operations


def plot(model_history, directory, functions, filename, plot_title, metric, x_variable, location='upper right'):
    """
    :param model_history: A dictionary of evidence_lower_bound values.
    :param directory: A string indicating the directory to which the evidence_lower_bound image is written.
    :param functions: A set of strings indicating the functions to be plotted.
    :param filename: A string indicating a filename for the plot.
    :param plot_title: A string indicating the title of the plot.
    :param metric: A string indicating the label for the vertical axis.
    :param x_variable: A string indicating the label for the horizontal axis.
    :param location: A string indicating the location of the key in the plot.
    :return: None
    """
    filepath = os.path.join(directory, filename + '.png')
    model_losses = functions.intersection(set(model_history.history.keys()))

    fig = plt.figure(dpi=200)
    for loss in model_losses:
        plt.plot(model_history.history[loss])
    plt.title(plot_title)
    plt.ylabel(metric)
    plt.xlabel(x_variable)
    plt.legend(['Train', 'Val'], loc=location)
    fig.savefig(filepath)
    plt.close(fig)


def loss(model_history, directory):
    plot(model_history, directory, {'loss', 'val_loss'}, 'loss', 'Model Loss', 'Loss', 'Epochs')


def accuracy(model_history, directory):
    plot(model_history, directory, {'acc', 'val_acc'}, 'accuracy',
         'Model Accuracy', 'Accuracy', 'Epochs', location='lower right')

