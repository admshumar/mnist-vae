import os
import re


class DirectoryCounter:
    """
    A class that counts the number of directories with a matching hyperparameter string.
    """
    @classmethod
    def get_root_directory(cls):
        """
        Make a root directory.
        :return: A string corresponding to the root directory.
        """
        directory = os.path.dirname(os.path.realpath(__file__))
        directory = os.path.abspath(os.path.join(directory, '..', 'data'))
        return directory

    @classmethod
    def make_output_directory(cls, hyper_parameter_string, model_name):
        """
        Make an output directory indexed by a set of hyperparameters.
        :return: A string corresponding to the output directory.
        """
        output_directory = os.path.abspath(os.path.join(os.getcwd(),
                                                        '..',
                                                        'data',
                                                        'experiments',
                                                        model_name,
                                                        hyper_parameter_string))
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        return output_directory

    def __init__(self, hyperparameter_string):
        self.data_directory = DirectoryCounter.get_root_directory()
        self.regex = re.compile(hyperparameter_string)

    def count(self):
        """
        Count the number of directories with a hyperparameter string that matches hyperparameter_string.
        :return: An integer that indicates the number of directories with a matching hyperparameter string.
        """
        directory_count = 0
        for _, dirs, _ in os.walk(self.data_directory):
            for directory in dirs:
                if self.regex.match(directory):
                    directory_count += 1
        return directory_count
