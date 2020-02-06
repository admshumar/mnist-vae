import os

hyper_parameter_string = "foo"

def make_output_directory(hyper_parameter_string):
    """
    Make an output directory indexed by a set of hyperparameters.
    :return: A string corresponding to the output directory.
    """
    base_output_directory = os.getcwd().split('/')
    base_output_directory = base_output_directory[:-1]
    base_output_directory.append('data')
    base_output_directory = '/'.join(base_output_directory)

    better_dir = os.path.abspath(os.path.join(os.getcwd(), '..', hyper_parameter_string))
    print(better_dir)

make_output_directory(hyper_parameter_string)