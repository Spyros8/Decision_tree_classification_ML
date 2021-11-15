import numpy as np

# Functions for loading and initial handling of data.
def load_data_as_array(filename):
    """ Loads and turns list data into numpay array

    Args:
        filename (str): Directory of .txt file to be input

    Returns:
        data_array (numpy.ndarray): Array of input data
    """
    data = load_data_as_list(filename)
    return np.array(data)


def load_data_as_list(filename):
    """ Loads data in list format [] from filename.

    Loads data, processes and separates them, before adding to the list. 

    Args:
        filename (str): Directory of .txt file to be input

    Returns:
        generated_list (list): List of input data
    """

    generated_list = list()
    with open(filename) as data:
        for line in data:
            list_line = line.strip("\n").replace("\t"," ").split(" ")
            temp = []
            for element in list_line:
                temp.append(int(float(element)))    
            generated_list.append(temp)
    return generated_list