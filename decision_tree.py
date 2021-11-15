import numpy as np
from copy import deepcopy


# Training functions
def decision_tree_learning(training_dataset, depth=0):
    """ Trains a decision tree recursively given a training dataset. Returns a trained tree and it's depth. 
        NOTE depth variable must be zero at the beginning
    
        Nodes contain the following keys:
            {"attribute": wifi_split (int),
             "value": split_value (float),
             "left": l_node (dict),
             "right": r_node (dict),
             "leaf": is_leaf (bool),
             "depth": depth (int)}
        
        Leaves contain the following keys:
            {"leaf": is_leaf (bool),
             "label": room_label (int),
             "label_counts": labels_counted_in_training (int),
             "depth": depth (int)}

    Args:
        training_dataset (numpy.ndarray): (X, 8) numpy array of wifi signal rows and room label
        depth (int): The depth of the current node

    Returns:
        node (dict): Tree node or leaf containing the attributes
        depth (int): The maximum depth of the tree
    """

    labels, label_counts = find_label_counts(training_dataset)
    if len(labels) == 1: # When only one label remains in subset, the recursion must terminate.
        leaf = {"leaf": True, "label": labels[0], "label_counts": label_counts[0], "depth": depth}
        return leaf, depth
    else: # keep splitting and training.
        # Sort data for all wifi signals and find split with max gain
        sorted_dataset = sort_all_data(training_dataset)
        split, split_value = find_split(sorted_dataset)
        # Split data and train next branches of tree
        left_data, right_data = get_left_and_right_data(training_dataset, split)
        l_node, l_depth = decision_tree_learning(left_data, depth+1)
        r_node, r_depth = decision_tree_learning(right_data, depth+1)
        # Set the left and right nodes into the current node
        node = {"attribute": split[0], "value": split_value, "left": l_node, "right": r_node, "leaf": False, "depth": depth}
        return (node, np.max([l_depth, r_depth]))


def sort_all_data(data_array):
    """ Sorts data for each wifi signal and returns with classification.

    Sorts for each signal and pairs each signal value with its classification.

    e.g. signal 1 is: [[-74   1]
                       [-73   1]
                       [-73   1]
                       ...
                       [-10   2]
                       [-10   2]
                       [-10   2]]

    where each row contains [signal  classification]

    sorted data is: [signal_1
                     signal_2
                     ...
                     signal_7]

    Args:
        data (numpy.ndarray): Data in ndarray form

    Returns:
        sorted_data (numpy.ndarray): (7, 2000, 2) numpy array of the sorted data with classifications 
    """

    sorted_data_list = []
    # Sorts for each wifi signal and returns the sorted indices
    sorted_wifi_indices = np.argsort(data_array, axis = 0)
    # For each wifi (1 to 7)
    for wifi in range(data_array.shape[1]-1):
        # Uses idices to get the signal and it's corresponding classification 
        sorted_wifi = data_array[sorted_wifi_indices[:, wifi], wifi]
        classifications = data_array[sorted_wifi_indices[:, wifi], data_array.shape[1]-1]
        # Appends the signals and their classifications to a list
        sorted_classifications = np.concatenate((sorted_wifi[np.newaxis, :].T, classifications[np.newaxis, :].T), axis=1)
        sorted_data_list.append(sorted_classifications)

    sorted_data = np.asarray(sorted_data_list)
    return sorted_data


def find_split(sorted_data):
    """ Finds the split of highest gain in the sorted data

    Args:
        sorted_data (numpy.ndarray): (7, X, 2) dataset sorted for each wifi signal with its classifications.

    Returns:
        split (tuple): (1, 2) The wifi and the index where the split happens
        split_value (float): The value where the split happens
    """

    # Get all the split indexes of the sorted data
    split_indexes, split_points_values = find_split_indexes(sorted_data)
    gains = []
    saved_splitpoints = []
    split_values = []
    # For each split, split the data into left and right sets, and find the gain
    for spl in range(split_indexes[1].shape[0]):
        # split_indexes[1][spl] requires +1 as the split points to one value lower where the data must be split
        data_left = sorted_data[split_indexes[0][spl], :split_indexes[1][spl]+1]
        data_right = sorted_data[split_indexes[0][spl], split_indexes[1][spl]+1:]
        # Find gain of split
        gain = find_gain(sorted_data[split_indexes[0][spl]], data_left, data_right)
        gains.append(gain)
        saved_splitpoints.append(np.array([split_indexes[0][spl], split_indexes[1][spl]]))
        split_values.append(split_points_values[split_indexes[0][spl], split_indexes[1][spl]])

    # Find the split where the maximum gain happens
    gains_array = np.asarray(gains)
    max_gain_index = gains_array.argmax()
    split = saved_splitpoints[max_gain_index]
    split[1] += 1
    split_value = split_values[max_gain_index]

    return split, split_value


def find_split_indexes(sorted_data):
    """ Indexes the sorted data at its split values and returns the list of split indexes and a list of its corresponding values.

    Finds indices of all splitpoints possible in the sorted dataset

    split_indexes is a 2 x X array where the first row represents the wifi signal
    and the second row represents the point where the data would be split into two
    datasets. The split point is the last index of the first dataset.
    e.g. signal_where_split_happens = split_indexes[0, a]
         index_where_split_happens = split_indexes[1, a]

    To split a dataset, the data would be sorted w.r.t. the wifi signal and the split
    would happen at split_indexes[1, a]+1
    e.g. left_data = data[start:split_indexes[1, a]+1], right_data = data[split_indexes[1, a]+1:end]

    split_points_values is a 7 x X list where each row represents the wifi signal
    and the column represents the value of the split, which is the midpoint between
    the two values where the split happens

    Args:
        sorted_data (numpy.ndarray): (7, X, 2) numpy array of the sorted data with classifications

    Returns:
        split_indexes (numpy.ndarray): (2, X) 2D Array of row indexes and column indexes
        split_points_values (numpy.ndarray): (7, X) Array of split values at each wifi signal
    """

    # Find the splits by subtracting array[x+1]-array[x] to find the difference between points.
    # Only subtracts the wifi signals, not their classifications
    # If the wifi signals are the same, the difference is zero and now split happens
    splits = (sorted_data[:,1:,0] - sorted_data[:,:sorted_data.shape[1]-1,0]) / 2
    # Need to append an array of zeros as the splits is now one row shorter
    zero_array =  np.zeros(splits.shape[0])[:, np.newaxis]
    split_points_values = sorted_data[:, :, 0] + np.concatenate((splits, zero_array), axis = 1)
    # Index where splits will happen (i.e. at nonzero differences)
    split_indexes = np.nonzero(splits)

    return split_indexes, split_points_values


def find_gain(data_all, data_left, data_right):
    """ Finds the gain given a dataset and its left and right datasets after it has been split.

        gain = entropy(data_all) - remainder(data_left, data_right)

    Args:
        data_all (numpy.ndarray): (X, Y) numpy array of X rows of signals and classifications. The last column in Y is the classifications
        data_left (numpy.ndarray): (A, Y) numpy array of X rows of signals and classifications. The last column in Y is the classifications
        data_right (numpy.ndarray): (B, Y) numpy array of X rows of signals and classifications. The last column in Y is the classifications
    
    NOTE X = A + B

    Returns:
        gain (int): The gain of the dataset and its split
    """

    gain = find_entropy(data_all) - find_remainder(data_left, data_right)
    return gain


def find_label_counts(data):
    """ Finds the labels in the data and counts each of them.

    Args:
        data (numpy.ndarray): (X, Y) numpy array of X rows of signals and classifications. The last column in Y is the classifications

    Returns:
        labels (numpy.ndarray): Room labels e.g. [1, 2, 3, 4] (can be less depending on the data)
        label_counts (numpy.ndarray): Corresponding count of each label
    """

    labels, label_counts = np.unique(data[:, data.shape[1]-1], return_counts=True)
    return labels, label_counts


def find_entropy(data):
    """ Finds the entropy of a set of data.

        entropy = -sum(pk*log2(pk))
        where pk is an numpy array of the counts of each label divided by the total counts

    Args:
        data (numpy.ndarray): (X, Y) numpy array of X rows of signals and classifications. The last column in Y is the classifications

    Returns:
        entropy_sum (int): The entropy of the data
    """

    labels, label_counts = find_label_counts(data)
    total_samples = np.sum(label_counts)
    p_counts = label_counts/total_samples
    entropy_sum = -np.sum(p_counts * np.log2(p_counts))
    return entropy_sum


def find_remainder(data_left, data_right):
    """ Finds the remainder given a left and right set of data.

        remainder = (num_of_left_labels/total_labels)*entropy_left + (num_of_right_labels/total_labels)*entropy_right

    Args:
        data_left (numpy.ndarray): (X, Y) numpy array of X rows of signals and classifications. The last column in Y is the classifications
        data_right (numpy.ndarray): (X, Y) numpy array of X rows of signals and classifications. The last column in Y is the classifications

    Returns:
        remainder (int): The remainder of the two datasets
    """

    left_labels, left_counts = find_label_counts(data_left)
    right_labels, right_counts = find_label_counts(data_right)
    left_samples = np.sum(left_counts)
    right_samples = np.sum(right_counts)
    left_remainder = (left_samples/(left_samples + right_samples))*find_entropy(data_left)
    right_remainder = (right_samples/(left_samples + right_samples))*find_entropy(data_right)
    remainder = left_remainder + right_remainder
    return remainder


def get_left_and_right_data(data_array, split):
    """ Splits a dataset to left and right datasets given a split index

    Args:
        data_array (numpy.ndarray): (X, 8) numpy array of wifi signal rows and room label
        split (tuple): (1, 2) The wifi and the index where the split happens

    Returns:
        left_data (numpy.ndarray): (0:split_point, 8) The left dataset after the split
        right_data (numpy.ndarray): (split_point:end, 8) The right dataset after the split
    """
    data_array = data_array.T
    # Sort the data with the wifi signal where the split must happen
    sorted_wifi_indices = np.argsort(data_array[split[0]], axis = 0)
    sorted_data = data_array[:, sorted_wifi_indices]
    # Split the data
    left_data = sorted_data[:, :split[1]]
    right_data = sorted_data[:, split[1]:]
    return left_data.T, right_data.T


# Pruning functions.
def prune(tree, validation_data):
    """ Prunes a decision tree repeatedly until no further pruning is possible. Returns pruned tree.

    Args:
        tree (dict): Trained decision tree
        validation_data (numpy.ndarray): (X, 8) numpy array of wifi signal rows and room label for pruning the tree

    Returns:
        tree (dict): Pruned decision tree
    """
    # Parses tree several times until the tree cannot be pruned anymore
    original_tree = deepcopy(tree)
    tree = prune_tree(tree, tree, validation_data)
    while not (original_tree == tree):
        original_tree = deepcopy(tree)
        tree = prune_tree(tree, tree, validation_data)
    return tree


def prune_tree(tree, node, validation_data):
    """ Prunes a node in the decision tree recursively for one iteration. Returns pruned node.

    Args:
        tree (dict): Trained decision tree. Used to calculate validation error for pruning.
        node (dict): The node to be pruned
        validation_data (numpy.ndarray): (X, 8) numpy array of wifi signal rows and room label for pruning the node

    Returns:
        node (dict): Pruned node
    """

    # Goes once through the whole node and prunes it
    original_accuracy = get_tree_accuracy(tree, validation_data)
    # Return single leaf nodes
    if node["leaf"]:
        return node
    # Check if left and right branches contain leafs
    if node["left"]["leaf"] and node["right"]["leaf"]:
        # Set node as leaf based on which of the two has the most classifications. Set the depth as the depth of the current node
        if node["left"]["label_counts"] > node["right"]["label_counts"]:
            depth = node["depth"]
            node = node["left"]
            node["depth"] = depth
        else:
            depth = node["depth"]
            node = node["right"]
            node["depth"] = depth
        return node
    
    # Deepcopy left and right nodes before pruning to keep originals
    old_left = deepcopy(node["left"])
    old_right = deepcopy(node["right"])
    # Prune left and right node recursively
    node["left"] = prune_tree(tree, node["left"], validation_data)
    node["right"] = prune_tree(tree, node["right"], validation_data)

    new_accuracy = get_tree_accuracy(tree, validation_data)
    # Check if the pruned tree performs worse than original tree and return the better performing node
    if new_accuracy < original_accuracy:
        node["left"] = old_left
        node["right"] = old_right
        return node
    else:
        return node


# Classification functions
def classify(data, node):
    """ Classifies a single row of wifi signals 1 to 7.

    Args:
        data (numpy.ndarray): (1, 8) numpy array of a single room classification containig wifi signals 1-7 and the room label

    Returns:
        label (int): Room label e.g. (1, 2, 3 or 4)
    """

    # Return the leaf label
    if node["leaf"]:
        label = node.get("label")
        return label
    else:
        # If the wifi signal value is less than the node value, take the left branch, otherwise take the right
        if data[node["attribute"]] < node["value"]:
            return classify(data, node["left"])
        else:
            return classify(data, node["right"])


def classify_array(dataset, tree):
    """ Classifies an array of wifi signal rows containing signals 1 to 7.

    Args:
        dataset (numpy.ndarray): (X, 8) numpy array of wifi signal rows and room label
    Returns:
        classifications (numpy.ndarray): X size array of predicted room labels (1, 2, 3 or 4)
        true_classifications (numpy.ndarray): The last column of dataset containing X rows of the real room labels
    """

    classifications = []
    for data in dataset:
        label = classify(data, tree)
        classifications.append(label)
    return np.asarray(classifications), dataset[:, dataset.shape[1]-1]


# Metrics.
def get_confusion_matrix(predicted_labels, true_labels):
    """Returns confusion matrix

    Args:
        predicted_labels (numpy.ndarray): Room labels predicted by tree
        true_labels (numpy.ndarray): True room labels

    Returns:
        confusion_matrix (numpy.ndarray): 4x4 matrix
    """

    labels, label_counts = np.unique(true_labels, return_counts=True)
    confusion_matrix = []
    for label in labels:
        # Find the indices of the true labels
        label_indices = np.where(true_labels == label)[0]
        matrix_row = []
        # Iterate through the 4 labels and count the number of classifications
        for label_check in labels:
            # Use the indices of the true labels to count good and bad classifications
            value = np.count_nonzero(predicted_labels[label_indices] == label_check)
            matrix_row.append(value)
        confusion_matrix.append(np.asarray(matrix_row))

    return np.asarray(confusion_matrix)


def get_depth(tree, depth=0):
    """ Returns depth of tree.
        NOTE depth variable must be zero at the beginning

    Args:
        tree (dict): Trained decision tree.
        depth (int): The depth of the current node

    Returns:
        depth (int): The maximum depth of the tree
    """

    # Go throgh all the nodes and return the maximum depth
    if tree["leaf"] and tree["depth"] > depth:
        return tree["depth"]
    else:
        l_depth = get_depth(tree["left"], depth)
        r_depth = get_depth(tree["right"], depth)
        return np.max([l_depth, r_depth])


def get_tree_accuracy(tree, data):
    """Returns accuracy of decision tree

    Args:
        tree (dict): Decision tree
        data (numpy.ndarray): Testing or validation data

    Returns:
        accuracy (float): Tree accuracy
    """

    predicted_values, true_values = classify_array(data, tree)
    confusion_matrix = get_confusion_matrix(predicted_values, true_values)
    accuracy = confusion_matrix.diagonal().sum()/confusion_matrix.sum()

    return accuracy