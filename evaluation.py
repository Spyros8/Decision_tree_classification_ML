import numpy as np
import decision_tree as dt


def divide_dataset(data_array, fold_size, fold=0):
    """ Divides dataset into two sets

        Divides dataset into N parts, where N is the fold_size
        Returns the nth part as the validation data where n is the fold
        Concatenates the rest of the data into one array and returns it
        as the training data

    Args:
        data_array (numpy.ndarray): The data to be split
        fold_size (int): The number of parts the data will be divided by
        fold (int): The part to be returned as validation data
    Returns:
        training_data (numpy.ndarray): Array of training data
        validation_data (numpy.ndarray): Array of validation data
    """

    # Splits the unsorted dataset into training and validation/testing sets

    # Separate the data into equal number of sets given the fold size
    splitted_data = np.split(data_array, fold_size, axis=0)
    # Take the set indexed by fold as the validation data
    validation_data = splitted_data[fold]
    # Concatenate the rest of the data into one array of training data
    if fold == 0:
        training_data = np.concatenate((splitted_data[(fold+1):]), axis=0)
    elif fold == fold_size-1:
        training_data = np.concatenate((splitted_data[:fold]), axis=0)
    else:
        training_data_1 = np.concatenate((splitted_data[:fold]), axis=0)
        training_data_2 = np.concatenate((splitted_data[(fold+1):]), axis=0)
        training_data = np.concatenate((training_data_1, training_data_2), axis=0)

    return training_data, validation_data


def validate_learning(data_array, fold_size):
    """ Test the decision_tree_learning function with n-fold validation and return confusion matrix and depth

        Splits the dataset into training and testing data. Cycles the test data and 
        returns a confusion matrix and depth averaged over n trees.
        NOTE this function was used before pruning the tree

    Args:
        data_array (numpy.ndarray): The data to be split
        fold_size (int): The number of folds the function will be tested
    Returns:
        average_confusion_matrix (numpy.ndarray): 4x4 array confusion matrix
        average_depth (float): Average depth of trees
    """

    # Validates the descision tree learning function given a data array and a fold size
    labels = np.unique(data_array[:, data_array.shape[1]-1])
    total_confusion_matrix = np.zeros((len(labels), len(labels)))
    total_depth = 0

    # separate the data into training and testing for each fold.
    for fold in range(fold_size):
        print(".", end="", flush=True)
        training_data, test_data = divide_dataset(data_array, fold_size, fold)
        # Train the tree and get its confusion matrix
        tree, depth = dt.decision_tree_learning(training_data)
        predicted_values, true_values = dt.classify_array(test_data, tree)
        confusion_matrix = dt.get_confusion_matrix(predicted_values, true_values)
        # Append its confusion matrix to the total matrix and depth to the total depth
        total_confusion_matrix += confusion_matrix
        total_depth += depth
    # Compute average by dividing total by fold size
    average_confusion_matrix = total_confusion_matrix/fold_size
    average_depth = total_depth/fold_size

    return average_confusion_matrix, average_depth


def single_validation(data_array, divider):
    """ Test the decision_tree_learning and prune with single validation and return confusion matrices and depths

        Divides the dataset into training, validation and testing data.
        Trains tree and tests it. Then prunes it and tests it again.
        Returns confusion matrices and depth of both trees

    Args:
        data_array (numpy.ndarray): The data to be split
        divider (int): The number of parts the data will be divided in. The first part is taken for testing and the second for validation
    Returns:
        confusion_matrix (numpy.ndarray): 4x4 array confusion matrix of normal tree
        depth (float): Depth of normal tree
        new_confusion_matrix (numpy.ndarray): 4x4 array confusion matrix of pruned tree
        new_depth (float): Depth of pruned tree
    """

    # Trains a tree, prunes and validates only once before and after pruning.
    training_data, test_data = divide_dataset(data_array, divider)
    training_data, validation_data = divide_dataset(training_data, divider-1)
    # Train and test tree
    tree, depth = dt.decision_tree_learning(training_data)
    predicted_values, true_values = dt.classify_array(test_data, tree)
    confusion_matrix = dt.get_confusion_matrix(predicted_values, true_values)
    # Prune and test tree
    tree = dt.prune(tree, validation_data)
    new_depth = dt.get_depth(tree)
    new_predicted_values, new_true_values = dt.classify_array(test_data, tree)
    new_confusion_matrix = dt.get_confusion_matrix(new_predicted_values, new_true_values)

    return confusion_matrix, depth, new_confusion_matrix, new_depth


def validate_pruning(data_array, fold_size):
    """ Test the decision_tree_learning and prune with nested n-fold validation and return confusion matrices and depths

        Divides the dataset into training, validation and testing data.
        Trains tree and tests it. Then prunes it and tests it again.
        Cycles through both validation and test data producing an average of n(n-1) trees
        where n = fold_size
        Returns confusion matrices and depth of both trees

    Args:
        data_array (numpy.ndarray): The data to be split
        fold_size (int): The number of folds the data will be divided and cycled through.
    Returns:
        average_confusion_matrix (numpy.ndarray): 4x4 array of average confusion matrix of normal tree
        average_depth (float): Average depth of normal tree
        average_pruned_matrix (numpy.ndarray): 4x4 array of average confusion matrix of pruned tree
        average_pruned_depth (float): Average depth of pruned tree
    """

    # This is a nested x-fold of the validate function.
    # Count the labels
    labels = np.unique(data_array[:, data_array.shape[1]-1])
    # Initialize variables
    total_confusion_matrix = np.zeros((len(labels), len(labels)))
    total_depth = 0
    total_pruned_matrix = np.zeros((len(labels), len(labels)))
    total_pruned_depth = 0
    print_counter = 1
    for fold in range(fold_size):
        # Prints for visualisation purposes
        print(f"Validating training set {print_counter}/{fold_size} ", end="", flush=True)
        # Divide data into test data and the rest
        training_visualisation_data, test_data = divide_dataset(data_array, fold_size, fold)
        # Initialize variables for current training set
        confusion_matrix = np.zeros((len(labels), len(labels)))
        training_depth = 0
        pruned_matrix = np.zeros((len(labels), len(labels)))
        pruned_depth = 0
        for validation_fold in range(fold_size-1):
            # Prints for visualisation purposes
            print(".", end="", flush=True)
            # Further divide data into training and validation data
            training_data, validation_data = divide_dataset(training_visualisation_data, fold_size-1, validation_fold)
            # train and test tree
            tree, depth = dt.decision_tree_learning(training_data)
            predicted_values, true_values = dt.classify_array(test_data, tree)
            # prune and test tree
            tree = dt.prune(tree, validation_data)
            new_depth = dt.get_depth(tree)
            # Append confusion matrices and depth of both trees
            predicted_pruned_values, true_pruned_values = dt.classify_array(test_data, tree)
            confusion_matrix += dt.get_confusion_matrix(predicted_values, true_values)
            training_depth += depth
            pruned_matrix += dt.get_confusion_matrix(predicted_pruned_values, true_pruned_values)
            pruned_depth += new_depth
        # Append the averages of confusion matrices and depths of all validation folds
        total_confusion_matrix += confusion_matrix/(fold_size-1)
        total_depth += training_depth/(fold_size-1)
        total_pruned_matrix += pruned_matrix/(fold_size-1)
        total_pruned_depth += pruned_depth/(fold_size-1)
        print_counter+=1
        print("")
    # Average matrices and depths of all training folds
    average_confusion_matrix = total_confusion_matrix/fold_size
    average_depth = total_depth/fold_size
    average_pruned_matrix = total_pruned_matrix/fold_size
    average_pruned_depth = total_pruned_depth/fold_size

    return average_confusion_matrix, average_depth, average_pruned_matrix, average_pruned_depth


def classification_metrics(confusion_matrix, depth):
    """ Calculates and prints classification metrix given the confusion matrix and depth

        Prints tree depth, confusion matrix, accuracy of tree,
        class accuracies, precisions, recalls and f1 measures

    Args:
        confusion_matrix (numpy.ndarray): 4x4 confusion matrix
        depth (float): The depth of the tree
    Returns:
        None
    """

    # Calculate and print the classification metrics from the confusion matrix and the depth
    accuracy = confusion_matrix.diagonal().sum()/confusion_matrix.sum()
    precision = confusion_matrix.diagonal()/confusion_matrix.sum(axis=0)
    recall = confusion_matrix.diagonal()/confusion_matrix.sum(axis=1)

    f1_measure = (2*precision*recall)/(precision+recall)

    print(f"Tree depth: {depth} \n\n",
          f"Confusion matrix:\n{confusion_matrix}\n\n",
          f"Accuracy:          {accuracy} \n",
          f"Class accuracies:  {confusion_matrix.diagonal()} \n",
          f"Class precisions:  {precision}\n",
          f"Class recalls:     {recall}\n" ,
          f"Class F1_measures: {f1_measure}")
