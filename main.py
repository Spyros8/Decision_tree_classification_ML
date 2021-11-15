""" Introduction to Machine Learning
    Coursework 1: Decision Trees
    Group Members: Christos Margadji, Marios Theodorou, Spyros Plousiou
"""

import numpy as np
from load_data import load_data_as_array
from decision_tree import decision_tree_learning
from tree_visualisation import visualise_decision_tree
from evaluation import validate_pruning, classification_metrics


#######################################################################
######################## Main program #################################

def main():
    seed = 7777
    np.random.seed(seed)

    clean_data_array = load_data_as_array("wifi_db/clean_dataset.txt")
    noisy_data_array = load_data_as_array("wifi_db/noisy_dataset.txt")

    # BONUS: Tree visualisation
    (d_tree, depth) = decision_tree_learning(clean_data_array)
    visualise_decision_tree(d_tree, depth, 'decision_tree_clean_data.png')

    # Random shuffle of data for separating into training, validation and testing
    np.random.shuffle(clean_data_array)
    np.random.shuffle(noisy_data_array)

    # Fold size for training and validation
    fold_size = 10

    # Train and test tree, then prune and test tree for clean data
    print("Training and evaluating clean data")
    clean_matrix, clean_depth, pruned_matrix, pruned_depth = validate_pruning(clean_data_array, fold_size)
    print("\nBefore pruning: ")
    classification_metrics(clean_matrix, clean_depth)
    print("\nAfter pruning: ")
    classification_metrics(pruned_matrix, pruned_depth)

    # Train and test tree, then prune and test tree for noisy data
    # NOTE this takes about 10 minutes for a nested 10-fold cross validation
    print("Training and evaluating noisy data")
    clean_matrix, clean_depth, pruned_matrix, pruned_depth = validate_pruning(noisy_data_array, fold_size)
    print("\nBefore pruning: ")
    classification_metrics(clean_matrix, clean_depth)
    print("\nAfter pruning: ")
    classification_metrics(pruned_matrix, pruned_depth)


if __name__ == "__main__":
    main()

