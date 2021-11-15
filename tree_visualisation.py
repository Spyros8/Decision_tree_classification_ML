from matplotlib import pyplot as plt


def visualise_decision_tree(decision_tree, depth, path_name='decision_tree_clean_data.png'):
    """ Plot and visualise decision tree and save the figure.

    Args:
        decision_tree (dict): Trained decision tree
        depth (int): Depth of tree
        path_name (str): Path to save figure

    Returns:
        None
    """

    # Initial fig contains 1 row, 1 column, and it forms 1 fig
    FIG_CONFIG = 111
    fig = plt.figure(figsize=(20, 10), facecolor='white')
    fig.clf()
    ax_properties = dict(xticks=[], yticks=[])
    visualise_decision_tree.ax1 = plt.subplot(FIG_CONFIG, frameon=False, **ax_properties)

    y_difference = 1/depth
    plot_tree(decision_tree, 0.0, 1.0, 1.0, y_difference, depth)
    plt.show()
    fig.savefig(path_name)


def plot_tree(node, x_min, x_max, y_value, y_difference, size, multiplier=1.000):
    """ Plot decision tree recursively.

    Args:
        node (dict): decision tree node
        x_min (float): The minimum x value to calculate the plot position
        x_max (float): The maximum x value to calculate the plot position
        y_value (float): The y value of the plot position
        y_difference (float): The difference to be added to the y_value for the next node
        size (float): The size of the bounding box of the node to be plotted
        multiplier (float): For shifting the plot position of deeper nodes for better visualisation

    Returns:
        None
    """

    centre = x_min+(x_max-x_min)/2
    centre = centre*multiplier
    mid_point_to_centre = (centre-x_min)/2

    if node["leaf"]:
        text = "leaf: "+str(node["label"])
        plot_node(text, (centre, y_value), (centre, y_value), True, size, node["label"])
    else:
        text = "[Wifi "+str(node["attribute"]+1)+"< "+str(node["value"])+"]"
        plot_node(text, (centre, y_value), (centre-mid_point_to_centre, y_value-y_difference), False, size, 0)
        plot_text_on_arrow((centre-mid_point_to_centre, y_value-y_difference), (centre, y_value), "yes")
        plot_tree(node["left"], x_min, centre, y_value-y_difference, y_difference, size-0.5, multiplier-0.001)
        plot_node(text, (centre, y_value), (centre+mid_point_to_centre, y_value-y_difference), False, size, 0)
        plot_text_on_arrow((centre+mid_point_to_centre, y_value-y_difference), (centre, y_value), "no")
        plot_tree(node["right"], centre, x_max, y_value-y_difference, y_difference, size-0.5, multiplier+0.001)


def plot_node(text, text_coordinate, arrow_coordinate, is_leaf, size, colour_index=0):
    """ Plots a node in the decision tree.

    Args:
        text (str): The text to write
        text_coordinate (tuple): (x, y) coordinates to plot the text
        arrow_coordinate (tuple): (x, y) coordinates to plot the arrow
        is_leaf (bool): checks if the node is a leaf
        size (float): The size of the bounding box of the text
        colour_index (int): The index of the colour list for colouring the nodes and leaves, based on room classifications.

    Returns:
        None
    """

    colour_list = ["1", "g", "r", "c", "y"]
    if is_leaf:
        arrow_bbox = dict(boxstyle="round", fc=colour_list[colour_index], alpha=0.5)
        visualise_decision_tree.ax1.annotate(text, textcoords='axes fraction', 
                                             xy=arrow_coordinate, xycoords='axes fraction',
                                             va='bottom', ha='center', bbox=arrow_bbox, size=size)
    else:
        arrow_bbox = dict(boxstyle="round", fc=colour_list[colour_index], alpha=0.5)
        arrow_args = dict(arrowstyle="->")
        visualise_decision_tree.ax1.annotate(text, xytext=text_coordinate, textcoords='axes fraction', 
                                             xy=arrow_coordinate, xycoords='axes fraction',
                                             va='bottom', ha='center', bbox=arrow_bbox, arrowprops=arrow_args, size=size)


#Function to define where to plot arrows
def plot_text_on_arrow(coordinate_child, coordinate_parent, text):
    """ Plot text on arrow connecting parent to child node.

    Get the mid_point between a parent and a child node and add text to it.

    Args:
        coordinate_child (list): The list of coordinates of child nodes
        coordinate_parent (list): The list of coordinates of parent nodes
        text (str): Text displayed on arrow 'yes' or 'no'

    Returns:
        None
    """
    x_middle = (coordinate_parent[0] - coordinate_child[0]) / 2 + coordinate_child[0]
    y_middle = (coordinate_parent[1] - coordinate_child[1]) / 2 + coordinate_child[1]
    visualise_decision_tree.ax1.text(x_middle, y_middle, text)
