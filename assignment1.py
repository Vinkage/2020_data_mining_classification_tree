import numpy as np

credit_data = np.genfromtxt('./credit_score.txt',
                            delimiter=',',
                            skip_header=True)

# "credit_data" is now a 2d NumPy array. Each rows represent a record and the
# columns represent the data attributes.
# [(22, 0, 0, 28, 1, 0)
#  (46, 0, 1, 32, 0, 0)
#  (24, 1, 1, 24, 1, 0)
#  (25, 0, 0, 27, 1, 0)
#  (29, 1, 1, 32, 0, 0)
#  (45, 1, 1, 30, 0, 1)
#  (63, 1, 1, 58, 1, 1)
#  (36, 1, 0, 52, 1, 1)
#  (23, 0, 1, 40, 0, 1)
#  (50, 1, 1, 28, 0, 1)]


class Tree():
    """
    @todo: docstring for Tree
    """
    def __init__(self, d_structure):
        """@todo: Docstring for init method.

        /d_structure/ @todo

        """
        self.d_structure = d_structure

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        print("Tree has been traversed")

    def __repr__(self):
        return str(self.d_structure)

    def drop_left(self) -> dict:
        """
        @todo: Docstring for drop_left
        """
        self.d_structure = self.d_structure["left"]

    def drop_right(self) -> dict:
        """
        @todo: Docstring for drop_right
        """
        self.d_structure = self.d_structure["right"]


tree_grow_defaults = {
    'x': credit_data[:, :5],
    'y': credit_data[:, 5],
    'n_min': 0,
    'min_leaf': 0,
    'n_feat': 5
}


def tree_grow(x=None,
              y=None,
              n_min=None,
              min_leaf=None,
              n_feat=None,
              **defaults) -> Tree:
    """
    @todo: Docstring for tree_grow
    """
    print(
        "All attribute columns in credit data (to be exhaustively searched per split):\n",
        x, "\n")
    print("Class label column in credit data:\n", y, "\n")
    print(
        f'Current node will be leaf node if (( (number of data "tuples" in child node) < {n_min=} )) \n'
    )
    print(
        f'Split will not happen if (( (number of data "tuples" potential split) < {min_leaf=} ))\n'
    )
    print(
        f"Number of features/attributes to be randomly drawn from {x=} to be considered for a split, should only be lower than {len(x[0,:])=} for random forest growing, {n_feat=}"
    )
    d_structure = {}
    return Tree(d_structure)


# Calling the function, unpacking default as argument
# print(tree_grow(**tree_grow_defaults))

tree_pred_defaults = {
    'x': credit_data[:, :5],
    'tr': tree_grow(**tree_grow_defaults)
}


def tree_pred(x=None, tr=None, **defaults) -> np.array:
    """
    @todo: Docstring for tree_pred
    """
    print("\n\n#########Tree_pred output start:\n")
    print(f"Drop a row in {x=} down the tree {tr.__repr__()}")
    # x should be a set of rows that represent cases to be dropped in the tree,
    # we iterate over the numpy rows
    y = np.array([])
    for case in x:
        # Assumes that tr is a data structure with the following form
        #
        # {"root":(split, {"left":(...), "right":(...)})
        #
        # Where the (...) is recursion of the pattern tuple(tuple, dict)
        with Tree(case, tr) as tr:
            # Unpack the root node, which is just a key value pair with the form:
            # "root": (split, dict(subtree))
            split, tr.d_structure = tr.d_structure["root"]
            # Unpack the split info we need to do the first split
            # c is the split value for the drop, and col is the attribute/feature
            # we are dropping on
            col, c = split
            while isinstance(tr.d_structure, dict):
                # The drop methods return a tuple by giving the current
                # d_structure a key that is either "left" or right":
                # def drop_left(self):
                # ...
                # return tr.d_structure["left"]
                #
                # Don't even need to check for leaf node, since the while loop
                # should do that.
                if case[col] > c:
                    split, tr.d_structure = tr.drop_right()
                elif case[col] <= c:
                    split, tr.d_structure = tr.drop_left()
                split = col, c
            # Assumes that the tr.d_structure is the leaf node that is just the
            # majority(!) class label, which is just the integer 1 or 0.
            #
            # This is also the reason we break out of the while loop, when we
            # arrive to a leaf node with the drop methods, the tr.d_structure
            # type is no longer a dict, but an int.
            y.append(tr.d_structure)
    return y


tree_pred(**tree_pred_defaults)

#
#
# Put all helper functions below this comment!


def impurity(array) -> int:
    """
    Assumes the argument array is a one dimensional vector of zeroes and ones.
    Computes the gini index impurity based on the relative frequency of ones in
    the vector.

    Example:

    >>> array=np.array([1,0,1,1,1,0,0,1,1,0,1])
    >>> array
    array([1,0,1,1,1,0,0,1,1,0,1])

    >>> impurity(array)
    0.23140495867768596
    """
    # Total labels
    n_labels = len(array)
    # Number of tuples labeled 1
    n_labels_1 = array.sum()
    # Calculate the relative frequency of ones with respect to the total labels
    rel_freq_1 = n_labels_1 / n_labels
    # Use the symmetry around the median property to also calculate the
    # relative frequency of zeroes
    rel_freq_0 = 1 - rel_freq_1
    # Multiply the frequencies to get the gini index
    gini_index = rel_freq_1 * rel_freq_0
    return gini_index


# array=np.array([1,0,1,1,1,0,0,1,1,0,1])
# print(impurity(array))
# Should give 0.23....


def bestsplit(x, y) -> int:
    """
    x = vector of num values
    y = vector of class labels ... array([{x: x is 0 or 1}]) ??

    Consider splits of type "x <= c" where "c" is the average of two consecutive
    values of x in the sorted order.

    x and y must be of the same length

    y[i] must be the class label of the i-th observation, and x[i] is the
    correspnding value of attribute x

    Example (best split on income):

    >>> bestsplit(credit_data[:,3],credit_data[:,5])
     36
    """
    # Sort all unique attribute values
    num_attr_sorted = np.sort(np.unique(x))

    # Use python vector addition to add all corresponding consecutive column
    # elements and take their average
    consec_avg_attr_splitpoints = (num_attr_sorted[0:7] +
                                   num_attr_sorted[1:8]) / 2

    # Convert array to list
    split_points = list(consec_avg_attr_splitpoints)

    # Prepare the constants for the delta impurity equation
    impurity_parent_node = impurity(y)
    n_obs_parent_node = len(y)

    # Init return list
    split_points_delta_impurities = []
    while split_points:
        # compute child nodes class vectors for the split value
        split_point = split_points.pop()
        child_node = {"l": y[x > split_point], "r": y[x <= split_point]}

        # Take the weighted average of child node impurities
        w_avg_child_impurities = (
            impurity(child_node["l"]) * len(child_node["l"]) + impurity(
                child_node["r"]) * len(child_node["r"])) / n_obs_parent_node

        # Add the used split point and delta impurity to the return list
        split_points_delta_impurities += [
            (split_point, impurity_parent_node - w_avg_child_impurities)
        ]

    # Take the maximum of the list, and unpack
    best_split, best_delta_impurity = max(split_points_delta_impurities,
                                          key=lambda x: x[1])
    return best_split


# print(bestsplit(credit_data[:,3],credit_data[:,5]))
# Should give 36
