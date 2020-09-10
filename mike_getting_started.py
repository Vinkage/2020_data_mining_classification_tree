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

def impurity(array) -> None:
    """
    @todo: Docstring for 
    """
    # Assumes array is a 1 dimensional array, the slice is actually arbitrary i think
    n_observations = len(array[0:])
    # print('Total observations is the cardinality of the vector containing all class labels:', n_observations)

    n_labels_1 = array[0:].sum()
    # print('Since we are working with binary class labels the amount of "1" labels equals the sum of the class label vector:', n_labels_1)

    # Calculate the relative frequency of label 1 with respect to the total sample size
    rel_freq_1 = n_labels_1 / n_observations

    # Use the symmetry property to also calculate the relative frequency of zeroes
    rel_freq_0 = 1 - rel_freq_1
    # print('\nThe rel. freq. of 1: ', rel_freq_1)
    # print('\nThe rel. freq. of 0: ', rel_freq_0)
    gini_index = rel_freq_1 * rel_freq_0
    # print('\nThe gini index: ', gini_index)
    # pass
    return gini_index


# impurity(test_array)

# x = vector of num values
# y = vector of class labels ... array([0,1]) ??
#
# x and y must be of the same length
#
# y[i] must be the class label of the i-th observation, and x[i] is the
# correspnding value of attribute x
#
# Consider splits of type "x <= c" where "c" is the average of two consecutive
# values of x in the sorted order.
#
# So one child contains all elements with
# "x <= c" and the other child contains all elements with "x > c". This should
# be considered depending on the modality and skew of the attribute value
# distribution I think, in an undesirable edge case you might for example
# consider a child split without observations in it. Here we prevent this
# putting the condition that the split value has to be in the middle of two
# attribute values, meaning that there is at least one observation in each
# child node.
#
# We are given already the class labels from the credit_data array
y = credit_data[:, 5]
# print(y)
# And in the example the splits are done based on the income
#
# Now we can choose some attribute from the array to make a split on.
x = credit_data[:, 3]


def bestsplit(x, y) -> None:
    """
    @todo: Docstring for bestsplit
    """
    # Make it unique since we don't want two the same split points
    num_attr_sorted = np.sort(np.unique(x))
    # print(num_attr_sorted)
    # print(type(num_attr_sorted))

    # Use python vector addition to add all corresponding elements and take
    # their average
    consec_avg_attr_splitpoints = (num_attr_sorted[0:7] +
                                   num_attr_sorted[1:8]) / 2

    split_points = list(consec_avg_attr_splitpoints)
    # print(consec_avg_attr_splitpoints)
    # print(type(consec_avg_attr_splitpoints))

    impurity_parent_node = impurity(y)
    n_obs_parent_node = len(y)
    split_points_delta_impurities = []
    while split_points:
        split_point = split_points.pop()
        # print(split_points)
        # print('Popped:', split_point)
        child_node = {"l": y[x > split_point], "r": y[x <= split_point]}
        w_avg_child_impurities = (
            impurity(child_node["l"]) * len(child_node["l"]) + impurity(
                child_node["r"]) * len(child_node["r"])) / n_obs_parent_node
        split_points_delta_impurities += [(split_point,
                             impurity_parent_node - w_avg_child_impurities)]

    # print(split_points_delta_impurities)
    best_split, best_delta_impurity = max(split_points_delta_impurities, key=lambda x: x[1])
    print(f"{best_split=}, {best_delta_impurity=}")
    # print('reached the end')

bestsplit(x,y)
