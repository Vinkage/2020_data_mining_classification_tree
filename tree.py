import numpy as np
from sklearn import metrics
#- Made by:

# Hunter Sterk
# Lonnie Bregman
# Mike Vink

#- Main functions:

# def tree_grow(x,y,nmin,minleaf,nfeat):
#     """
#     /x/ numpy.ndarray, 2D numpy array containing data rows and feature columns
#     /y/ numpy.ndarray, 1D numpy array containing binary x-row labels
#     /nmin/ int, number of x-rows that a parent must contain before splitting
#     /minleaf/ int, number of x-rows a child must have before splitting
#     /nfeat/ int, number of x-columns randomly considered before splitting

#       Returns -> Tree object

#       tree_grow returns a tree object that stores a classification tree in a
#     data structure that is similar to a linked list. To build
#     the tree, it exhaustively considers the splits possible using nfeat
#     random x-columns. The gini-index is used to determine the best split.
#     Stopping rules constraining the number of x-rows in parent and child
#     nodes are used as complexity parameters.

#     EXAMPLE:
#     >>> x
#     array([[22.,  0.,  0., 28.,  1.],
#            [46.,  0.,  1., 32.,  0.],
#            [24.,  1.,  1., 24.,  1.],
#            [25.,  0.,  0., 27.,  1.],
#            [29.,  1.,  1., 32.,  0.],
#            [45.,  1.,  1., 30.,  0.],
#            [63.,  1.,  1., 58.,  1.],
#            [36.,  1.,  0., 52.,  1.],
#            [23.,  0.,  1., 40.,  0.],
#            [50.,  1.,  1., 28.,  0.]])
#     >>> y
#     array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.])
#     >>> tree_grow(x=x,y=y,nmin=2,minleaf=1,nfeat=5)
#     <__main__.Tree object at 0x10d752ee0>

#     """

# def tree_pred(x,tr,true):
#     """
#     /x/ numpy.ndarray, 2D numpy array containing data rows and feature columns
#     /tr/ Tree object, tree to predict a binary label for each row in x
#     /true/ numpy.ndarray, 1D numpy array containing "true" labels
#

#       Returns -> numpy.ndarray, 1D numpy array containing predicted binary
#                   labels for each row in x

#       tree_pred uses a tree object to predict binary labels on a given 2D
#     data array x. The "true" argument should only be given if predictions
#     metrics are to be calculated and printed, which gives an immediate idea
#     if the tree is erronous by seeing low prediction performance on the
#     training data for example.

#     EXAMPLE:
#     >>> x
#     array([[22.,  0.,  0., 28.,  1.],
#            [46.,  0.,  1., 32.,  0.],
#            [24.,  1.,  1., 24.,  1.],
#            [25.,  0.,  0., 27.,  1.],
#            [29.,  1.,  1., 32.,  0.],
#            [45.,  1.,  1., 30.,  0.],
#            [63.,  1.,  1., 58.,  1.],
#            [36.,  1.,  0., 52.,  1.],
#            [23.,  0.,  1., 40.,  0.],
#            [50.,  1.,  1., 28.,  0.]])
#     >>> tr = tree_grow(x=x,y=y,nmin=2,minleaf=1,nfeat=5)
#     >>> tree_pred(x, tr)
#     array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.])

#     """

# def tree_grow_b(x,y,nmin,minleaf,nfeat,m):
#     """
#     /x/ numpy.ndarray, 2D numpy array containing data rows and feature columns
#     /y/ numpy.ndarray, 1D numpy array containing binary x-row labels
#     /nmin/ int, number of x-rows that a parent must contain before splitting
#     /minleaf/ int, number of x-rows a child must have before splitting
#     /nfeat/ int, number of x-columns randomly considered before splitting
#     /m/ int, number of bootstrap samples to draw

#       Returns -> list of m Tree objects

#       tree_grow_b returns a list of Tree objects that store a classification
#     tree in a data structure that is similar to a linked list. To build the
#     tree, it exhaustively considers the splits possible using nfeat random
#     x-columns. In case of a random forest nfeat should be lower than the
#     number of columns in x. The gini-index is used to determine the best
#     split. Stopping rules constraining the number of x-rows in parent and
#     child nodes are used as complexity parameters.

#     EXAMPLE:
#     >>> x
#     array([[22.,  0.,  0., 28.,  1.],
#            [46.,  0.,  1., 32.,  0.],
#            [24.,  1.,  1., 24.,  1.],
#            [25.,  0.,  0., 27.,  1.],
#            [29.,  1.,  1., 32.,  0.],
#            [45.,  1.,  1., 30.,  0.],
#            [63.,  1.,  1., 58.,  1.],
#            [36.,  1.,  0., 52.,  1.],
#            [23.,  0.,  1., 40.,  0.],
#            [50.,  1.,  1., 28.,  0.]])
#     >>> y
#     array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.])
#     >>> trees = tree_grow_b(x=x,y=y,nmin=2,minleaf=1,nfeat=4,m=50)
#     >>> type(trees), type(trees[0]), len(trees) == m
#     (<class 'list'>, <class '__main__.Tree'>, True)

#     """

# def tree_pred_b(x,tr,true):
#     """
#     /x/ numpy.ndarray, 2D numpy array containing data rows and feature columns
#     /tr/ list of Tree objects, trees predict a binary label for each row in
#                           x, which are used to make a majority vote on the
#                           final predicted labels
#     /true/ numpy.ndarray, 1D numpy array containing "true" labels
#

#       Returns -> numpy.ndarray, 1D numpy array containing predicted binary
#                   labels for each row in x

#       tree_pred_b uses a list of tree objects to predict binary labels on a
#     given 2D data array x. The main difference with tree_pred is that now we
#     get for each tree a 1D numpy array of predicted binary labels for each
#     row in x. A 2D numpy array is constructed where each column corresponds
#     to all predictions for one tree, and a row corresponds to a row in x.
#     Therefore we take the majority vote of rows in this array to return a 1D
#     numpy array with the final predicted labels of the trees.

#     The "true" argument should only be given if predictions metrics are to be
#     calculated and printed, which gives an immediate idea if the tree is
#     erronous by seeing low prediction performance on the training data for
#     example.

#     EXAMPLE:
#     >>> x
#     array([[22.,  0.,  0., 28.,  1.],
#            [46.,  0.,  1., 32.,  0.],
#            [24.,  1.,  1., 24.,  1.],
#            [25.,  0.,  0., 27.,  1.],
#            [29.,  1.,  1., 32.,  0.],
#            [45.,  1.,  1., 30.,  0.],
#            [63.,  1.,  1., 58.,  1.],
#            [36.,  1.,  0., 52.,  1.],
#            [23.,  0.,  1., 40.,  0.],
#            [50.,  1.,  1., 28.,  0.]])
#     >>> type(trees), type(trees[0]), len(trees) == 50
#     (<class 'list'>, <class '__main__.Tree'>, True)
#     >>> tree_pred_b(x=x,tr=trees)
#     array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

#     """

#- Miscellaneous functions:

# def major_vote(classes):
#     """
#     /classes/ numpy.array, 1D numpy array of zeroes and ones

#       Returns -> int

#       Uses numpy methods to calculate if 1 or 0 elements are the majority in
#     the classes vector. Note that when the number of 1 and 0 elements are
#     equal, it returns 0.

#     EXAMPLE:
#     >>> y
#     array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.])
#     >>> major_vote(y)
#     0

#     """

# def impurity(array):
#     """
#     /array/ numpy.array, 1D numpy array of zeroes and ones

#      Returns -> float

#      Computes the gini index impurity based on the relative frequency of ones
#    in the vector.
#
#    EXAMPLE:
#    >>> array=np.array([1,0,1,1,1,0,0,1,1,0,1])
#    >>> array
#    array([1,0,1,1,1,0,0,1,1,0,1])
#    >>> impurity(array)
#    0.23140495867768596

#     """

# def bestsplit(x,y,minleaf):
#     """
#     /x/ numpy.array, 1D numpy array corresponding to a feature column of the
#                     2D data array corresponding to some node
#     /y/ numpy.array, 1D numpy array of binary labels corresponding to the
#                     rows in the 2D data array corresponding to some node
#     /minleaf/ int, number of x-rows a child must have before splitting

#       Returns -> tuple

#      Computes the best split based on the given features using the impurity
#    function.

#    EXAMPLE:
#    >>> x
#    array([28., 32., 24., 27., 32., 30., 58., 52., 40., 28.])
#    >>> y
#    array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.])
#    >>> bestsplit(x,y,minleaf=1)
#    (1.4285714285714286, array([False, False, False, False, False, False,  True,  True,  True,
#           False]), array([ True,  True,  True,  True,  True,  True, False, False, False,
#            True]), 36.0)
#

#     """

# def exhaustive_split_search(rows, classes, minleaf):
#     """
#     /rows/ numpy.array, 2D data array corresponding to some node
#     /classes/ numpy.array, 1D numpy array of binary labels corresponding to the
#                     rows in the 2D data array corresponding to some node
#     /minleaf/ int, number of x-rows a child must have before splitting

#       Returns -> List

#       Stores the best splits computed with the bestsplit function for the
#     considered columns in a list, if the list is empty then there are no
#     splits and the node becomes a leaf node.

#     """

# def add_children(node, best_split):
#     """
#     /node/ Node object, the current node in the main tree growing loop
#     /best_split/ tuple, tuple containing (delta_i, rows_left, rows_right, splitvalue)

#       Processes the splits into the tree data structure and returns children yet
#     to be splitted to the main nodelist in tree_grow.

#     """

# def update_mask(mask, current_mask):
#     """
#     /mask/ np.array, 1D boolean vector corresponding to the rows in the new
#       child node that might have a length that is incompatible with the rows in
#       the main 2D data array x of tree_grow
#     /current_mask/ np.array, 1D boolean vector corresponding to the rows in the current node

#       Updates the spit bool array from any dimension to an array with length
#     equal to the total number of rows in dataset x of tree_grow.

#     """

import numpy as np


class Node:
    """
    The node object points to two other Node objects.
    """
    def __init__(self, split_value_or_rows=None, col=None):
        """Initialises the column and split value for the node.

        /split_value_or_rows=None/ can either be the best split value of
        a col, or a boolean mask for x that selects the rows to consider for
        calculating the split_value

        /col=None/ if the node object has a split_value, then it also has a col
        that belongs to this value

        """
        self.split_value_or_rows = split_value_or_rows
        self.col = col

    def add_split(self, left, right):
        """
        Lets the node object point to two other objects that can be either Leaf
        or Node.
        """
        self.left = left
        self.right = right

    def is_leaf_node(self, node_classes):
        """
        is_leaf_node is used to change the col attribute to None to indicate a
        leaf node
        """
        self.col = None
        self.split_value_or_rows = major_vote(node_classes)


class Tree:
    """
    Tree object that points towards the root node.
    """
    def __init__(self, root_node_obj, hyper_params):
        """Initialises only by pointing to a Node object.

        /root_node_obj/ is a node object that is made before entering the main
        loop of tree grow.

        """
        self.tree = root_node_obj
        self.hyper_params = hyper_params

    def predict(self, x):
        """
        Makes a list of root nodes, and drops one row of x through the tree per
        loop
        """
        # Maak een lijst van nodes, wiens indexes overeen komen met de rows in
        # x die we willen droppen
        rows_to_predict = len(x)
        nodes = np.array([self.tree] * rows_to_predict)
        predictions = np.zeros(rows_to_predict)

        # # De index van de row van x die we in de boom willen droppen
        drop = 0
        while nodes.size != 0:
            node = nodes[0]
            if node.col is None:
                node = node.split_value_or_rows
                predictions[drop] = node
                nodes = nodes[1:]
                drop += 1
                continue
            elif x[drop, node.col] > node.split_value_or_rows:
                nodes[0] = node.left
            else:
                nodes[0] = node.right
        return predictions


def tree_grow(x=None, y=None, nmin=None, minleaf=None, nfeat=None) -> Tree:
    """
    Builds a classification tree given training data and labels using stopping
    rule complexity parameters.
    """
    mask = np.full(len(x), True)
    root = Node(split_value_or_rows=mask)
    tr = Tree(root, (nmin, minleaf, nfeat))

    nodelist = [root]
    while nodelist:
        node = nodelist.pop()
        node_classes = y[node.split_value_or_rows]

        if len(node_classes) < nmin:
            node.is_leaf_node(node_classes)
            continue

        if impurity(node_classes) > 0:
            # FIX: feature choice that was lost in versioning
            # OLD: node_rows = x[node.split_value_or_rows]
            # node_rows = x[node.split_value_or_rows]
            # print(node.split_value_or_rows)

            nfeat_col_choice = np.random.choice(range(x.shape[1]), nfeat, replace=False)
            feat_select = np.sort(nfeat_col_choice)
            node_rows = x[node.split_value_or_rows][:, feat_select]

            exhaustive_best_list = exhaustive_split_search(
                node_rows, node_classes, feat_select, minleaf)
            if not exhaustive_best_list:
                node.is_leaf_node(node_classes)
                continue
            best_split = min(exhaustive_best_list, key=lambda z: z[0])
            nodelist += add_children(node, best_split)

        else:
            # impurity 0
            node.is_leaf_node(node_classes)
            continue
    return tr


def tree_pred(x=None, tr=None, true=None) -> np.array:
    """
    Predicts a binary label for each row in x using tr.predict.
    """
    y = tr.predict(x).astype(float)
    nmin, minleaf, nfeat = tr.hyper_params
    if true is not None:
        print(
            f'Results from: prediction single tree({nmin=}, {minleaf=}, {nfeat=})'
        )
        print(f'\t->Confusion matrix:\n{metrics.confusion_matrix(y, true)}')
        print(f'\t->Accuracy:\n\t\t{metrics.accuracy_score(y, true)}')
        print(f'\t->Precission:\n\t\t{metrics.precision_score(y, true)}')
        print(f'\t->Recall:\n\t\t{metrics.recall_score(y, true)}')
    return y


def tree_grow_b(x=None,
                y=None,
                nmin=None,
                minleaf=None,
                nfeat=None,
                m=None) -> Tree:
    """
    The m times repeated application of tree_grow with bagged data.
    """
    forest = []
    for i in range(m):
        choice = np.random.randint(len(x), size=len(x))
        x_bag, y_bag = x[choice], y[choice]
        forest.append(
            tree_grow(x=x_bag,
                      y=y_bag,
                      nmin=nmin,
                      minleaf=minleaf,
                      nfeat=nfeat))
    return forest


def tree_pred_b(x=None, tr=None, true=None) -> np.array:
    """
    The repeated application of tree.predict to construct a 2D array which is
    used to make a majority vote label prediction for the rows in x.
    """
    y_bag = np.zeros((len(x), len(tr)))
    for i, tree in enumerate(tr):
        y_bag[:, i] = tree.predict(x).astype(float)
    nmin, minleaf, nfeat = tr[0].hyper_params
    y = np.array([major_vote(y_bag[i]) for i in range(len(y_bag))])
    if true is not None:
        if nfeat == x.shape[1]:
            print(
                f'Results from: prediction bagged tree({nmin=}, {minleaf=}, {nfeat=}, trees={len(tr)})'
            )
        else:
            print(
                f'Results from: prediction random forest({nmin=}, {minleaf=}, {nfeat=}, trees={len(tr)})'
            )
        print(f'\t->Confusion matrix:\n{metrics.confusion_matrix(y, true)}')
        print(f'\t->Accuracy:\n\t\t{metrics.accuracy_score(y, true)}')
        print(f'\t->Precission:\n\t\t{metrics.precision_score(y, true)}')
        print(f'\t->Recall:\n\t\t{metrics.recall_score(y, true)}')
    return y


#
#
# Put all helper functions below this comment!
def major_vote(classes):
    """
    Returns a zero or one based on the highest occurence in the classes vector.
    """
    return np.argmax(np.bincount(classes.astype(int)))


def impurity(array) -> int:
    """
    Calculates the impurity of the labels in a node.
    """
    n_labels = len(array)
    n_labels_1 = array.sum()
    rel_freq_1 = n_labels_1 / n_labels
    rel_freq_0 = 1 - rel_freq_1
    gini_index = rel_freq_1 * rel_freq_0
    return gini_index


def bestsplit(x, y, minleaf) -> None:
    """
    x = vector of single col
    y = vector of classes (last col in x)

    Consider splits of type "x <= c" where "c" is the average of two consecutive
    values of x in the sorted order.

    x and y must be of the same length

    y[i] must be the class label of the i-th observation, and x[i] is the
    correspnding value of attribute x

    Example (best split on income):

    >>> bestsplit(credit_data[:,3],credit_data[:,5])
     36
    """
    x_sorted = np.sort(np.unique(x))
    split_points = (x_sorted[:len(x_sorted) - 1] + x_sorted[1:]) / 2

    # Hieren stoppen we (delta_i, split_value, rows_left, rows_right)
    best_list = []
    while split_points.size != 0:
        split_value = split_points[-1]

        mask_left, mask_right = x > split_value, x <= split_value
        classes_left, classes_right = y[mask_left], y[mask_right]

        if len(classes_left) < minleaf or len(classes_right) < minleaf:
            split_points = split_points[:-1]
            continue

        delta_i = (impurity(classes_left) * len(classes_left) +
                   impurity(classes_right) * len(classes_right))

        best_list.append((delta_i, mask_left, mask_right, split_value))

        split_points = split_points[:-1]

    # Bereken de best split voor deze x col, als er ten minste 1 bestaat die
    # voldoet aan min leaf
    if best_list:
        return min(best_list, key=lambda x: x[0])
    else:
        return False


def exhaustive_split_search(rows, classes, feat_select, minleaf):
    """
    The nfeat repeated application of bestsplit.
    """
    # We hebben enumerate nodig, want we willen weten op welke col (i)
    # (age,married,house,income,gender) we een split doen
    exhaustive_best_list = []
    for i, col in enumerate(rows.transpose()):
        col_best_split = bestsplit(col, classes, minleaf)
        if col_best_split:
            # add for which row we calculated the best split
            col_best_split += (feat_select[i], )
            exhaustive_best_list.append(col_best_split)
    return exhaustive_best_list


def add_children(node, best_split):
    """
    Processes the splits into the tree data structure and returns children yet
    to be splitted to the nodelist in tree_grow.
    """
    current_mask = node.split_value_or_rows
    mask_left, mask_right, node_split_value, node_col = best_split[1:]

    # Give the current node the split_value and col it needs for predictions
    node.split_value_or_rows, node.col = node_split_value, node_col

    # Updating the row masks to give it to children, keeping numpy dimension consistent
    mask_left, mask_right = update_mask(mask_left, current_mask), update_mask(
        mask_right, current_mask)

    # Adding the pointer between parent and children
    node.add_split(Node(split_value_or_rows=mask_left),
                   Node(split_value_or_rows=mask_right))
    return [node.left, node.right]


def update_mask(mask, current_mask):
    """
    Updates the spit bool array from any dimension to an array with length
    equal to the total number of rows in dataset x.
    """
    copy = np.array(current_mask, copy=True)
    copy[current_mask == True] = mask
    return copy

if __name__ == '__main__':
    c = np.loadtxt('./data/credit_score.txt', delimiter=',', skiprows=1)
    x, y = c[:,0:5], c[:,5]
    tr = tree_grow(x=x, y=y, nmin=2, minleaf=1, nfeat=5)
    tree_pred(x, tr, true=y)

    c = np.loadtxt('./data/credit_score.txt', delimiter=',', skiprows=1)
    x, y = c[:,0:5], c[:,5]

    trs = tree_grow_b(x=x, y=y, nmin=2, minleaf=1, nfeat=4, m=50)
    tree_pred_b(x, trs, true=y)


    c = np.loadtxt('./data/pima_indians.csv', delimiter=',')
    x, y = c[:,0:8], c[:,8].astype(int)

    tr = tree_grow(x=x, y=y, nmin=20, minleaf=5, nfeat=8)
    tree_pred(x, tr, true=y)


    c = np.loadtxt('./data/pima_indians.csv', delimiter=',')
    x, y = c[:,0:8], c[:,8].astype(int)

    trs = tree_grow_b(x=x, y=y, nmin=20, minleaf=5, nfeat=4, m=5)
    tree_pred_b(x, trs, true=y)


