import numpy as np
# import cProfile
# import pstats
# import tqdm

# from tqdm import trange
# from pstats import SortKey
from sklearn import metrics

# age,married,house,income,gender,class
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

# In the program data points are called rows

# In the program categorical or numerical attributes are called cols for columns

# The last column are the classes and will be called as classes in the program


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
        # This weird numpy line gives the majority vote, which is 1 or 0
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
        node = nodes[0]
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

    # Work in progress tree printer
    #
    # def __repr__(self):
    #     tree_string = ''
    #     node = self.tree
    #     depth = 0
    #     nodelist = [node]
    #     while nodelist:
    #         node = nodelist.pop()
    #         depth += 1
    #         if node.col is not None:
    #             left, right = node.left, node.right
    #             nodelist += [left, right]
    #         else:
    #             continue
    #         tree_string += '\n' + depth * ' '
    #         tree_string += (depth + 4) * ' ' + '/' + ' ' * 2 + '\\'
    #         tree_string += '\n' + ' ' * 2 * depth
    #         for direc in left, right:
    #             if not direc.split_value_or_rows%10:
    #                 tree_string += ' ' * 4
    #             else:
    #                 tree_string += ' ' * 3
    #             tree_string += str(int(direc.split_value_or_rows))

    #     tree_string = depth * ' ' + str(int(self.tree.split_value_or_rows)) + tree_string
    #     return tree_string

def major_vote(classes):
    """
    @todo: Docstring for major_vote(classes
    """
    return np.argmax(np.bincount(classes.astype(int)))

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
    # Stop wanneer de array met split points leeg is
    while split_points.size != 0:
        split_value = split_points[-1]
        mask_left, mask_right = x > split_value, x <= split_value
        classes_left, classes_right = y[mask_left], y[mask_right]

        # Kijk of er genoeg rows in de gesplitte nodes terechtkomen, anders
        # mogen we de split niet toelaten vanwege de minleaf constraint
        if len(classes_left) < minleaf or len(classes_right) < minleaf:
            split_points = split_points[:-1]
            continue

        delta_i = (impurity(classes_left) * len(classes_left) +
                   impurity(classes_right) * len(classes_right))
        # stop huidige splits in de lijst om best split te berekenen
        np.append(best_list, (delta_i, mask_left, mask_right, split_value))
        # Haal de huidige split_point uit split_points
        split_points = split_points[:-1]

    # Bereken de best split voor deze x col, als er ten minste 1 bestaat die
    # voldoet aan min leaf
    if best_list:
        return min(best_list, key=lambda x: x[0])
    else:
        return False


def exhaustive_split_search(rows, classes, minleaf):
    """
    @todo: Docstring for exhaustive_split_search
    """
    # We hebben enumerate nodig, want we willen weten op welke col (i)
    # (age,married,house,income,gender) we een split doen
    exhaustive_best_list = []
    for i, col in enumerate(rows.transpose()):
        col_best_split = bestsplit(col, classes, minleaf)
        if col_best_split:
            # add for which row we calculated the best split
            col_best_split += (i, )
            exhaustive_best_list.append(col_best_split)
    return exhaustive_best_list


def add_children(node, best_split):
    """
    @todo: Docstring for add_children
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


#
#
# Put all helper functions above this comment!


def tree_grow(x=None,
              y=None,
              nmin=None,
              minleaf=None,
              nfeat=None,
              **defaults) -> Tree:
    """
    @todo: Docstring for tree_grow
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
            node_rows = x[node.split_value_or_rows]
            exhaustive_best_list = exhaustive_split_search(
                node_rows, node_classes, minleaf)
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


def tree_grow_b(x=None,
                y=None,
                nmin=None,
                minleaf=None,
                nfeat=None,
                m=None,
                **defaults) -> Tree:
    forest = []
    for i in range(m):# ,desc=f'planting a forest, growing {m} trees'):
        choice = np.random.randint(len(x),size=len(x))
        x_bag, y_bag = x[choice], y[choice]
        forest.append(tree_grow(x=x_bag,y=y_bag,nmin=nmin,minleaf=minleaf,nfeat=nfeat))
    return forest



def tree_pred(x=None, tr=None, training=None, **defaults) -> np.array:
    """
    @todo: Docstring for tree_pred
    """
    y = tr.predict(x).astype(float)
    nmin, minleaf, nfeat = tr.hyper_params
    if training is not None:
        # print(np.mean(training == y))
        print(
            f'Results from: prediction single tree({nmin=}, {minleaf=}, {nfeat=})'
        )
        print(
            f'\t->Confusion matrix:\n{metrics.confusion_matrix(y, training)}')
        print(f'\t->Accuracy:\n\t\t{metrics.accuracy_score(y, training)}')
        print(f'\t->Precission:\n\t\t{metrics.precision_score(y, training)}')
        print(f'\t->Recall:\n\t\t{metrics.recall_score(y, training)}')
    return y


def tree_pred_b(x=None, tr=None, training=None, **defaults) -> np.array:
    y_bag = np.zeros((len(x), len(tr)))
    for i, tree in enumerate(tr):   # , total=len(tr),desc=f'making also {len(tr)} predictions!'):
        y_bag[:,i] = tree.predict(x).astype(float)
    nmin, minleaf, nfeat = tr[0].hyper_params
    y = np.array([major_vote(y_bag[i]) for i in range(len(y_bag))])
    if training is not None:
        # print(np.mean(training == y))
        if nfeat == x.shape[1]:
            print(
                f'Results from: prediction bagged tree({nmin=}, {minleaf=}, {nfeat=}, trees={len(tr)})'
            )
        else:
            print(
                f'Results from: prediction random forest({nmin=}, {minleaf=}, {nfeat=}, trees={len(tr)})'
            )
        print(
            f'\t->Confusion matrix:\n{metrics.confusion_matrix(y, training)}')
        print(f'\t->Accuracy:\n\t\t{metrics.accuracy_score(y, training)}')
        print(f'\t->Precission:\n\t\t{metrics.precision_score(y, training)}')
        print(f'\t->Recall:\n\t\t{metrics.recall_score(y, training)}')
    return y


if __name__ == '__main__':
    credit_data = np.genfromtxt('./data/credit_score.txt',
                                delimiter=',',
                                skip_header=True)

    pima_indians = np.genfromtxt('./data/pima_indians.csv',
                                 delimiter=',',
                                 skip_header=True)

    print("\nDataset: credit data")
    tree_pred(x=credit_data[:, :5],
              tr=tree_grow(x=credit_data[:, 0:5],
                           y=credit_data[:, 5],
                           nmin=2,
                           minleaf=1,
                           nfeat=5),
              training=credit_data[:, 5])

    print("\nDataset: credit data")
    tree_pred_b(x=credit_data[:, :5],
                tr=tree_grow_b(x=credit_data[:, 0:5],
                               y=credit_data[:, 5],
                               nmin=2,
                               minleaf=1,
                               nfeat=4,
                               m=50),
                training=credit_data[:, 5])

    print('\nDataset: pima indians')
    tree_pred(x=pima_indians[:, :8],
              tr=tree_grow(x=pima_indians[:, :8],
                           y=pima_indians[:, 8],
                           nmin=20,
                           minleaf=5,
                           nfeat=pima_indians.shape[1] - 1),
              training=pima_indians[:, 8])


    print('\nDataset: pima indians')
    tree_pred_b(x=pima_indians[:, :8],
                tr=tree_grow_b(x=pima_indians[:, :8],
                               y=pima_indians[:, 8],
                               nmin=20,
                               minleaf=5,
                               nfeat=4,
                               m=5),
                training=pima_indians[:, 8])

    

    # Time profiles: see what functions take what time! :)

    # print("prediction metrics single tree pima indians:")
    # cProfile.run("tree_pred(x=credit_data[:,:5], tr=tree_grow(x=credit_data[:,0:5], y=credit_data[:,5], nmin=2, minleaf=1, nfeat=5), dataset='credit score')", 'restats')

    # Time profile of pima indians data prediction with single tree
    # print("prediction metrics single tree pima indians:")
    # cProfile.run(
    #     "tree_pred_b(x=pima_indians[:, :8], tr=tree_grow_b(x=pima_indians[:, :8], y=pima_indians[:, 8], nmin=20, minleaf=5, nfeat=4, m=5), training=pima_indians[:, 8])",
    #     'restats')

    # p = pstats.Stats('restats')
    # p.sort_stats(SortKey.TIME)
    # p.print_stats()
