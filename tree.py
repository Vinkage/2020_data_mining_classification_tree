# +
import numpy as np
import sklearn
import cProfile
import pandas as pd
import pstats
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
plt.style.use('seaborn-whitegrid')
plt.rc('text', usetex=True)
plt.rc('font', family='times')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('font', size=12)
plt.rc('figure', figsize=(12, 5))
# import tqdm

# from tqdm import trange
from pstats import SortKey
from sklearn import metrics
# -

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
        best_list.append((delta_i, mask_left, mask_right, split_value))
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
    for i in range(m):  # ,desc=f'planting a forest, growing {m} trees'):
        choice = np.random.randint(len(x), size=len(x))
        x_bag, y_bag = x[choice], y[choice]
        forest.append(
            tree_grow(x=x_bag,
                      y=y_bag,
                      nmin=nmin,
                      minleaf=minleaf,
                      nfeat=nfeat))
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
    for i, tree in enumerate(
            tr
    ):  # , total=len(tr),desc=f'making also {len(tr)} predictions!'):
        y_bag[:, i] = tree.predict(x).astype(float)
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


# def triple_mcnemar(x=None,
#                    y=None,
#                    predictions=[
#                        tree_pred(x=pima_indians[:, :8],
#                                  tr=tree_grow(x=pima_indians[:, :8],
#                                               y=pima_indians[:, 8],
#                                               nmin=20,
#                                               minleaf=5,
#                                               nfeat=pima_indians.shape[1] - 1),
#                                  training=pima_indians[:, 8]),
#                        tree_pred_b(x=pima_indians[:, :8],
#                                    tr=tree_grow_b(x=pima_indians[:, :8],
#                                                   y=pima_indians[:, 8],
#                                                   nmin=20,
#                                                   minleaf=5,
#                                                   nfeat=4,
#                                                   m=5),
#                                    training=pima_indians[:, 8]),
#                        tree_pred_b(x=pima_indians[:, :8],
#                                    tr=tree_grow_b(x=pima_indians[:, :8],
#                                                   y=pima_indians[:, 8],
#                                                   nmin=20,
#                                                   minleaf=5,
#                                                   nfeat=pima_indians.shape[1] -
#                                                   1,
#                                                   m=5),
#                                    training=pima_indians[:, 8])
#                    ]):
#     """
#     @todo: Docstring for significance_t_test(x
#     """
# fig, axes = plt.subplots(nrows=1, ncols=3)

# sns.set_theme(style='whitegrid')

# sns.histplot(y=, ax=ax)
# ax.set(xlabel='binary labels of y', ylabel='Rel. freq.', title='relative frequency zeroes or ones')
# ax.grid()

# fig.savefig("plots/significance_test.png")


def report_load_data_describe_balance():
    # Training data
    # +
    eclipse2 = pd.read_csv(
        './data/part2_bug_detection_article/eclipse-metrics-packages-2.0.csv',
        sep=';')
    df = pd.DataFrame(eclipse2)
    df_x_train = df[df.columns[4:44]]
    df_x_train['pre'] = df['pre']
    df_y_train = df['post']
    (df_y_train.values != 0).sum()
    len(df_y_train) - (df_y_train.values != 0).sum()
    df_y_train = df_y_train.apply(lambda x: 1 if x != 0 else 0)
    df_y_train.value_counts()
    x_train = df_x_train.to_numpy()
    y_train = df_y_train.to_numpy()
    print("Shape of training predictors:", x_train.shape,
          "\nShape of training target:", y_train.shape)
    # -

    # Test data
    # +
    eclipse3 = pd.read_csv(
        './data/part2_bug_detection_article/eclipse-metrics-packages-3.0.csv',
        sep=';')
    df = pd.DataFrame(eclipse3)
    df_x_test = df[df.columns[4:44]]
    df_x_test['pre'] = df['pre']
    df_y_test = df['post']
    (df_y_test.values != 0).sum()
    len(df_y_test) - (df_y_test.values != 0).sum()
    df_y_test = df_y_test.apply(lambda x: 1 if x != 0 else 0)
    df_y_test.value_counts()
    x_test = df_x_test.to_numpy()
    y_test = df_y_test.to_numpy()
    print("Shape of testing predictors:", x_train.shape,
          "\nShape of testing target:", y_train.shape)
    # -

    # Balance plot
    # +
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.0, 6.0))
    ax_train = axes[0]
    ax_test = axes[1]

    sns.set_theme(style='whitegrid')
    ax_train.set(
        title='Post release bugs presence in training data (\%, shape = ' +
        str(y_train.shape) + ')')
    ax_train.pie(np.c_[np.sum(np.where(y_train == 1)),
                       np.sum(np.where(y_train == 0))][0],
                 labels=['present', 'absent'],
                 colors=['g', 'r'],
                 shadow=False,
                 autopct='%.2f')
    ax_test.set(title='Post release bugs presence in test data (\%, shape = ' +
                str(y_test.shape) + ')')
    ax_test.pie(np.c_[np.sum(np.where(y_test == 1)),
                      np.sum(np.where(y_test == 0))][0],
                labels=['present', 'absent'],
                colors=['g', 'r'],
                shadow=False,
                autopct='%.2f')
    fig.savefig("./plots/pies.png", dpi=300, bbox_inches='tight')
    # -
    return df_x_test, x_train, y_train, x_test, y_test


def report_growing(X_train=None, y_train=None, feature_names=None):
    """
    @todo: Docstring for report_growing(
    """
    single_tree = tree_grow(x=X_train, y=y_train, nmin=15, minleaf=5, nfeat=41)

    bagging = tree_grow_b(x=X_train,
                          y=y_train,
                          nmin=15,
                          minleaf=5,
                          nfeat=41,
                          m=100)

    random_forest = tree_grow_b(x=X_train,
                                y=y_train,
                                nmin=15,
                                minleaf=5,
                                nfeat=6,
                                m=100)

    # Single tree plot (root split, and left or right child) plot
    # +
    root = single_tree.tree
    root_split = root.split_value_or_rows
    root_feature = root.col
    root_left, root_right = y_train[
        X_train[:, root_feature] > root_split], y_train[
            X_train[:, root_feature] <= root_split]
    df_root_left = pd.DataFrame(data=root_left, columns=['left'])
    df_root_right = pd.DataFrame(data=root_right, columns=['right'])
    df_root_left.value_counts()
    df_root_right.value_counts()
    root_feature_name = feature_names[root_feature]
    df_root = pd.DataFrame(
        data={
            'split': [
                str(root_feature_name) + ' $>$ ' + str(root_split),
                str(root_feature_name) + ' $\leq$ ' + str(root_split)
            ] * 2,
            'label': ['present'] * 2 + ['absent'] * 2,
            'value': [
                df_root_left.value_counts().values[0],
                df_root_right.value_counts().values[1],
                df_root_left.value_counts().values[1],
                df_root_right.value_counts().values[0],
            ]
        })
    df_root
    # -

    # node.is_leaf_node(node_classes)
    # +
    len(root_left)
    np.sum(root_left)
    left = root.left
    left_split = left.split_value_or_rows
    left_feature = left.col
    left_left, left_right = root_left[X_train[(
        X_train[:, root_feature] > root_split
    )][:, left_feature] > left_split], root_left[X_train[(
        X_train[:, root_feature] > root_split)][:, left_feature] <= left_split]
    df_left_left = pd.DataFrame(data=left_left, columns=['left'])
    df_left_right = pd.DataFrame(data=left_right, columns=['right'])
    df_left_left.value_counts()
    df_left_right.value_counts()
    left_feature_name = feature_names[left_feature]
    df_left = pd.DataFrame(
        data={
            'split': [
                str(left_feature_name).replace('_', ' ') + ' $>$ ' +
                '{:.3f}'.format(left_split),
                str(left_feature_name).replace('_', ' ') + ' $\leq$ ' +
                '{:.3f}'.format(left_split)
            ] * 2,
            'label': ['present'] * 2 + ['absent'] * 2,
            'value': [
                df_left_left.value_counts().values[0],
                df_left_right.value_counts().values[0],
                df_left_left.value_counts().values[1],
                df_left_right.value_counts().values[1],
            ]
        })
    df_left
    # -
    # node.is_leaf_node(node_classes)
    # +
    len(root_right)
    np.sum(root_right)
    len(root_right) - np.sum(root_right)
    right = root.right
    right_split = left.split_value_or_rows
    right_feature = left.col
    right_left, right_right = root_right[X_train[(
        X_train[:, root_feature] <= root_split
    )][:, right_feature] > right_split], root_right[X_train[(
        X_train[:, root_feature] <= root_split)][:,
                                                 right_feature] <= right_split]
    df_right_left = pd.DataFrame(data=right_left, columns=['left'])
    df_right_right = pd.DataFrame(data=right_right, columns=['right'])
    df_right_left.value_counts()
    df_right_right.value_counts()
    right_feature_name = feature_names[right_feature]
    df_right = pd.DataFrame(
        data={
            'split': [
                str(right_feature_name).replace('_', ' ') + ' $>$ ' +
                '{:.1f}'.format(right_split),
                str(right_feature_name).replace('_', ' ') + ' $\leq$ ' +
                '{:.1f}'.format(right_split)
            ] * 2,
            'label': ['present'] * 2 + ['absent'] * 2,
            'value': [
                df_right_left.value_counts().values[0],
                df_right_right.value_counts().values[1],
                df_right_left.value_counts().values[1],
                df_right_right.value_counts().values[0],
            ]
        })
    df_right
    # -

    # +
    fig, root_ax = plt.subplots(nrows=1, ncols=1, figsize=(3.0, 3.0))

    sns.set_theme(style='whitegrid')
    # ax_train.set(
    #     title='Post release bugs presence in training data (\%, shape = ' +
    #     str(y_train.shape) + ')')
    sns.barplot(x='split',
                y='value',
                hue="label",
                data=df_root,
                ax=root_ax,
                palette=['g', 'r'])

    fig.savefig("./plots/simple_tree_root.svg", dpi=300)
    # -

    # +
    fig, left_ax = plt.subplots(nrows=1, ncols=1, figsize=(3.0, 3.0))

    sns.set_theme(style='whitegrid')
    # ax_train.set(
    #     title='Post release bugs presence in training data (\%, shape = ' +
    #     str(y_train.shape) + ')')
    sns.barplot(x='split',
                y='value',
                hue="label",
                data=df_left,
                ax=left_ax,
                palette=['g', 'r'])

    fig.savefig("./plots/simple_tree_left.svg", dpi=300)
    # -
    # +
    fig, right_ax = plt.subplots(nrows=1, ncols=1, figsize=(3.0, 3.0))

    sns.set_theme(style='whitegrid')
    # ax_train.set(
    #     title='Post release bugs presence in training data (\%, shape = ' +
    #     str(y_train.shape) + ')')
    sns.barplot(x='split',
                y='value',
                hue="label",
                data=df_right,
                ax=right_ax,
                palette=['g', 'r'])

    fig.savefig("./plots/simple_tree_right.svg", dpi=300)
    # -

    # Leaf left plot
    # +
    fig, leaf = plt.subplots(nrows=1, ncols=1, figsize=(3.0, 3.0))

    sns.set_theme(style='whitegrid')
    leaf.set(title='Leaf node majority ' + str())
    leaf.pie(np.c_[np.sum(np.where(root_left == 1)),
                   np.sum(np.where(root_left == 0))][0],
             labels=['present', 'absent'],
             colors=['g', 'r'],
             shadow=False,
             autopct='%.2f')
    fig.savefig("./plots/leaf_left.svg", dpi=300, bbox_inches='tight')
    # -
    # Leafs right plot
    # +
    fig, leafs = plt.subplots(nrows=1, ncols=2, figsize=(6.0, 3.0))
    leaf_left = leafs[0]
    leaf_right = leafs[1]

    sns.set_theme(style='whitegrid')
    leaf_right.set(title='Leaf node majorities')
    leaf_left.pie(np.c_[np.sum(np.where(right_left == 1)),
                        np.sum(np.where(right_left == 0))][0],
                  labels=['present', 'absent'],
                  colors=['g', 'r'],
                  shadow=False,
                  autopct='%.2f')
    leaf_right.pie(np.c_[np.sum(np.where(right_right == 1)),
                         np.sum(np.where(right_right == 0))][0],
                   labels=['present', 'absent'],
                   colors=['g', 'r'],
                   shadow=False,
                   autopct='%.2f')
    fig.savefig("./plots/leafs_right.svg", dpi=300, bbox_inches='tight')
    # -
    return single_tree, bagging, random_forest


def report_predictions(X_test=None,
                       y_test=None,
                       single_tree=None,
                       bagging=None,
                       random_forest=None):
    """
    @todo: Docstring for report_predictions(X_test=None, y_test=None, single_tree=None, bagging=None, random_forest=None)
    """
    if single_tree is not None:
        yhat_single_tree = tree_pred(x=X_test, tr=single_tree, training=y_test)
    if bagging is not None:
        yhat_bagging = tree_pred_b(x=X_test, tr=bagging, training=y_test)
    if random_forest is not None:
        yhat_random_forest = tree_pred_b(x=X_test,
                                         tr=random_forest,
                                         training=y_test)
    return yhat_single_tree, yhat_bagging, yhat_random_forest


def confusion_matrix(y_test, yhat_single_tree, yhat_bagging,
                     yhat_random_forest):
    """
    @todo: Docstring for confusion_matrix(
    """
    single_confusion = metrics.confusion_matrix(yhat_single_tree, y_test)
    bag_confusion = metrics.confusion_matrix(yhat_bagging, y_test)
    rf_confusion = metrics.confusion_matrix(yhat_random_forest, y_test)
    df_single_confusion = pd.DataFrame(data=single_confusion,
                                       columns=['N', 'P'],
                                       index=['N', 'P'])
    df_bag = pd.DataFrame(data=bag_confusion,
                          columns=['N', 'P'],
                          index=['N', 'P'])
    df_rf = pd.DataFrame(data=rf_confusion,
                         columns=['N', 'P'],
                         index=['N', 'P'])
    # Confusion matrices using seaborn
    # +
    fig, matrices = plt.subplots(nrows=1, ncols=3, figsize=(12.0, 3.0))
    single = matrices[0]
    bag = matrices[1]
    rf = matrices[2]

    sns.set_theme(style='whitegrid')
    single.tick_params(labelbottom=False,labeltop=True, length=0.5)
    sns.heatmap(data=df_single_confusion,
                cbar=True,
                ax=single,
                square=True,
                annot=True,
                fmt='d',
                cmap="YlGnBu")
    single.set(title='Test', ylabel='Single tree')

    bag.tick_params(labelbottom=False,labeltop=True, length=0.5)
    sns.heatmap(data=df_bag,
                cbar=True,
                ax=bag,
                square=True,
                annot=True,
                fmt='d',
                cmap="YlGnBu")
    bag.set(title='Test', ylabel='Bagged trees')

    rf.tick_params(labelbottom=False,labeltop=True, length=0.5)
    sns.heatmap(data=df_rf,
                cbar=True,
                ax=rf,
                square=True,
                annot=True,
                fmt='d',
                cmap="YlGnBu")
    rf.set(title='Test', ylabel='Random forest')

    from matplotlib.ticker import PercentFormatter
    cbar = single.collections[0].colorbar
    cbar.set_ticks(sorted([i[j] for i in single_confusion.tolist() for j in range(len(i))]))
    cbar.ax.tick_params(labelsize=10, length=0.5) 
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(len(y_test), 0))

    cbar = bag.collections[0].colorbar
    cbar.set_ticks(sorted([i[j] for i in bag_confusion.tolist() for j in range(len(i))]))
    cbar.ax.tick_params(labelsize=10, length=0.5) 
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(len(y_test), 0))

    cbar = rf.collections[0].colorbar
    cbar.set_ticks(sorted([i[j] for i in rf_confusion.tolist() for j in range(len(i))]))
    cbar.ax.tick_params(labelsize=10, length=0.5) 
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(len(y_test), 0))

    fig.savefig("./plots/cm.png", dpi=300, bbox_inches='tight')
    # -


def report():
    """
    @todo: Docstring for report():
    """

    # +
    # -
    df_x_test, X_train, y_train, X_test, y_test = report_load_data_describe_balance(
    )

    # 1. Should save tree plot in './plots/single_tree_splits.png'
    single_tree, bagging, random_forest = report_growing(
        X_train=X_train, y_train=y_train, feature_names=df_x_test.columns)

    # 1. Should write to a latex file the confusion matrices and quality measures
    yhat_single_tree, yhat_bagging, yhat_random_forest = report_predictions(
        X_test=X_test,
        y_test=y_test,
        single_tree=single_tree,
        bagging=bagging,
        random_forest=random_forest)
    confusion_matrix(y_test, yhat_single_tree, yhat_bagging,
                     yhat_random_forest)

    # 1. Calculates the mcnemar X2 value and plots the single degree of freedom X2 plot
    report_significance()
    pass


if __name__ == '__main__':
    # credit_data = np.genfromtxt('./data/credit_score.txt',
    #                             delimiter=',',
    #                             skip_header=True)

    # pima_indians = np.genfromtxt('./data/pima_indians.csv',
    #                              delimiter=',',
    #                              skip_header=True)

    # print("\nDataset: credit data")
    # tree_pred(x=credit_data[:, :5],
    #           tr=tree_grow(x=credit_data[:, 0:5],
    #                        y=credit_data[:, 5],
    #                        nmin=2,
    #                        minleaf=1,
    #                        nfeat=5),
    #           training=credit_data[:, 5])

    # print("\nDataset: credit data")
    # tree_pred_b(x=credit_data[:, :5],
    #             tr=tree_grow_b(x=credit_data[:, 0:5],
    #                            y=credit_data[:, 5],
    #                            nmin=2,
    #                            minleaf=1,
    #                            nfeat=4,
    #                            m=50),
    #             training=credit_data[:, 5])

    # print('\nDataset: pima indians')
    # tree_pred(x=pima_indians[:, :8],
    #           tr=tree_grow(x=pima_indians[:, :8],
    #                        y=pima_indians[:, 8],
    #                        nmin=20,
    #                        minleaf=5,
    #                        nfeat=pima_indians.shape[1] - 1),
    #           training=pima_indians[:, 8])

    # print('\nDataset: pima indians')
    # tree_pred_b(x=pima_indians[:, :8],
    #             tr=tree_grow_b(x=pima_indians[:, :8],
    #                            y=pima_indians[:, 8],
    #                            nmin=20,
    #                            minleaf=5,
    #                            nfeat=4,
    #                            m=5),
    #             training=pima_indians[:, 8])

    # print('\nDataset: pima indians')
    # tree_pred_b(x=pima_indians[:, :8],
    #             tr=tree_grow_b(x=pima_indians[:, :8],
    #                            y=pima_indians[:, 8],
    #                            nmin=20,
    #                            minleaf=5,
    #                            nfeat=pima_indians.shape[1] - 1,
    #                            m=5),
    #             training=pima_indians[:, 8])

    # Time profiles: see what functions take what time! :)

    # print("prediction metrics single tree pima indians:")
    # cProfile.run("tree_pred(x=credit_data[:,:5], tr=tree_grow(x=credit_data[:,0:5], y=credit_data[:,5], nmin=2, minleaf=1, nfeat=5), dataset='credit score')", 'restats')

    # triple_mcnemar(x=pima_indians[:, :8], y=pima_indians[:, 8])

    # Time profile of pima indians data prediction with single tree
    # print("prediction metrics single tree pima indians:")
    # cProfile.run(
    #     "tree_pred_b(x=pima_indians[:, :8], tr=tree_grow_b(x=pima_indians[:, :8], y=pima_indians[:, 8], nmin=20, minleaf=5, nfeat=4, m=5), training=pima_indians[:, 8])",
    #     'restats')

    # p = pstats.Stats('restats')
    # p.sort_stats(SortKey.TIME)
    # p.print_stats()
    report()
