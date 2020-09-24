import numpy as np
from sklearn import metrics
import pstats
from pstats import SortKey
import cProfile


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
        self.split_value_or_rows = np.argmax(
            np.bincount(node_classes.astype(int)))


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
        nodes = [self.tree] * len(x)

        # De index van de row van x die we in de boom willen droppen
        drop = 0
        while not all(pred_class in {0,1} for pred_class in nodes):
            # Als de col None is dan is het een leaf node, dus dan is de row
            # van x hier beland
            # print(nodes[drop].col, nodes[drop].split_value_or_rows)
            if nodes[drop].col is None:
                nodes[drop] = nodes[drop].split_value_or_rows
                drop += 1
                continue

            # Vergelijk de x col (age,married,house,income,gender,class), in de
            # gedropte row met het split value van de node. Op basis hiervan
            # drop naar links of rechts
            if x[drop, nodes[drop].col] > nodes[drop].split_value_or_rows:
                nodes[drop] = nodes[drop].left
            else:
                nodes[drop] = nodes[drop].right
        return np.array(nodes)

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
    if n_labels == 0:
        print(
            "division by zero will happen, child node is pure, doesnt contain anything"
        )
        n_labels = 1
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
    # if len(x_sorted) <= 2:
    #     # Allows splitting on categorical (0 or 1) cols
    #     split_points = x_sorted / 2
    # else:
    #     # Take average between consecutive numerical rows in the x col
    #     split_points = (x_sorted[:len(x_sorted) - 1] + x_sorted[1:]) / 2

    split_points = (x_sorted[:len(x_sorted) - 1] + x_sorted[1:]) / 2
    # print(split_points)

    # De toepassing van bestsplit verdeelt de x col vector in tweeen, twee
    # arrays van "x rows". Deze moeten we terug krijgen om de in de child nodes
    # bestsplit toe te passen.
    #
    # Deze lus berekent de best split value, en op basis daarvan weten we welke
    # twee "x rows" arrays we moeten returnen, en welke split value het beste
    # was natuurlijk.

    # Hieren stacken we arrays onder (rows for children, delta_i, split value, col will be added later)
    best_array = np.zeros((split_points.shape[0], x.shape[0] + 3), dtype='float64')
    # print(f"{best_array=}")
    # Stop wanneer de array met split points leeg is
    while split_points.size != 0:
        # Huidige split value
        split_value = split_points[-1]
        # print(split_value)
        # boolean masks om x rows mee te masken/slicen
        # print(best_array.shape)
        # print(best_array)
        # print(f"{best_array[current_row_of_best_array,:-2]=}")

        # class voor beide split kanten
        classes_left, classes_right = y[x > split_value], y[x <= split_value]
        # print(f"{classes_left=} {classes_right=}")

        # Kijk of er genoeg rows in de gesplitte nodes terechtkomen, anders
        # mogen we de split niet toelaten vanwege de minleaf constraint
        # print(f"{classes_left.shape[0]=}")
        if classes_left.shape[0] < minleaf or classes_right.shape[0] < minleaf:
            # Haal de huidige split_point uit split_points
            split_points = split_points[:-1]
            continue

        # delta_i formule, improved by not taking the parent impurity into
        # account, and not making the weighted average but the weigthed sum
        # only (thanks Lonnie)
        delta_i = (
            impurity(classes_left) * classes_left.shape[0] +
            impurity(classes_right) * classes_right.shape[0])
        # stop huidige splits in de lijst om best split te berekenen
        # print(f"{np.array((delta_i, mask_left, mask_right, split_value))=}")
        # print(f"{best_array[:, classes_left.shape[0]-1]=}")  
        current_row_of_best_array = split_points.shape[0]-1
        best_array[current_row_of_best_array, :x.shape[0]] = x > split_value
        best_array[current_row_of_best_array,-3:-1] = delta_i, split_value
        # print(best_array)
        # Haal de huidige split_point uit split_points
        split_points = split_points[:-1]

    # Bereken de best split voor deze x col, als er ten minste 1 bestaat die
    # voldoet aan min leaf
    return best_array


def exhaustive_split_search(rows, classes, minleaf):
    """
    @todo: Docstring for exhaustive_split_search
    """
    # print("\t\t->entering exhaustive split search")
    # We hebben enumerate nodig, want we willen weten op welke col
    # (age,married,house,income,gender) we een split doen
    exhaustive_best_array = []
    # print(f"Rows:\n{rows},\n Classes:\n{classes}")
    for i, col in enumerate(rows.transpose()):
        # calculate the best split for the col that satisfies the minleaf
        # constraint
        best_array_for_col = bestsplit(col, classes, minleaf)
        # add col number to rows
        best_array_for_col[:,-1] = i
        exhaustive_best_array.append(best_array_for_col)
    exhaustive_best_array = np.concatenate(exhaustive_best_array)
    # print("The array with exhaustive splits is\n", exhaustive_best_array)
    # print("\t\t->returning from exhaustive split search")
    return exhaustive_best_array


def add_children(node, best_split):
    """
    @todo: Docstring for add_children
    """
    # print("\t\t\t->entering add children")
    # The mask that was used to get the rows for the current node from x, we
    # need this to update the rows for the children
    current_mask = node.split_value_or_rows

    # Give the current node the split_value and col it needs for predictions
    node.split_value_or_rows, node.col = best_split[-2], int(best_split[-1])

    # Updating the row masks to give it to children, keeping numpy dimension consistent
    mask_left, mask_right = update_mask(best_split[:-3], current_mask)

    # Adding the pointer between parent and children
    node.add_split(Node(split_value_or_rows=mask_left), Node(split_value_or_rows=mask_right))
    # print("\t\t\t->children added to node and node list\n")
    return [node.left, node.right]


def update_mask(mask, current_mask):
    """
    Updates the spit bool array from any dimension to an array with length
    equal to the total number of rows in dataset x.
    """
    # print(
    #     "\t\t\t\t->entering update mask to update mask that calculates which rows belong to child"
    # )
    # Copy heb ik gedaan omdat we een slice assignment doen.
    #
    # Het punt van deze functie is dat tijdens het tree growen het aantal rows
    # in een node afneemt. Deze functie update de bool array die gebruikt is om
    # te splitten in bestsplit naar een array met als lengte het totale aantal
    # rows in de data set x. Op deze manier kunnen we in tree grow de totale x
    # splitten. 
    #
    # Een andere optie zou zijn om de rows letterlijk tijdelijk op te slaan in
    # nodes... weet niet wat beter is, het kan best veel geheugen in beslag
    # nemen voor grote dataset neem ik aan... Op de manier hier zijn er altijd
    # maar 1D bool arrays, die dus een lagere dimensionaliteit hebben als x veel columnen/rows heeft.
    current_mask = np.vstack((current_mask, current_mask))
    # print(f"\nBefore update (upper rows goes to the left child, bottom to the right):\n{current_mask}")
    # print(mask)
    # print(mask.astype(bool))
    # print(current_mask[current_mask == True])
    current_mask[0][current_mask[0] == True] = mask.astype(bool)
    current_mask[1][current_mask[1] == True] = np.invert(mask.astype(bool))
    # copy = np.array(current_mask, copy=True)
    # copy[np.where(current_mask)] = mask
    # print(f"After update:\n{current_mask}")
    # print("\t\t\t\t->updated row mask for child node")
    return current_mask[0], current_mask[1]


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
    # De nodelist heeft in het begin alleen een root node met alle rows van x,
    # omdat alle rows in de root in acht worden genomen voor bestsplit berekening.
    #
    # Dit representeren we met een boolean mask over x, met lengte het aantal rows
    # in x en elementen True. Deze boolean mask zullen we repeatedly gebruiken als een
    # mask over x om de rows voor bestsplit op te halen.
    mask = np.full(len(x), True)

    # Het eerste node object moet nu geinstantieerd worden
    root = Node(split_value_or_rows=mask)

    # We instantieren ook gelijk het Tree object
    tr = Tree(root, (nmin, minleaf, nfeat))

    # De eerste nodelist heeft alleen de root node, daarna zijn twee childs,
    # etc. totdat alle splits gemaakt zijn en de lijst leeg is.
    nodelist = [root]

    while nodelist:
        # print("->Taking new node from the node list")
        # Pop de current node uit de nodelist
        node = nodelist.pop()
        # Gebruik de boolean mask van de node om de rows in de node uit x te halen
        # print(node_rows)
        # Gebruik de boolean mask van de node om de classes in de node uit y te halen
        node_classes = y[node.split_value_or_rows]
        # print(node_classes)

        # print(f"{node_classes}, {node_rows}")

        # Test of de node een leaf node is met nmin
        if len(node_classes) < nmin:
            node.is_leaf_node(node_classes)
            print(
                "\t->Leafnode! Node has less rows than nmin, it is a leaf node, continueing to next node"
            )
            continue
        # print("\t->Node has more rows than nmin, it is not a leaf node")

        # Als de node niet puur is, probeer dan te splitten
        if impurity(node_classes) > 0:
            # print("\t->Node is not pure yet starting exhaustive split search")

            # Pick a random set of cols from rows (nfeat, important for forests)
            nfeat_col_choice = np.random.choice(range(x.shape[1]), nfeat, replace=False)
            node_rows = x[node.split_value_or_rows][:,np.sort(nfeat_col_choice)]

            # We gaan exhaustively voor de rows in de node over de cols
            # (age,married,house,income,gender) om de bestsplit te
            # bepalen
            #
            # We geven minleaf als argument omdat:
            # "If the algorithm performs a split, it should be the best split that meets the minleaf constrain"
            #
            # We krijgen een exhaustive lijst terug met splits
            exhaustive_best_array = exhaustive_split_search(
                node_rows, node_classes, minleaf)
            
            # Als de array alleen maar zeroes heeft -> no splits for the minleaf leaf node
            # print(exhaustive_best_array[:,:-1].any())
            # raise SystemExit
            if not exhaustive_best_array[:,:-1].any():
                node.is_leaf_node(node_classes)
                print(
                    "\t\t->Leaf node! No split that fullfils minleaf was found continueing to next node"
                )
                continue
            # print(
            #     "\t->Exhaustive search found a split fulfilling the minleaf rule!"
            # )
            # Hier halen we de beste split, en rows voor de child/split nodes
            # uit de exhaustive best list
            # [(exhaustive_best_array[:,:-3] == 0).sum(axis=1)]
            best_split = exhaustive_best_array[exhaustive_best_array[:,:-3].sum(axis=1) != 0][np.argmin(exhaustive_best_array[:,-3])]
            # print((exhaustive_best_array[:,:-3] == 0).sum(axis=1))
            # print(exhaustive_best_array.shape)
            # print(exhaustive_best_array)
            # print(exhaustive_best_array[:,:-3])
            # print(exhaustive_best_array[:,:-3] == 0)
            # print(exhaustive_best_array[:,:-3].sum(axis=1) != 0)
            # print(exhaustive_best_array[exhaustive_best_array[:,:-3]])
            # print(exhaustive_best_array[exhaustive_best_array[:,:-3].sum(axis=0) == 0].shape)
            # print(exhaustive_best_array[exhaustive_best_array[:,:-3].sum(axis=0) == 0][:,-3].shape)
            # print(best_split)
            # raise SystemExit
            # print(exhaustive_best_array[:,:-1])
            # raise SystemExit
            # print(f"\n######################best split array, the last three element=[delta_i, splitvalue, col/attribute], the rest is boolean include or not include row in child node############################\n\n{best_split}\n\n###############################################################################\n")
            # Hier voegen we de twee nieuwe nodes toe aan de gesplitte "parent"
            nodelist += add_children(node, best_split)
            # print(exhaustive_best_array.shape)
            # node.add_split(left_child_node, right_child_node)
        else:
            node.is_leaf_node(node_classes)
            print("\t\t->Leafnode! The node is already pure, it is necessarily a leaf!")
            continue
    # print(tr)
    return tr


def tree_pred(x=None, tr=None, **defaults) -> np.array:
    """
    @todo: Docstring for tree_pred
    """
    y = tr.predict(x).astype(float)
    nmin, minleaf, nfeat = tr.hyper_params
    print(np.mean(pima_indians[:,-1] == y))
    # print("-"*80+f"\nPrediction parameters:\nnmin={nmin}\nminleaf={minleaf}\nnfeat={nfeat}")
    # print(f"\nFor x rows\n{x}\n\n predicted classes are:\n {y}")
    return y


if __name__ == '__main__':
    # credit_data = np.genfromtxt('./data/credit_score.txt',
    #                             delimiter=',',
    #                             skip_header=True)

    pima_indians = np.genfromtxt('./data/pima_indians.csv',
                                delimiter=',',
                                skip_header=True)
    #### IMPURITY TEST
    # array=np.array([1,0,1,1,1,0,0,1,1,0,1])
    # print(impurity(array))
    # Should give 0.23....

    #### BESTSPLIT TEST
    # print(bestsplit(credit_data[:, 3], credit_data[:, 5]))
    # Should give 36

    #### TREE_GROW TEST
    # tree_grow_defaults = {
    #     'x': credit_data[:, :5],
    #     'y': credit_data[:, 5],
    #     'nmin': 2,
    #     'minleaf': 1,
    #     'nfeat': None # 5, should be 5 for bootstrapping folds
    # }

    # Calling the tree grow, unpacking default as argument
    # tree_grow(**tree_grow_defaults)

    #### TREE_PRED TEST
    # tree_pred_defaults = {
    #     'x': credit_data[:, :5],
    #     'tr': tree_grow(**tree_grow_defaults)
    # }

    # y_pred = tree_pred(x=credit_data[:,:5], tr=tree_grow(x=credit_data[:,0:5], y=credit_data[:,5], nmin=2, minleaf=1, nfeat=5), dataset='credit score')
    # print(credit_data[:,5] == y_pred)

    # predictions = tree_pred(x=pima_indians[:,:8], tr=tree_grow(x=pima_indians[:,:8], y=pima_indians[:,8], nmin=20, minleaf=5, nfeat=pima_indians.shape[1]-1), dataset='pima indians')
    # print(np.mean(pima_indians[:,8] == y_pred))

    # Time profile of credit data prediction with single tree
    # cProfile.run("tree_pred(x=credit_data[:,:5], tr=tree_grow(x=credit_data[:,0:5], y=credit_data[:,5], nmin=2, minleaf=1, nfeat=5), dataset='credit score')", 'restats')

    # Time profile of pima indians data prediction with single tree
    cProfile.run("tree_pred(x=pima_indians[:,:8], tr=tree_grow(x=pima_indians[:,:8], y=pima_indians[:,8], nmin=20, minleaf=5, nfeat=pima_indians.shape[1]-1), dataset='pima indians')", 'restats')


    p = pstats.Stats('restats')
    p.sort_stats(SortKey.TIME)
    p.print_stats()
