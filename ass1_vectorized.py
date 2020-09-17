import time
import numpy as np

credit_data = np.genfromtxt('./credit_score.txt',
                            delimiter=',',
                            skip_header=True)

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


class Tree():
    """
    @todo: docstring for Tree
    """
    def __init__(self, tree_vec_of_tuples):
        """@todo: Docstring for init method.

        /root_node_obj/ @todo

        """
        self.tree_vec_of_tuples = tree_vec_of_tuples
        self.leaf_nodes = np.where(tree_vec_of_tuples[:,1] == -1)
        self.classes = [(1, -1), (1, -1)]

    def drop(self, y):
        """
        @todo: Docstring for drop
        """
        return y * 100
        
    def leaf(self, y):
        """
        @todo: Docstring for drop
        """
        return y
        




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
        # Prevents division by zero, when the potential split does not have any rows
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


def bestsplit(x, y) -> int:
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
    if len(x_sorted) <= 2:
        # Allows splitting on categorical (0 or 1) cols
        split_points = [0.5]
    else:
        # Take average between consecutive numerical rows in the x col
        split_points = (x_sorted[:len(x_sorted) - 1] + x_sorted[1:]) / 2

    # De toepassing van bestsplit verdeelt de x col vector in tweeen, twee
    # arrays van "x rows". Deze moeten we terug krijgen om de in de child nodes
    # bestsplit toe te passen.
    #
    # Deze lus berekent de best split value, en op basis daarvan weten we welke
    # twee "x rows" arrays we moeten returnen, en welke split value het beste
    # was natuurlijk.
    best_delta_i = None
    for split in split_points:
        # np.index_exp maakt een boolean vector die zegt welke elementen in de
        # col van x hoger of lager zijn dan split
        col_slice_boolean_matrices = {
            "left": np.index_exp[x > split],
            "right": np.index_exp[x <= split]
        }

        # delta_i formule met de boolean vector van hierboven
        delta_i = impurity(
            y) - (len(y[col_slice_boolean_matrices["left"]]) *
                  impurity(y[col_slice_boolean_matrices["left"]]) +
                  len(y[col_slice_boolean_matrices["right"]]) *
                  impurity(y[col_slice_boolean_matrices["right"]])) / len(y)

        print(f"{split=}, {delta_i=}")
        #
        if best_delta_i is not None:
            if delta_i > best_delta_i:
                best_delta_i, best_split, best_col_slice_boolean_matrices = delta_i, split, col_slice_boolean_matrices
        else:
            best_delta_i, best_split, best_col_slice_boolean_matrices = delta_i, split, col_slice_boolean_matrices
    return best_delta_i, best_split, best_col_slice_boolean_matrices

def tree_example(x=None, tr=None, **defaults) -> None:
    """
    @todo: Docstring for tree_example
    """
    tree_vec = []
    print(tree_vec)
    print(type(tree_vec))
    tree_vec.append((36,3))
    print(tree_vec)
    tree_vec.append((0,3))
    print(tree_vec)
    tree_vec.append((1, -1))
    print(tree_vec)
    print(tree_vec[0])
    print(type(tree_vec[0]))
    tree_vec = np.array(tree_vec) # , dtype=(int, 2))
    print(tree_vec)
    print(type(tree_vec[0]))

    tree = Tree(tree_vec)
    
    # Let's show how to predict
    # 1. maak een vector met root node voor elke row in x waarvoor je een class
    # wil predicten.
    y = np.ones(len(x), dtype=int)
    print(y)
    print(type(y))
    print(y.shape)
    # 2. Herinner recurrence relatie van whatsapp, en pas hem toe met
    # tree.drop(x) op nodes die geen leaf node zijn
    # 
    # Returns indices where not leaf node
    print(tree.leaf_nodes)
    # print(tree.classes)
    # leafs = np.searchsorted(
    # print(leafs)
    y = np.where(np.searchsorted(tree.leaf_nodes, y))

    # y = y[:,0]
    # print(y)



#
#
# Put all helper functions above this comment!


def tree_grow(x=None,
              y=None,
              n_min=None,
              min_leaf=None,
              n_feat=None,
              **defaults) -> Tree:
    """
    @todo: Docstring for tree_grow
    """
    # Voor de lus die onze tree growt instantieren we een list die tuples als
    # elementen zal hebben, het grote voordeel is dat we op deze manier vector
    # operaties meerdere parallele
    # prediction drops kunnen doen. (Je kan bijvoorbeeld geen object methodes
    # broadcasten als je een numpy array van node objecten hebt)
    #
    # De tuple moet uiteindelijk de informatie bevatten om voor een row in x
    # een class te voorspellen. Hier hebben we voor nodig:
    # 
    # 1. Het split value, voor lager of hoger test
    # 2. Het col nummer waar de split bij hoort, anders weten we niet waar we op testen
    #
    # (split, col)
    #
    # De enige uitzondering hierop zijn leaf nodes. Om de tree data structure
    # een numpy array te maken moeten dit ook tuples zijn. Dit lossen we op
    # door een negatieve col aan te duiden. Dit zorgt ervoor dat de prediction
    # functie hier eindigt.
    #
    # (class, negative_int: -1)
    #
    # Checkout tree example function for more info
    tree_vec = []
    # De nodelist heeft in het begin alleen de alle rows van x, omdat alle rows
    # altijd in de root in acht worden genomen.
    #
    # Dit representeren we met een boolean vector, met lengte het aantal rows in x en elementen True.
    rows = np.full((1,len(x)), True)
    nodelist = [rows]

    # tree_array = np.empty 
    # while nodelist:
    #     current_node = nodelist.pop()
    #     slices = current_node.value
    #     node_classes = y[slices]
    #     # print(node_classes)

    #     # f'Current node will be leaf node if (( (number of data "tuples" in child node) < {n_min=} )) \n'
    #     # put stopping rules here before making a split
    #     if len(node_classes) < n_min:
    #         current_node.value = Leaf(
    #             np.argmax(np.bincount(node_classes.astype(int))))
    #         print(f"leaf node has majority clas:\n{current_node.value.value=}")
    #         continue

    #     if impurity(node_classes) > 0:
    #         # print(
    #         #     f"Exhaustive split search says, new node will check these rows for potential spliterinos:\n{x[slices]}"
    #         # )

    #         # If we arrive here ever we are splitting
    #         # bestsplit(col, node_labels) ->
    #         # {"slices": list[int], "split": numpyfloat, "best_delta_i": numpyfloat}

    #         # slices (list) used for knowing which rows (int) to consider in a node
    #         # best_split saved in current_node.value
    #         # best_delta_i used to find best split among x_columns
    #         best_dict = None
    #         for i, x_col in enumerate(x[slices].transpose()):
    #             print(
    #                 "\nExhaustive split search says; \"Entering new column\":")
    #             col_split_dict = bestsplit(x_col, node_classes, slices)

    #             if best_dict is not None:
    #                 if col_split_dict["delta_i"] > best_dict["delta_i"]:
    #                     best_dict = col_split_dict
    #                     best_dict["col"] = i
    #             else:
    #                 best_dict = col_split_dict
    #                 best_dict["col"] = i
    #         print("\nThe best split for current node:", best_dict)

    #         # Here we store the splitted data into Node objects
    #         current_node.value = best_dict["split"]
    #         current_node.col = best_dict["col"]
    #         # Split will not happen if (( (number of data "tuples" potential split) < {min_leaf=} ))\n'
    #         if min([len(x) for x in best_dict["slices"].values()]) < min_leaf:
    #             continue
    #         else:
    #             # Invert left and right because we want left to pop() first
    #             children = [
    #                 Node(value=best_dict["slices"]["right"]),
    #                 Node(value=best_dict["slices"]["left"])
    #             ]
    #             current_node.add_split(children[1], children[0])
    #             nodelist += children
    #     else:
    #         current_node.value = Leaf(
    #             np.argmax(np.bincount(node_classes.astype(int))))
    #         print(
    #             f"\n\nLEAF NODE has majority clas:\n{current_node.value.value=}"
    #         )
    #         continue
    # return tree


def predict(x, nodes) -> list:
    """
    @todo: Docstring for predict
    """
    # which row to drop
    # print(x)
    drop = 0
    while not set(nodes).issubset({0, 1}):
        print(nodes)
        # print(x[drop])
        if isinstance(nodes[drop].value, Leaf):
            nodes[drop] = nodes[drop].value.value
            drop += 1
            continue

        print(nodes[drop].value)
        print(nodes[drop].col)
        # print(nodes[drop].col)
        if x[drop, nodes[drop].col] > nodes[drop].value:
            nodes[drop] = nodes[drop].left
        else:
            nodes[drop] = nodes[drop].right
    return np.array(nodes)


def tree_pred(x=None, tr=None, **defaults) -> np.array:
    """
    @todo: Docstring for tree_pred
    """
    nodes = [tr.tree] * len(x)
    # y = np.linspace(0, len(x), 0)
    # y = np.array(ele)
    y = predict(x, nodes)
    print(f"\n\nPredicted classes for {x=}\n\n are: {y=}")
    return y


if __name__ == '__main__':
    #### IMPURITY TEST
    # array=np.array([1,0,1,1,1,0,0,1,1,0,1])
    # print(impurity(array))
    # Should give 0.23....

    #### BESTSPLIT TEST
    # print(bestsplit(credit_data[:, 3], credit_data[:, 5]))
    # Should give 36

    #### TREE_GROW TEST
    tree_grow_defaults = {
        'x': credit_data[:, :5],
        'y': credit_data[:, 5],
        'n_min': 2,
        'min_leaf': 1,
        'n_feat': 5
    }

    # Calling the tree grow, unpacking default as argument
    # tree_grow(**tree_grow_defaults)

    #### TREE_PRED TEST
    tree_pred_defaults = {
        'x': credit_data[:, :5],
        # 'tr': tree_grow(**tree_grow_defaults)
    }

    tree_example(**tree_pred_defaults)
    # tree_pred(**tree_pred_defaults)

# start_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))
