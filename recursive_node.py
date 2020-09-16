import numpy as np


class Node():
    """
    @todo: docstring for Node
    """
    def __init__(self, value=None):
        """@todo: Docstring for init method.

        /value=None/ @todo

        """
        self.value = value

    def add_split(self, left, right):
        """
        @todo: Docstring for add_split
        """
        self.left = left
        self.right = right


class Tree():
    """
    @todo: docstring for Tree
    """
    def __init__(self, root_node_obj):
        """@todo: Docstring for init method.

        /root_node_obj/ @todo

        """
        self.tree = root_node_obj

    def __repr__(self):
        nodelist = [self.tree]
        tree_str = ''
        while nodelist:
            current_node = nodelist.pop()
            # print(current_node.value)
            try:
                childs = [current_node.right, current_node.left]
                nodelist += childs
            except AttributeError:
                pass
            tree_str += current_node.value
        return tree_str


n1 = Node(value="root\n")
n2 = Node(value="left child of n1, ")
n3 = Node(value="right child of n1")

n1.add_split(n2, n3)

my_tree = Tree(n1)
print(my_tree)
