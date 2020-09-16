import numpy as np
import random
import math
from copy import deepcopy

credit_data = np.genfromtxt('/Users/mikevink/Documents/python/2020_data_mining_assignments/credit_score.txt', delimiter=',', skip_header=True)

#print(credit_data)
#print(credit_data[0])
#print(credit_data[:,3])
#print(credit_data[4,0])
#print(np.sort(np.unique(credit_data[:,3]))) #Give the distinct values of income, sorted from low to high
#print(np.sum(credit_data[:,5]))
#print(credit_data.sum(axis=0)) #Add the entries of each column of credit_data
#print(credit_data.sum(axis=1)) #Add the entries of each row
#print(credit_data[credit_data[:,0] > 27]) # Select all rows where the first column is bigger than 27
#
#x = np.array([2, 5, 10])
#print(x)
#print(np.arange(0, 10))
#
#print(np.arange(0, 10)[credit_data[:,0] > 27]) #Select the *row numbers* of the rows where the first column of credit_data is bigger than 27
#
#index = np.random.choice(np.arange(0, 10), size=5, replace=False) #Draw a random sample of size 5 from the numbers 1 through 10 (without replacement)
#print(index)
#train = credit_data[index,]
#print(train)
#test = np.delete(credit_data, index, axis=0) #Select all rows with row number not in "index"
#print(test)
#
#print(random.choice(train))


### Practice exercise 1 ###
def impurity(vector): # vector = list of 0s and 1s
    num_of_class_labels = len(vector)
    num_of_class_1 = sum(vector)
    num_of_class_0 = num_of_class_labels - num_of_class_1
    return (num_of_class_0 / num_of_class_labels) * (num_of_class_1 / num_of_class_labels)

array=np.array([1,0,1,1,1,0,0,1,1,0,1])
print(impurity(array))


### Practice exercise 2 ###
def bestsplit(x, y): # x = numeric values; y = class labels
    x_sorted = np.sort(np.unique(x))
    split_points = (x_sorted[:len(x_sorted)-1] + x_sorted[1:]) / 2
    
    best_impurity_after_split = math.inf
    for split in split_points:
        impurity_after_split = impurity(y[x <= split]) + impurity(y[x > split])
        if impurity_after_split < best_impurity_after_split:
            best_split = split
            best_impurity_after_split = impurity_after_split

    return best_split

print(bestsplit(credit_data[:,3], credit_data[:,5]))



class Node:
    def _init_(self):
        self.left  = None
        self.right = None
        self.split_value = None
        
class Leaf:
    def __init__(self, predicted_class: int):
        self.predicted_class = predicted_class


def tree_grow(x, y): # x = numeric values; y = class labels
    root = Node()
    root.split_value = bestsplit(x, y)
    root.left = Leaf(0)
    root.right = Leaf(1)
    return root
    
def tree_pred(x, tr):
    y = []
    for value in x:
        y.append(single_value_pred(value, tr))  
    return y

def single_value_pred(value, current_tree):
    if isinstance(current_tree, Leaf):
        return current_tree.predicted_class
    else:
        if value <= current_tree.split_value:
            return single_value_pred(value, current_tree.left)
        else:
            return single_value_pred(value, current_tree.right)        

tree = tree_grow(credit_data[:,3], credit_data[:,5])
print(tree_pred([32, 38, 3, 40], tree))



