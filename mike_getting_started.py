import numpy as np

credit_data = np.genfromtxt('./credit_score.txt', delimiter=',', skip_header=True)

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

# print(credit_data)

first_row = credit_data[0]
# print('The first row: ', first_row)

fourth_col = credit_data[:,3]
# print('The fourth column: ', fourth_col)

# print(credit_data[...,1:])

# print(credit_data[:3,2])

# print(np.sort(np.unique(credit_data[:,3])))

# print('Total number of examples with binary label 1:', np.sum(credit_data[:,5]))

# print('Sum of all entries in the cols:', credit_data.sum(axis=0))

# print('Select all rows where the first column is bigger than 27:', credit_data[credit_data[:,0] > 27])

x = np.array([2,5,10])
# print(x)

# print(np.arange(0, 10))

# Select the *row numbers* of the rows where the first column of credit_data is bigger than 27:
# print(np.arange(0,10)[credit_data[:,0] > 27])

# Draw a random sample of size 5 from the numbers 1 through 10 (without replacement):
index = np.random.choice(np.arange(0,10), size=5, replace=False)
# print(index)

train = credit_data[index,]
# print(train)

test = np.delete(credit_data, index, axis=0)
# print(test)

# help(np.random.choice)

# Practice exercise1

# test_array = credit_data[:,-1]
test_array = np.array([1,0,1,1,1,0,0,1,1,0,1])
def impurity(array) -> None:
    """
    @todo: Docstring for 
    """
    # print(array)
    rel_freq_1_len = len(test_array[0:])
    print('len of the vector:', rel_freq_1_len)
    rel_freq_1_sum = test_array[0:].sum()
    print(rel_freq_1_sum)
    rel_freq_1 = rel_freq_1_sum / rel_freq_1_len
    rel_freq_0 = 1 - rel_freq_1
    print('\nThe rel. freq. of 1: ', rel_freq_1)
    print('\nThe rel. freq. of 0: ', rel_freq_0)
    gini_index = rel_freq_1 * rel_freq_0
    print('\nThe gini index: ', gini_index)
    # pass

impurity(test_array)    
