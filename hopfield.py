import numpy as np
import matplotlib.pyplot as plt
import copy


# Patterns and false patterns definition
pattern_for_eight = np.array([
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0]
])
false_pattern_for_eight = np.array([
    [-1, -1, 1, 1, -1],
    [-1, 1, -1, 1, -1],
    [-1, 1, -1, 1, -1],
    [-1, 1, -1, 1, -1],
    [-1, -1, -1, 1, -1]
])
false_pattern_for_eight_all_minus = np.array([
    [-1, -1, -1 - 1 - 1],
    [-1, -1 - 1, -1 - 1],
    [-1, -1 - 1, -1 - 1],
    [-1, -1 - 1, -1 - 1],
    [-1, -1, -1, -1 - 1]
])

pattern_for_three = np.array([
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0]
])

false_pattern_for_three = np.array([
    [-1, 1, -1, 1, -1],
    [-1, -1, -1, -1, -1],
    [-1, 1, 1, 1, -1],
    [-1, -1, -1, 1, -1],
    [-1, 1, -1, 1, -1]
])

pattern_for_x = np.array([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1]
])
false_pattern_for_x = np.array([
    [-1, -1, -1, -1, 1],
    [-1, 1, -1, 1, -1],
    [-1, -1, -1, -1, -1],
    [-1, 1, -1, 1, -1],
    [-1, -1, -1, -1, -1]
])
test_pattern = np.array([
    [1, 0],
    [0, 1]
])


# false_pattern = np.array([
#     [-1, -1],
#     [-1, 1]
# ])

# This method will flatten given  pattern and afterwards zeroes in given pattern will be changed to -1
def zeroOutPatterns(pattern):
    pattern_t = pattern.flatten()
    for i in range(len(pattern_t)):
        if pattern_t[i] == 0:
            pattern_t[i] = -1
    return pattern_t


# This method computes the outer product of flattened pattern and its transpose, then it puts 0 to diagonal
def createWeightedMatrixWithZeroes(pattern_T):
    weighted_matrix = np.outer(pattern_T, pattern_T.T)
    np.fill_diagonal(weighted_matrix, 0)
    return weighted_matrix


# Synchronous recovery of given pattern from weighted matrix, computes dot product of flattened input pattern
# with weighted matrix, reshapes it by size of input_pattern, then changes the result by signum function,
# afterwards replaces -1 by 0
def synchronous_recovery(weighted_matrix, input_pattern):
    # return np.dot(weighted_matrix, input_pattern)
    recoverered_pattern = np.vectorize(np.sign)(
        np.reshape(np.dot(input_pattern.flatten(), weighted_matrix),
                   (len(input_pattern), len(input_pattern))))
    recoverered_pattern[recoverered_pattern == -1] = 0
    return recoverered_pattern

# Asynchronous recovery of given pattern from weighted matrix, firstly flattens given pattern, then in for loop
# for each element of flattened pattern is calculated new value that is result of dot product of flattened pattern
# and one column of weighted matrix, then the value is changed by signum function and assigned to recovered_pattern
# at the same position as position of flattened pattern, finally result is reshaped by sizes of input_pattern
def asynchronous_recovery(weighted_matrix, input_pattern):
    recoverered_pattern = copy.deepcopy(input_pattern.flatten())
    for i in range(len(recoverered_pattern)):
        sign_number = np.sign(
            (np.dot(input_pattern.flatten(), weighted_matrix[i])))
        recoverered_pattern[i] = sign_number

    return recoverered_pattern.reshape(len(input_pattern), len(input_pattern))

# Processing given patterns to remember
zeroedPattern = zeroOutPatterns(pattern_for_three)
w_matrix = createWeightedMatrixWithZeroes(zeroedPattern)

zeroedPattern2 = zeroOutPatterns(pattern_for_x)
w_matrix2 = createWeightedMatrixWithZeroes(zeroedPattern2)

zeroedPattern3 = zeroOutPatterns(pattern_for_eight)
w_matrix3 = createWeightedMatrixWithZeroes(zeroedPattern3)

# created matrix with all pattern to remember by their sum
sum_of_weighted_matrix = (w_matrix + w_matrix2 + w_matrix3)

# calling of each methods of recovery
# recovery_result = synchronous_recovery(sum_of_weighted_matrix, false_pattern_for_x)
recovery_result = asynchronous_recovery(sum_of_weighted_matrix, false_pattern_for_x)

# visualisation of input pattern and recovered pattern
plt.matshow(false_pattern_for_x, fignum=1, cmap=plt.cm.viridis)
plt.matshow(recovery_result, fignum=100, cmap=plt.cm.viridis)

plt.show()
