import numpy as np

def normalization(matrix, axis=None, norm_type='l2'):
    np_matrix = np.array(matrix)
    if norm_type=='l2':
        nf = np.sqrt(np.sum(np.power(np_matrix, 2), axis=axis, keepdims=True))
        return np.divide(np_matrix, nf)
    if norm_type=='l1':
        nf = np.sum(np.abs(np_matrix), axis=axis, keepdims=True)
        return np.divide(np_matrix, nf)
    if norm_type=='max':
        nf = np.max(np.abs(np_matrix), axis=axis, keepdims=True)
        return np.divide(np_matrix, nf)