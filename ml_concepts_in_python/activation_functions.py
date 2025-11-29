import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    arr = np.array(x, dtype=np.float64)
    return 1/(1 + np.exp(-arr))

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    if len(x.shape)>1:
        axis=1
    else:
        axis=0
    numerator = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return numerator/np.sum(numerator, axis=axis, keepdims=True)

def tanh(x):
    """(e^x - e^-x)/(e^x + e^-x)"""
    np_x = np.array(x, dtype=np.float64)
    pos_exp_x = np.exp(np_x)
    neg_exp_x = np.exp(-np_x)
    numerator = pos_exp_x - neg_exp_x
    denominator = pos_exp_x + neg_exp_x
    
    tanh_ = numerator/denominator
    if type(tanh_) == np.float64:
        return np.array([tanh_.item()])
    else:
        return tanh_
    
def relu(x):
    """
    Implement ReLU activation function.
    """
    arr = np.array(x)
    new_arr = np.maximum(arr, 0)
    return new_arr