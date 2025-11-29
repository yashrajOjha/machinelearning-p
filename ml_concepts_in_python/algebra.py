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

def cosine_similarity(v1, v2):
    numr = np.dot(v1, v2)
    denor = np.sqrt(np.sum(v1**2))*np.sqrt(np.sum(v2**2))
    return round(numr/denor, 3)


def f_score(y_true, y_pred, beta):
    tp = np.sum((y_pred==1) & (y_true==1))
    fp = np.sum((y_pred==1) & (y_true==0))
    fn = np.sum((y_pred==0) & (y_true==1))
    tn = np.sum((y_pred==0) & (y_true==1))
    mf = 1 + (beta**2)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    numerator = mf*precision*recall
    denominator = (beta**2)*precision + recall
    return round(numerator/denominator, 3)