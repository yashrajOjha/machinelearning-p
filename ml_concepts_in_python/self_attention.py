import numpy as np

def softmax(x):
	max_x = np.max(x, axis=1, keepdims=True)
	e_x = np.exp(x-max_x)
	return e_x/np.sum(e_x, axis=1, keepdims=True)

def compute_qkv(X, W_q, W_k, W_v):
	Q = np.dot(X, W_q)
	K = np.dot(X, W_k)
	V = np.dot(X, W_v)
	return Q, K, V
	
def self_attention(Q, K, V):
    K_t = np.transpose(K)
    numr = np.dot(Q, K_t)
    sqrt_k = np.sqrt(Q.shape[-1])
    scores = softmax(numr/sqrt_k)
    attention_output = np.dot(scores, V)
    return attention_output