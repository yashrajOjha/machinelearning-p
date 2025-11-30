import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Loss Equation is 1/m * ∑(y-y_preds)^2 -> where y_preds is mx+c,
    So we do chain rule derivation to get the derivative
    Derivative is -2/m * X.T @ (y - y_preds)
    Removing the - factor -2/m*(-1)*X.T @ (y_preds-y)
    Equation becomes 2/m*X.T @ (y_preds-y)
    We can incorporate the 2 with learning rate
    """
    m, n = X.shape
    y = y.reshape(-1, 1)
    theta = np.zeros((n, 1))

    for i in range(iterations):
        y_preds = X @ theta
        gradients = 1/m * (X.T @ (y_preds-y))
        theta -= alpha*gradients

    return np.round(theta.flatten(), 4) 