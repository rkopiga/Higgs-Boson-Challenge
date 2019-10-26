import numpy as np

import proj1_helpers as helper


def mean_square_error(y, tx, w):
    e = y - tx @ w
    N = tx.shape[0]
    loss = (e.T @ e) / (2 * N)
    loss_gradient = -(tx.T @ e) / N
    return loss, loss_gradient


def mean_absolute_error(y, tx, w):
    """
    Mean absolute error (MAE) cost function.
    Pros: convex,
    Cons:

    Parameters
    ----------
    y: vector
        The outputs
    tx: vector
        The inputs
    w: vector
        The weight vector

    Returns
    -------
    (loss, loss_gradient):
        TODO
    """
    e = y - tx @ w
    N = len(tx)
    loss = np.sum(np.absolute(e)) / N
    loss_gradient = -1/N*(tx.T@np.sign(e))
    return loss, loss_gradient


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss = None
    for i in range(max_iters):
        loss, loss_gradient = mean_square_error(y, tx, w)
        w = w - gamma * loss_gradient
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w_temp = initial_w
    batch_size = 1
    loss = None
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            gradient, _ = compute_stoch_gradient(y_batch, tx_batch, w_temp)
            w_temp = w_temp - gamma * gradient
            loss, _ = mean_square_error(y, tx, w_temp)
    w = w_temp
    return w, loss


def compute_stoch_gradient(y, tx, w):
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad, e


def least_squares(y, tx):
    tx_T = tx.T
    w = np.linalg.inv(tx_T @ tx) @ tx_T @ y
    loss, _ = mean_square_error(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.
    
    Parameters
    ----------
    y: vector
        The outputs
    tx: vector
        The inputs
    lambda_: scalar
        The regularizer

    Returns
    -------
    (w, loss):
        the last weight vector of the method, 
        and the corresponding loss value (cost function)
    """
    gamma_prime = 2 * tx.shape[0] * lambda_
    additional_matrix = np.identity(tx.shape[1]) * gamma_prime
    tx_T = tx.T
    w = np.linalg.inv(tx_T @ tx + additional_matrix) @ tx_T @ y
    loss, _ = mean_square_error(y, tx, w)
    return w, loss


def logistic_function(z):
    """
    Computes logistic function of scalar or array
    """
    logistic_value = np.exp(z)/(1 + np.exp(z))
    return logistic_value


def logistic_loss(y, tx, w, lambda_):
    """
    Computes loss using logistic cost function with regularizer
    
    Parameters
    ----------
    y: vector
        The outputs
    tx: vector
        The inputs
    w: vector
        Vector of weights
    lambda_:
        define value of regularizer in cost function
    
    Returns
    -------
    value of cost function: scalar
    """
    return 1 / len(y) * np.sum(np.log(1 + np.exp(tx @ w)) - y @ (tx @ w)) + lambda_ * np.linalg.norm(w) ** 2


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, decreasing_gamma=False):
    """
    Logistic regression using gradient descent
    
    Parameters
    ----------
    y: vector
        The outputs
    tx: vector
        The inputs
    lambda_:
        define value of regulator in cost function
    initial_w: vector
        Initial value of weights
    max_iters: scalar
        maximum number of iteration
    gamma: scalar
        define step size of gradient descent
    
    
    Returns
    -------
    (w, loss):
        the last weight vector of the method, 
        and the corresponding loss value (cost function)
    
    Recommended values: gamma = 0.05, max_iters = 20-25, regulator = 1 with standardized data
    """
    y = (y+1)/2
    w = initial_w
    for i in range(max_iters):
        loss_gradient = 1 / len(y) * tx.T @ (logistic_function(tx@w) - y) + 2 * lambda_ * w
        if decreasing_gamma:
            w = w - helper.gamma(i) * loss_gradient
        else:
            w = w - gamma * loss_gradient
    loss = logistic_loss(y, tx, w, lambda_)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, decreasing_gamma=False):
    """
    Logistic regression using stochastic gradient descent
    
    Parameters
    ----------
    y: vector
        The outputs
    tx: vector
        The inputs
    initial_w: vector
        Initial value of weights
    max_iters: scalar
        maximum number of iteration
    gamma: scalar
        define step size of gradient descent
    
    Returns
    -------
    (w, loss):
        the last weight vector of the method, 
        and the corresponding loss value (cost function)
    
    Recommended values: gamma = 0.5, max_iters = 40 with standardized data
    """

    w_temp = initial_w
    for i in range(max_iters):
        random_data_row = np.random.randint(low=0, high=tx.shape[0], size=1)
        tx_n = tx[random_data_row]
        y_n = y[random_data_row]
        loss_gradient = tx_n * (logistic_function(np.dot(tx_n, w_temp.T)) - y_n)
        if decreasing_gamma:
            w_temp -= helper.gamma(i) * loss_gradient
        else:
            w_temp -= gamma * loss_gradient
    w = w_temp
    loss = logistic_loss(y_n, tx_n, w.T, lambda_=0)
    return w, loss


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
