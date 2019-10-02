import numpy as np

def mean_square_error(y, tx, w):
	e = y - tx@w
	N = tx.shape[0]
	loss = (e.T@e)/(2*N)
	loss_gradient = -(tx.T@e)/N
	return (loss, loss_gradient)
	
	
def mean_absolute_error():
	e = y - tx@w
	N = tx.shape[0]
	loss = np.sum(np.absolute(e))/N
	loss_gradient = None
	return (loss, gradient)
	

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
	"""
	Linear regression using gradient descent.
	"""
	w_temp = initial_w
	for i in range(max_iters):
		loss, loss_gradient = mean_square_error(y, tx, w_temp)
		w_temp = w_temp - gamma*loss_gradient
	return (w_temp, loss)
	

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
	"""
	Linear regression using stochastic gradient descent.
	"""
    w_temp = initial_w
    for i in range(max_iters):
        
        random_data_row = np.random(range(tx.shape[0]))
        e = (y - tx@w_temp)[random_data_row]
        Ln_grad = -2*e*tx[random_data_row]
        w_temp = w_temp - gamma * Ln_grad
    loss, _ = mean_square_error(y, tx, w_temp)
    w = w_temp
	return (w, loss)
	
	
def least_squares(y, tx):
	"""
	Linear squares regression using normal equations.
	"""
    tx_T = tx.T
    w = linalg.inv(tx_T @ tx) @ tx_T @ y   " ??? Verifiy l'inveritibility ???
    loss, loss_gradient = mean_square_error(y, tx, w)
	return (w, loss)
	

def ridge_regression(y, tx, lamda_):
	"""
	Ridge regression using normal equations.
	"""
    gamma_prime = 2 * tx.shape[0] * lambda_
    additional_matrix = np.ones(tx.shape[1])
    tx_T = tx.T
    w = linalg.inv(tx_T @ tx + additional_matrix) @ tx_T @ y
    loss, loss_gradient = mean_square_error(y, tx, w)
	return (w, loss)

def logistic_regression_GD(y, tx, initial w, max_iters, gamma):
    """
    Logistic regression using gradient descent or SGD
    """
    w_temp = initial_w
    for i in range(max_iters):
        loss_gradient = tx.T @ (logistic_function(tx@w) - y)
        w_temp = w_temp - gamma * loss_gradient
    w = w_temp
    loss, _ = mean_square_error(y, tx, w_temp)
    return (w,loss)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
	"""
	Regularized logistic regression using gradient descent.
	"""
    w_temp = initial_w
    for i in range(max_iters):
        random_data_row = np.random(range(tx.shape[0]))
        tx_n = tx[random_data_row]
        y_n = y[random_data_row]
        loss_gradient = tx_n * (logistic_function(np.dot(tx_n,y_n)) - y_n)
        w_temp =  w_temp - gamma * loss_gradient
    w = w_temp    
    loss, _ = mean_square_error(y, tx, w)
	return (w,loss)

def logistic_function(z):
    return np.exp(z)/(1 + np.exp(z))


