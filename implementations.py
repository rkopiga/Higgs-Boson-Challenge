import numpy as np

def mean_square_error(y, tx, w):
	"""
	Mean square error (MSE) cost function.
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
	e = y - tx@w
	N = tx.shape[0]
	loss = (e.T@e)/(2*N)
	loss_gradient = -(tx.T@e)/N
	return (loss, loss_gradient)
	
	
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
	e = y - tx@w
	N = tx.shape[0]
	loss = np.sum(np.absolute(e))/N
	loss_gradient = None  # TODO
	return (loss, loss_gradient)
	

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
	"""
	Linear regression using gradient descent.
	
	Parameters
	----------
	y: vector
		The outputs
	tx: vector
		The inputs
	initial_w: vector
		The initial weight vector
	max_iters: int
		The number of steps to run
	gamma: double
		The step-size (>0)
		
	Returns
	-------
	(w, loss):
		the last weight vector of the method, 
		and the corresponding loss value (cost function)
	"""
	
	w = initial_w
	for i in range(max_iters):
		loss, loss_gradient = mean_square_error(y, tx, w)
		w = w - gamma*loss_gradient
	return (w, loss)
	

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
	"""
	Linear regression using stochastic gradient descent.
	
	Parameters
	----------
	y: vector
		The outputs
	tx: vector
		The inputs
	initial_w: vector
		The initial weight vector
	max_iters: int
		The number of steps to run
	gamma: double
		The step-size (>0)
		
	Returns
	-------
	(w, loss):
		the last weight vector of the method, 
		and the corresponding loss value (cost function)
	"""
    w_temp = initial_w
    for i in range(max_iters):
        loss, loss_gradient = mean_square_error(y, tx, w_temp)
        random_data_row = np.random(range(tx.shape[0]))
        e = (y - tx@w_temp)[random_data_row]
        Ln_grad = -2*e*tx[random_data_row]
        w_temp = w_temp - gamma * Ln_grad
	return (w_temp, loss)
	
	
def least_squares(y, tx):
	"""
	Linear squares regression using normal equations.
	
	Parameters
	----------
	y: vector
		The outputs
	tx: vector
		The inputs
		
	Returns
	-------
	(w, loss):
		the last weight vector of the method, 
		and the corresponding loss value (cost function)
	"""
    tx_T = tx.T
    w = linalg.inv(tx_T @ tx) @ tx_T @ y   " ??? Verifiy l'inveritibility ???
    loss, loss_gradient = mean_square_error(y, tx, w)
	return (w, loss)
	

def ridge_regression(y, tx, initial_w, max_iters, gamma):
	"""
	Ridge regression using normal equations.
	"""
	return None
	

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
	"""
	Regularized logistic regression using gradient descent.
	"""
	return None
