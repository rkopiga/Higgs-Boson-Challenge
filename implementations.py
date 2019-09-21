def mean_square_error(y, tx, w):
	e = y - tx@w
	N = tx.shape[0]
	loss = (e.T@e)/(2*N)
	loss_gradient = -(tx.T@e)/N
	return (loss, loss_gradient)
	
	
def mean_absolute_error():
	return None
	

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
	tupple
			(w, loss) the last weight vector of the method, 
			and the corresponding loss value (cost function)
	"""
	
	w_temp = initial_w
	for i in range(max_iters):
		loss, loss_gradient = mean_square_error(y, tx, w_temp)
		w = w_temp - gamma*loss_gradient
	return (w_temp, loss)
	

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
	"""
	Linear regression using stochastic gradient descent.
	"""
	return None
	
	
def least_squares(y, tx):
	"""
	Linear squares regression using normal equations.
	"""
	return None
	

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
