import numpy as np

def mean_squared_error(y, t):
	return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y,t):
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)

	delta = 1e-7
	batch_size = y.shape[0]

	return -np.sum(t*np.log(y+delta)) / batch_size

def numerical_diff(f, x):
	h = 1e-4
	return (f(x+h)-f(x-h))/(2*h)

def numercial_gradient(f, x):
	h = 1e-4
	grad = np.zeros_like(x)

	for idx in range(x.size):
		tmp_val = x[idx]
		x[idx] = tmp_val + h
		fxh1 = f(x)

		x[idx] = tmp_val - h
		fxh2 = f(x)

		grad[idx] = (fxh1 - fxh2)/(2*h)
		x[idx] = tmp_val

	return grad

def gradient_descent(f, init_x, lr , step_num):
	x = init_x
	x_history = []

	for i in range(step_num):
		x_history.append(x.copy())
		grad = numercial_gradient(f, x)
		x -= lr*grad
	return x , np.array(x_history)