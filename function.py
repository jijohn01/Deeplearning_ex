import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def softmax(a):
	c = np.max(a)# 오버플로우에 대한 대책 >> c 
	exp_a = np.exp(a - c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a

	return y
