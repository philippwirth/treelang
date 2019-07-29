import torch
import torch.nn as nn

def log_sigmoid(x):
	# input: x of shape (nsamples + 1) x n
	# output: loss (i.e. mean of pos - neg)

	y = torch.nn.functional.logsigmoid(x)
	return (-y[0] + y[1:].sum(0)).mean()

def log_softmax(x):

	# input: x of shape (nsamples + 1) x n
	# output: loss (i.e. mean of pos - neg)

	return -torch.nn.functional.log_softmax(x, dim=0)[0].mean()