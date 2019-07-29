import torch
import torch.nn as nn
import numpy as np


class LinearRNNCell(nn.RNN):

	def __init__(self, ninp, nhid, dropout=0):
		super(LinearRNNCell, self).__init__(ninp, nhid, 1, dropout=dropout)

	def forward(self, input_, h_0):

		seq_len, bsz, ninp = input_.size()
		in_times_W = torch.nn.functional.linear(input_, self.weight_ih_l0, self.bias_ih_l0)
	
		h = h_0
		output = []
		for t in range(seq_len):

			h_times_U = torch.nn.functional.linear(h, self.weight_hh_l0, self.bias_hh_l0)
			output.append(in_times_W[t] + h_times_U)
			h = output[-1]

		return torch.cat(output, 1), None


class ELURNNCell(nn.RNN):

	def __init__(self, ninp, nhid, dropout=0, alpha=1.0):

		super(ELURNNCell, self).__init__(ninp, nhid, 1, dropout=dropout)
		self.elu = nn.ELU(alpha=alpha)

	def forward(self, input_, h_0):

		seq_len, bsz, ninp = input_.size()
		in_times_W = torch.nn.functional.linear(input_, self.weight_ih_l0, self.bias_ih_l0)
	
		h = h_0
		output = []
		for t in range(seq_len):

			h_times_U = torch.nn.functional.linear(h, self.weight_hh_l0, self.bias_hh_l0)
			output.append(self.elu(in_times_W[t] + h_times_U))
			h = output[-1]

		return torch.cat(output, 1), None


class DExpRNNCell(nn.RNN):


	def __init__(self, ninp, nhid, dropout=0, alpha=0.0001):

		super(DExpRNNCell, self).__init__(ninp, nhid, 1, dropout=dropout)
		self.alpha = alpha

	def dilated_exp(self, x):
		return (torch.exp(self.alpha * x) - 1) / self.alpha


	def forward(self, input_, h_0):

		seq_len, bsz, ninp = input_.size()
		in_times_W = torch.nn.functional.linear(input_, self.weight_ih_l0, self.bias_ih_l0)
	
		h = h_0
		output = []
		for t in range(seq_len):

			h_times_U = torch.nn.functional.linear(h, self.weight_hh_l0, self.bias_hh_l0)
			output.append(self.dilated_exp(in_times_W[t] + h_times_U))
			h = output[-1]

		return torch.cat(output, 1), None

class DynamicRNNCell(nn.RNN):

	def __init__(self, ninp, nhid, dropout=0, k=4):

		super(DynamicRNNCell, self).__init__(ninp, nhid, 1, dropout=dropout)
		self.W1 = torch.randn(k * nhid, nhid, dtype=torch.float).cuda()
		self.W2 = torch.randn(k, ninp, dtype=torch.float).cuda()
		self.k = k
		self.ninp, self.nhid = ninp, nhid

	def _in_times_W(self, in_, h):
		Wprime = torch.nn.functional.linear(h, self.W1).view(self.nhid, self.k)
		W = torch.mm(Wprime, self.W2)
		return torch.nn.functional.linear(in_, W, self.bias_ih_l0)

	def forward(self, input_, h_0):

		seq_len, ninp = input_.size()
	
		h = h_0
		output = []
		for t in range(seq_len):

			in_times_W = self._in_times_W(input_[t], h)
			h_times_U = torch.nn.functional.linear(h, self.weight_hh_l0, self.bias_hh_l0)
			output.append(torch.nn.functional.tanh(in_times_W + h_times_U))
			h = output[-1]

		return torch.cat(output, 1)

