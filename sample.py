import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

class NegativeSampler(nn.Module):

	def __init__(self, nsamples, frequencies, exp=0.75):

		self.nsamples = nsamples
		self.frequencies = (frequencies / torch.sum(frequencies)).pow(exp); self.frequencies[-1] = 0.; print(self.frequencies)

		super(NegativeSampler, self).__init__()

	def forward(self, bsz, seq_len, cuda=True):
		# returns bsz*seq_len*nsamples samples in shape nsamples x (bsz x seq_len)

		# sample based on frequencies
		wrs = WeightedRandomSampler(self.frequencies, self.nsamples * bsz * seq_len, replacement=True)
		samples = torch.LongTensor(list(wrs)).cuda() if cuda else torch.LongTensor(list(wrs))

		return samples.view(-1, bsz)

