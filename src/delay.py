import torch

def delay_in_freq(n, samples):
	"""
	Time delay in frequency domain helper

	n				delay in samples
	samples			total number of samples
	"""

	return torch.exp(1j * torch.arange(samples) * 2 * torch.pi * n / samples)
