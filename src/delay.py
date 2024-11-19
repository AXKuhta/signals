import torch

def delay_in_freq(n, samples):
	"""
	Time delay in frequency domain helper

	n				delay in samples
	samples			total number of samples
	"""

	return torch.exp(1j * torch.arange(samples) * 2 * torch.pi * n / samples)

class ConvDelayEstimator():
	"""
	General purpose delay estimation for any kind of signal

	Steps:
	- Performing a convolution using FFT
	- Fitting a curve using least squares method
	- Finding the peak of the curve
	"""

	def __init__(self, model_signal):
		self.spectrum_m = torch.fft.fft(model_signal.flip(0).roll(1).conj())
		self.n = 6

	def estimate(self, signal):
		spectrum_s = torch.fft.fft(signal)
		spectrum_c = spectrum_s * self.spectrum_m
		convolved = torch.fft.ifft(spectrum_c)
		score = convolved.abs()

		values, indices = score.sort()
		values = values[-self.n:]
		indices = indices[-self.n:]
		matrix = torch.vstack([ indices, indices*indices, torch.ones(self.n) ]).double().T
		solution, _, _, _ = torch.linalg.lstsq(matrix, values)
		a, b, c = solution

		# ax + bx^2 + c = ...
		# a + 2bx = 0
		# 2bx = -a
		# x = -a/(2b)
		sample_delay = -a / 2 / b

		return sample_delay
