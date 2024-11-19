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
	A general purpose delay estimator for any kind of signal

	Steps:
	- Performing a convolution using FFT
	- Fitting a curve using least squares method
	- Finding the peak of the curve

	Parameters:
	model_signal		iq of reference signal with no delay
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

class SpectralDelayEstimator():
	"""
	A delay estimator that relies on the fact that delay in time
	is equivelent to multiplication by exp(i omega tau) in frequency

	Parameters:
	model_signal		iq of reference signal with no delay
	indices_allow		indices of non-marginal spectral content
	"""

	def __init__(self, model_signal, indices_allow):
		self.spectrum_m = torch.fft.fft(model_signal.flip(0).roll(1).conj())
		self.frames = model_signal.shape[0]
		self.indices_allow = indices_allow

	def estimate_old(self, signal):
		spectrum_s = torch.fft.fft(signal)
		spectrum_c = spectrum_s * self.spectrum_m

		offset = 1 # [hack] set to 1 for noisy signals
		tau = torch.fft.fftshift( (spectrum_c.angle().diff() - offset) % -torch.pi + offset )*self.frames/(2*torch.pi)
		sample_delay = -tau[self.indices_allow].mean()

		return sample_delay

	def estimate(self, signal):
		spectrum_s = torch.fft.fft(signal)
		spectrum_c = spectrum_s * self.spectrum_m

		shifted = torch.fft.fftshift(spectrum_c)
		diff = shifted * shifted.roll(1).conj()
		tau = diff.angle() * self.frames / (2*torch.pi)
		sample_delay = -tau[self.indices_allow].mean()

		return sample_delay
