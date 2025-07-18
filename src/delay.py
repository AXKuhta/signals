import numpy as np

def delay_in_freq(n, samples):
	"""
	Time delay in frequency domain helper

	n				delay in samples
	samples			total number of samples
	"""

	return np.exp(1j * np.arange(samples) * 2 * np.pi * n / samples)

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
		self.spectrum_m = np.fft.fft(np.roll(np.flip(model_signal), 1).conj())
		self.n = 6

	def estimate(self, signal):
		spectrum_s = np.fft.fft(signal)
		spectrum_c = spectrum_s * self.spectrum_m
		convolved = np.fft.ifft(spectrum_c)
		score = np.abs(convolved)

		values, indices = np.sort(score), np.argsort(score)
		values = values[-self.n:]
		indices = indices[-self.n:]
		matrix = np.vstack([ indices, indices*indices, np.ones(self.n) ]).double().T
		solution, _, _, _ = np.linalg.lstsq(matrix, values)
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
		self.spectrum_m = np.fft.fft(np.roll(np.flip(model_signal), 1).conj())
		self.frames = model_signal.shape[0]
		self.indices_allow = indices_allow

	# V0: initial version
	def estimate_old_old(self, signal):
		spectrum_s = np.fft.fft(signal)
		spectrum_c = spectrum_s * self.spectrum_m

		offset = 1 # [hack] set to 1 for noisy signals
		tau = np.fft.fftshift( (np.diff(np.angle(spectrum_c)) - offset) % -np.pi + offset )*self.frames/(2*np.pi)
		sample_delay = -tau[self.indices_allow].mean()

		return sample_delay

	# V1: avoid diffing angles, diff IQ
	def estimate_old(self, signal):
		spectrum_s = np.fft.fft(signal)
		spectrum_c = spectrum_s * self.spectrum_m


		shifted = np.fft.fftshift(spectrum_c)
		diff = shifted * np.roll(shifted, 1).conj()
		tau = np.angle(diff) * self.frames / (2*np.pi)
		sample_delay = -tau[self.indices_allow].mean()

		def visual_debug():
			from .display import minmaxplot, page

			rate = 5*1000*1000
			freqs = np.linspace(-rate/2, rate/2, 8192)

			spectral = minmaxplot("Hz")
			spectral.trace(freqs[self.indices_allow], tau[self.indices_allow])
			disp = page([spectral])
			disp.show()
			input()

		#visual_debug()

		return sample_delay

	# V2: avoid excessive use np.angle() which is expensive
	def estimate(self, signal):
		spectrum_s = np.fft.fft(signal)
		spectrum_c = spectrum_s * self.spectrum_m

		shifted = np.fft.fftshift(spectrum_c)
		diff = shifted * np.roll(shifted, 1).conj()
		acc_angle = np.sum(diff[self.indices_allow])
		tau = np.angle(acc_angle) * self.frames / (2*np.pi)

		return -tau
