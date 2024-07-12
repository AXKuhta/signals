import torch

def sinc_in_time(n=40, m=20):
	"""
	Produces a lowpass filtering kernel with 2n+1 elements which is essentially a sinc function

	n			level of detail
	m			cutoff point

	Bandwidth of the resulting filter is samplerate * m / n

	https://en.wikipedia.org/wiki/Sinc_filter
	https://en.wikipedia.org/wiki/Raised-cosine_filter

	Usage:

	```
	filter = torch.zeros_like(time)
	filter[:(2*n+1)] += ddc.sinc_filter(40, 20)

	signal = ddc.filter(signal, filter)
	```
	"""

	x = torch.arange(-n, n + 1) / n * m + 0.001
	return torch.sin(torch.pi*x) / (torch.pi*x) / (n / m)

def sinc_in_freq(n=512, offset=0.5, order=5):
	"""
	Produces a filtering kernel that looks like a sinc function in frequency domain;
	Intended primarily to be used along with invert_filter() for compensating the effect of a CIC filter - that's also what the order option is there for

	n			level of detail
	offset		how much of sinc function should be computed towards both sides from zero
	order		raise the function to nth power
	"""

	x = torch.linspace(-offset, +offset, n)
	x = torch.fft.fftshift(x)
	y = torch.sin(torch.pi*x) / (torch.pi*x)
	y = y**order

	assert not torch.any(torch.isnan(y))

	return torch.fft.ifft(y).roll(-n//2)

def cic_as_fir_filter(n=512, d=2, stages=5):
	"""
	Produces an idealized CIC filtering kernel

	n			level of detail
	d			cutoff point
	stages		number of stages

	Bandwidth of the resulting filter is (roughly) samplerate / d
	"""

	sinc_in_freq = torch.fft.fft(torch.tensor([1]*d) / d, n)

	cic_in_freq = sinc_in_freq**stages
	cic_in_time = torch.fft.ifft(cic_in_freq)

	return cic_in_time

def filter(signal, filter):
	"""
	Filter a signal using fft convolution
	"""

	spectrum_s = torch.fft.fft(signal)
	spectrum_f = torch.fft.fft(filter)
	spectrum_c = spectrum_s * spectrum_f

	return torch.fft.ifft(spectrum_c)

def invert_filter(filter):
	"""
	Attempts to compute a compensating filter for a given FIR filter

	filter		filter weights
	"""

	spectrum_f = torch.fft.fft(filter)
	spectrum_i = 1 / spectrum_f.conj() # Keep delay same
	filter_i = torch.fft.ifft(spectrum_i)

	assert not torch.any(torch.isnan(filter_i.real))
	assert not torch.any(torch.isnan(filter_i.imag))

	return filter_i

def cic(signal, d=2, stages=5):
	"""
	Run a signal through a CIC filter

	signal		input signal
	d			cutoff point
	stages		number of stages

	Bandwidth of the resulting filter is (roughly) samplerate / d

	Does not perform decimation, be sure to:
	signal = signal[::d]
	time = time[::d]

	Takes only int16 inputs, convert to int16 first:

	signal = ( signal / signal.abs().max() * 32767.0 ).to(torch.int16)
	"""

	assert signal.dtype == torch.int16

	for i in range(stages):
		signal = torch.cumsum(signal, 0, dtype=torch.int64)

	# Decimation can be performed here
	#signal = signal[::d]

	for i in range(stages):
		signal = signal - torch.roll(signal, d) # Replace d with 1 if decimating

	# Initial elements contain nonsense
	# Remove d if decimating
	signal[:(d*stages)] = 0

	return signal / d**stages
