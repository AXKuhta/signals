from signals import display, dds, ddc, delay

def psk_demo():
	time = dds.time_series(100, 1)
	sine = dds.psk(time, 100//5)

	display.signal_fft(time, sine)

def sine_demo():
	time = dds.time_series(44100, 1)
	sine = dds.sine(time, 4)

	display.signal_fft(time, sine)

def pulse_demo():
	time = dds.time_series(1000, 1)
	sine = dds.sine(time, 300, 90, offset=0.1, duration=0.7)

	display.signal_fft(time, sine)

def sweep_demo():
	time = dds.time_series(300, 1)
	sine = dds.sweep(time, 4, 8)

	display.signal_fft(time, sine)

def tune_demo():
	time = dds.time_series(44100, 1)
	data = dds.sweep(time, 6000, 14000)
	sine = dds.sine(time, 10000)

	data = data / sine

	display.signal_fft(time, data)

def sweep_pulse_demo():
	samplerate = 5*1000*1000
	band = 4*1000*1000

	# 2ms recording
	time = dds.time_series(samplerate, 2/1000)

	# Sweep, 100 us offset, 900 us duration
	signal = dds.sweep(time, -band/2, +band/2, 100/1000/1000, 900/1000/1000)

	display.signal_fft(time, signal)

def naive_downsampling_demo():
	time = dds.time_series(225000, 1)
	sine = dds.sweep(time, 45000, 99000)

	time = time[::25]
	sine = sine[::25]

	display.signal_fft(time, sine)

def time_delay_in_frequency_domain_demo():
	time = dds.time_series(1000, 1)

	reference = dds.sweep(time, 4, 8, 0/1000, 400/1000)
	delayed = dds.sweep(time, 4, 8, 1/1000, 400/1000)

	import torch

	spectrum_ref = torch.fft.fftshift( torch.fft.fft(reference) )
	spectrum_del = torch.fft.fftshift( torch.fft.fft(delayed) )

	recov = torch.fft.ifft( torch.fft.fftshift( spectrum_del * spectrum_ref.conj() ) )

	display.signal_fft(time, recov)

def phase_rotation_demo():
	samplerate = 5*1000*1000
	band = 4*1000*1000

	time = dds.time_series(samplerate, 2/1000)

	signal = dds.sweep(time, -band/2, +band/2, 100/1000/1000, 900/1000/1000)

	# Suppose one wants to know the phase in the middle of a sweep compared to a reference sweep
	# One could compute the fft and extract the angle of 0 Hz
	# But but but...
	#
	# dft[k] = sum over n( signal[n] * exp(-i 2 pi k n/N) )
	#
	# note k=0 is a special case that involves less computation:
	#
	# dft[0] = sum over n( signal[n] * exp(i0) )
	# dft[0] = sum over n( signal[n] )
	#
	reference_angle = signal.sum(0).angle()

	signal *= dds.rotator(42)

	altered_angle = signal.sum(0).angle()

	difference = (altered_angle - reference_angle) * (180 / 3.141592)

	print("Rotated by:", difference)

	display.signal_fft(time, signal)

def batched_fft_demo():
	"""
	A demonstration of batched fft being faster when estimating delay in a number of signals
	"""

	from random import random
	from time import time_ns
	import torch

	torch.set_printoptions(sci_mode=False)

	frames = 1000
	rate = 1000

	# The model signal is a sweep
	time = dds.time_series(frames, frames/rate)
	model_sweep = dds.sweep(time, -80, 80, 0/1000, 400/1000)
	band = 80

	spectrum_m = torch.fft.fftshift(torch.fft.fft(model_sweep))
	spectral_freq = torch.linspace(-rate/2, rate/2, frames)

	delays = [i/99 for i in range(100)]
	delayed_lst = []

	# Produce a number of delayed copies
	for x in delays:
		spectrum_d = spectrum_m * delay.delay_in_freq(x, frames).conj()
		delayed = torch.fft.ifft(torch.fft.fftshift(spectrum_d))
		delayed_lst.append(delayed)

	spectrum_m = torch.fft.fft(model_sweep.flip(0).roll(1).conj())

	def eliminate_time_delay(signal):
		spectrum_s = torch.fft.fft(signal)
		spectrum_c = spectrum_s * spectrum_m

		# Time delay estimation
		offset = 1 # [hack] set to 1 for noisy signals
		f_shift = torch.fft.fftshift( (spectrum_c.angle().diff() - offset) % -torch.pi + offset )*frames/(2*torch.pi)
		f_indices = torch.where( (spectral_freq >= -band/4) * (spectral_freq <= band/4) )
		sample_delay = -f_shift[f_indices].mean()

		# Time delay elimination
		spectrum_e = spectrum_s * torch.fft.fftshift( delay.delay_in_freq(sample_delay, frames) )
		signal = torch.fft.ifft(spectrum_e)

		return sample_delay, signal


	def estimate_and_eliminate_time_delay_batch(signals):
		spectra_s = torch.fft.fft(signals)
		spectra_c = spectra_s * spectrum_m

		# Time delay estimation
		offset = 1 # [hack] set to 1 for noisy signals
		f_shift = torch.fft.fftshift( (spectra_c.angle().diff() - offset) % -torch.pi + offset, dim=1 )*frames/(2*torch.pi)
		f_indices, = torch.where( (spectral_freq >= -band/4) * (spectral_freq <= band/4) )
		sample_delay = -f_shift[:, f_indices].mean(1)

		# Time delay elimination
		spectra_e = spectra_s * torch.fft.fftshift( delay.delay_in_freq(sample_delay[:, None], frames) )
		signals = torch.fft.ifft(spectra_e)

		return sample_delay, signals

	delays_recov_v1 = []

	start = time_ns()
	for delayed in delayed_lst:
		delay_recov, undelayed = eliminate_time_delay(delayed)
		delays_recov_v1.append(delay_recov)
	elapsed_v1 = time_ns() - start
	elapsed_v1 /= 1000*1000

	delays_recov_v1 = torch.hstack(delays_recov_v1)

	start = time_ns()
	delays_recov_v2, undelayed = estimate_and_eliminate_time_delay_batch(torch.vstack(delayed_lst))
	elapsed_v2 = time_ns() - start
	elapsed_v2 /= 1000*1000

	delays = torch.tensor(delays).double()

	print(delays)
	print(delays_recov_v1)
	print(delays_recov_v2)

	# Verify that the results match and are close to actual delay
	assert torch.all( torch.isclose(delays, delays_recov_v1) )
	assert torch.all( torch.isclose(delays, delays_recov_v2) )

	print(f"Elapsed v1: {elapsed_v1:.3f} ms")
	print(f"Elapsed v2: {elapsed_v2:.3f} ms")


def filter_demo():
	samplerate = 5*1000*1000

	time = dds.time_series(samplerate, 2/1000)

	signal = time*0

	n = 40
	m = 20

	signal[:(2*n+1)] += ddc.sinc_in_time(n, m)
	signal = signal + 0j

	display.signal_fft(time, signal)

def ddc_demo():
	samplerate = 200*1000*1000		# ADC speed
	band = 4*1000*1000				# Sweep width
	center = 158*1000*1000			# Local oscillator

	time = dds.time_series(samplerate, 2/1000)
	frames = time.shape[0]

	def add_freq(freq):
		actual_f = freq - center
		mirror_f = -freq - center

		actual_apparent_f = actual_f - samplerate*round(actual_f/samplerate)
		mirror_apparent_f = mirror_f - samplerate*round(mirror_f/samplerate)

		print(f"==== {freq/1000/1000} MHz ====")
		print("actual apparent", actual_apparent_f/1000/1000)
		print("mirror apparent", mirror_apparent_f/1000/1000)

		return dds.sine(time, freq, 0, 13.6/1000/1000, 900/1000/1000)

	signal = add_freq(156*1000*1000)*0.5  + add_freq(158*1000*1000)*0.75 + add_freq(160*1000*1000) + add_freq(10*1000*1000)*0.1
	#signal = dds.sweep(time, center - 2.5*1000*1000, center + 2.5*1000*1000, 13.6/1000/1000, 900/1000/1000)

	# Downconversion
	lo = dds.sine(time, center)
	signal = signal.real * lo.conj()

	# Filtering
	filter = signal.real*0
	filter[:8192] += ddc.cic_as_fir_filter(8192, 40).real
	signal = ddc.filter(signal, filter) * 2

	#display.signal_fft(time, filter + 0j)
	#input()

	# Downsampling
	time = time[::40]
	signal = signal[::40]

	# Counteracting the passband droop from the CIC filter
	n = 512
	filter = signal.real * 0
	filter[:n] += ddc.sinc_in_freq(n).real

	filter = ddc.invert_filter(filter)
	signal = ddc.filter(signal, filter)

	display.signal_fft(time, signal)
