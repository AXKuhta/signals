from signals import display, dds, ddc

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
