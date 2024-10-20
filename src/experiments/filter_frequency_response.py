
import torch

from src.orda import StreamORDA
from src.display import page, minmaxplot
import src.delay as delay
import src.dds as dds

#
# AD9910 sweep calculations
#
def ad9910_sweep_bandwidth(a, b, duration=900/1000/1000, sysclk=1000*1000*1000):
	fstep = sysclk / 2**32
	steps = sysclk / 4 / b * duration

	assert steps % 1 == 0

	return a * fstep * (steps - 1)

def run_v1():
	#
	# Load all captures
	#
	with open("20241001_071338_000_0000_003_000.ISE", "rb") as f:
		captures_ref = StreamORDA(f).all_captures()

	with open("20241001_072840_000_0000_003_000.ISE", "rb") as f:
		captures_dut = StreamORDA(f).all_captures()

	#
	# We have a number of frequencies in the files
	#
	freqs_ref = set( [x.center_freq for x in captures_ref] )
	freqs_dut = set( [x.center_freq for x in captures_dut] )

	assert freqs_ref == freqs_dut

	#
	# Prepare model signal
	#
	frames = captures_ref[0].samplecount
	rate = captures_ref[0].samplerate

	duration = frames / rate

	time = dds.time_series(rate, duration)
	band = ad9910_sweep_bandwidth(77, 1)
	pulse_duration = 900/1000/1000

	model_sweep = dds.sweep(time, -band/2, band/2, 0, pulse_duration)
	spectrum_m = torch.fft.fft(model_sweep.flip(0).roll(1).conj())

	temporal_freq = (time / pulse_duration)*band - band/2
	spectral_freq = torch.linspace(-rate/2, rate/2, frames)

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

		return signal

	indices = time <= pulse_duration

	spectral = minmaxplot("Hz")
	spectral.ytitle("dB")
	spectral.xtitle("Частота")

	#
	# Two of the frequencies are bugged and must be skipped
	#
	freq_set = freqs_ref - set( [0, 100*1000*1000] )

	#
	# Do one at a time
	#
	for freq in sorted(freq_set):
		filter_fn = lambda x: x.center_freq == freq and x.channel_number == 1
		ref = filter(filter_fn, captures_ref)
		dut = filter(filter_fn, captures_dut)

		a = [eliminate_time_delay(x.iq) for x in ref]
		b = [eliminate_time_delay(x.iq) for x in dut]
		c = torch.vstack( [ v.abs() / u.abs() for u, v in zip(a, b) ] )

		mampl = c.mean(0)
		lower, _ = c.min(0)
		upper, _ = c.max(0)

		mampl = 10*torch.log10(mampl)
		lower = 10*torch.log10(lower)
		upper = 10*torch.log10(upper)

		print(freq)

		x = temporal_freq[indices] + freq
		y = mampl[indices]

		spectral.trace(x, y, error_band=(lower[indices], upper[indices]), name=f"{freq/1000/1000}MHz")

	disp = page([spectral])
	disp.show()
	input()

def run_v2():
	frames = 8192
	rate = 5*1000*1000
	duration = frames / rate

	time = dds.time_series(rate, duration)
	band = ad9910_sweep_bandwidth(77, 1)
	pulse_duration = 900/1000/1000

	temporal_freq = (time / pulse_duration)*band - band/2
	spectral_freq = torch.linspace(-rate/2, rate/2, frames)

	# Prepare model signal
	model_sweep = dds.sweep(time, -band/2, band/2, 0, pulse_duration)
	spectrum_m = torch.fft.fft(model_sweep.flip(0).roll(1).conj())

	# Load all captures
	with open("20241001_071338_000_0000_003_000.ISE", "rb") as f:
		captures_ref = StreamORDA(f).all_captures()

	with open("20241001_072840_000_0000_003_000.ISE", "rb") as f:
		captures_dut = StreamORDA(f).all_captures()

	# We have a number of frequencies in the files
	freqs_ref = set( [x.center_freq for x in captures_ref] )
	freqs_dut = set( [x.center_freq for x in captures_dut] )

	assert freqs_ref == freqs_dut

	# Two of the frequencies are bugged and must be skipped
	freq_set = freqs_ref - set( [0, 100*1000*1000] )

	print(len(captures_ref))

	# Filter captures to only include channel 1 and allowed frequencies
	filter_fn = lambda x: x.center_freq in freq_set and x.channel_number == 1
	captures_ref = list( filter(filter_fn, captures_ref) )
	captures_dut = list( filter(filter_fn, captures_dut) )

	def estimate_and_eliminate_time_delay_batch(signals):
		spectra_s = torch.fft.fft(signals)
		spectra_c = spectra_s * spectrum_m

		# Time delay estimation
		offset = 1 # [hack] set to 1 for noisy signals
		f_shift = torch.fft.fftshift( (spectra_c.angle().diff() - offset) % -torch.pi + offset )*frames/(2*torch.pi)
		f_indices, = torch.where( (spectral_freq >= -band/4) * (spectral_freq <= band/4) )
		sample_delay = -f_shift[:, f_indices].mean(1)

		# Time delay elimination
		spectra_e = spectra_s * torch.fft.fftshift( delay.delay_in_freq(sample_delay[:, None], frames) )
		signals = torch.fft.ifft(spectra_e)

		return signals

	zeroed_ref = estimate_and_eliminate_time_delay_batch( torch.vstack( [x.iq for x in captures_ref] ) )
	zeroed_dut = estimate_and_eliminate_time_delay_batch( torch.vstack( [x.iq for x in captures_dut] ) )

	for x, iq in zip(captures_ref, zeroed_ref):
		x.iq = iq

	for x, iq in zip(captures_dut, zeroed_dut):
		x.iq = iq

	indices = time <= pulse_duration

	spectral = minmaxplot("Hz")
	spectral.ytitle("dB")
	spectral.xtitle("Частота")

	#
	# Do one at a time
	#
	for freq in sorted(freq_set):
		filter_fn = lambda x: x.center_freq == freq
		ref = filter(filter_fn, captures_ref)
		dut = filter(filter_fn, captures_dut)

		a = [x.iq for x in ref]
		b = [x.iq for x in dut]
		c = torch.vstack( [ v.abs() / u.abs() for u, v in zip(a, b) ] )

		mampl = c.mean(0)
		lower, _ = c.min(0)
		upper, _ = c.max(0)

		mampl = 10*torch.log10(mampl)
		lower = 10*torch.log10(lower)
		upper = 10*torch.log10(upper)

		print(freq)

		x = temporal_freq[indices] + freq
		y = mampl[indices]

		spectral.trace(x, y, error_band=(lower[indices], upper[indices]), name=f"{freq/1000/1000}MHz")

	disp = page([spectral])
	disp.show()
	input()
