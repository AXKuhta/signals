
import torch

from src.orda import StreamORDA
from src.display import page, minmaxplot
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
		# V3 experimental
		# + hack to work despite noise
		f_shift = torch.fft.fftshift( (spectrum_c.angle().diff() - 1) % -torch.pi + 1 )*frames/(2*torch.pi)
		f_indices = torch.where( (spectral_freq >= -band/4) * (spectral_freq <= band/4) )
		f_shift2 = f_shift[f_indices]

		sample_delay = -f_shift2.mean()
		print("Delay V3:", sample_delay)

		# Time delay elimination
		# V1 exp(i 2pi k n/N)
		spectrum_e = spectrum_s * torch.fft.fftshift( dds.timedelay(sample_delay, frames) )
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
