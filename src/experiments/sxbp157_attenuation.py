
import torch

from src.orda import StreamORDA
from src.display import page, minmaxplot
import src.delay as delay
import src.dds as dds

sxbp157_mhz = torch.tensor([
	#1.0,
	115.0,
	131.0,
	139.0,
	143.0,
	146.0,
	150.0,
	157.0,
	164.0,
	169.0,
	172.0,
	178.0,
	187.0,
	215.0,
	#750.0,
	#1000.0,
	#1500.0,
	#2000.0
])

sxbp157_mean = torch.tensor([
	#85.08,
	52.32,
	30.98,
	15.86,
	7.41,
	3.62,
	2.44,
	2.22,
	2.53,
	4.80,
	9.17,
	18.91,
	29.82,
	49.38,
	#93.83,
	#83.28,
	#70.47,
	#60.88
])

sxbp157_sigma = torch.tensor([
	#2.10,
	0.36,
	0.35,
	0.38,
	0.39,
	0.21,
	0.03,
	0.03,
	0.04,
	0.31,
	0.42,
	0.32,
	0.22,
	0.19,
	#3.88,
	#1.86,
	#0.42,
	#0.91
])

#
# AD9910 sweep calculations
#
def ad9910_sweep_bandwidth(a, b, duration=900/1000/1000, sysclk=1000*1000*1000):
	fstep = sysclk / 2**32
	steps = sysclk / 4 / b * duration

	assert steps % 1 == 0

	return a * fstep * (steps - 1)

#
# Testing a combined filter+amplifier
# consisting of
# - a SXBP-157+
# - a PHA-13HLN+
#
# https://www.minicircuits.com/pdfs/SXBP-157+.pdf
# https://www.minicircuits.com/pdfs/PHA-13HLN+.pdf
#
def run_v1():
	#
	# Load all captures
	#
	with open("cal_2024_10_29/20241029_074522_000_0000_003_000.ISE", "rb") as f:
		captures_ref = StreamORDA(f).all_captures()

	with open("cal_2024_10_29/20241029_075859_000_0000_003_000.ISE", "rb") as f:
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

	#
	# Snip full pulse duration (a bit over 4 MHz)
	# Or snip exactly 4 MHz
	#
	indices = time <= pulse_duration
	indices = (temporal_freq >= -2*1000*1000)*(temporal_freq < 2*1000*1000)

	spectral = minmaxplot("Hz")
	spectral.yrange([0 -2.5, 50 +2.5])
	spectral.xlogscale()
	spectral.ytitle("dB")
	spectral.xtitle("Частота")

	#
	# 0 Hz must be skipped
	#
	freq_set = freqs_ref - set( [0] )

	lst_x = []
	lst_y = []
	lst_lower = []
	lst_upper = []

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

		# Gain from PHA-13HLN+
		amp_gain = 22.5

		mampl = -10*torch.log10(mampl) + amp_gain
		lower = -10*torch.log10(lower) + amp_gain
		upper = -10*torch.log10(upper) + amp_gain

		x = temporal_freq[indices] + freq
		y = mampl[indices]

		lower = lower[indices]
		upper = upper[indices]

		#spectral.trace(x, y, error_band=(lower, upper), name=f"{freq/1000/1000} MHz")

		lst_x.append(x)
		lst_y.append(y)
		lst_lower.append(lower)
		lst_upper.append(upper)

	x = torch.hstack(lst_x)
	y = torch.hstack(lst_y)
	lower = torch.hstack(lst_lower)
	upper = torch.hstack(lst_upper)

	#spectral.trace(x, y, error_band=(lower, upper), name="Attenuation")
	spectral.trace(x, y, name="Attenuation")

	spectral.trace([115*1000*1000] * 2, [40, 0], name="F5")
	spectral.trace([131*1000*1000] * 2, [20, 0], name="F3")
	spectral.trace([150*1000*1000] * 2, [ 3, 0], name="F1")

	spectral.trace([164*1000*1000] * 2, [ 3, 0], name="F2")
	spectral.trace([187*1000*1000] * 2, [20, 0], name="F4")
	spectral.trace([215*1000*1000] * 2, [40, 0], name="F6")

	x = sxbp157_mhz*1000*1000
	y = sxbp157_mean
	upper = sxbp157_mean + 3*sxbp157_sigma
	lower = sxbp157_mean - 3*sxbp157_sigma

	spectral.hsl_color_cycler.pop(0)
	spectral.trace(x, y, error_band=(lower, upper), name="SXBP-157+ Datasheet")

	disp = page([spectral])
	disp.show()
	input()
