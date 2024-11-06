
import torch

from src.orda import StreamORDA
from src.touchstone import S2PFile
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

"""
Resistor box:

         +--10K-+
IN --+---+      +---+-- OUT
     |   +--1K--+   |
    50 ohms        50 ohms
     |              |
    GND            GND
"""
def run_v1():
	fname_ddc = "cal_2024_10_23/20241023_054128_000_0000_003_000.ISE"
	fname_box = "cal_2024_10_29/20241029_060435_000_0000_003_000.ISE"

	#
	# Load all captures
	#
	with open(fname_ddc, "rb") as f:
		captures_ref = StreamORDA(f).all_captures()

	with open(fname_box, "rb") as f:
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
	indices = (temporal_freq >= -2*1000*1000)*(temporal_freq < 2*1000*1000)

	spectral = minmaxplot("Hz")
	spectral.yrange([-50, 50])
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

		mampl = 20*torch.log10(mampl)
		lower = 20*torch.log10(lower)
		upper = 20*torch.log10(upper)

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

	#spectral.trace(x, y, error_band=(lower, upper), name="Calibrator")
	spectral.trace(x, y, name="Calibrator")

	fname_vna_box = "VNA/10k_resistor_box_minus_10dbm.s2p"

	with open(fname_vna_box, "rb") as f:
		vna = S2PFile(f)

	spectral.trace(vna.freqs, 20*torch.log10(vna.s21.abs()), name="VNA")

	disp = page([spectral])
	disp.show()
	input()
