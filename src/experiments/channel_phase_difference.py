
import argparse
import torch

from src.misc import ad9910_sweep_bandwidth, parse_numeric_expr, roll_lerp
from src.delay import SpectralDelayEstimator
from src.orda import StreamORDA
import src.delay as delay
import src.dds as dds

import src.display as display

def run_v1(fname, spectral, a=10, b=1, pulse_duration=900/1000/1000, retain=500*1000, sysclk=1000*1000*1000):
	#
	# Load all captures
	#
	with open(fname, "rb") as f:
		captures = StreamORDA(f).all_captures()

	#
	# We have a number of frequencies
	#
	freqs = set( [x.center_freq for x in captures] )

	#
	# Prepare model signal
	#
	frames = captures[0].samplecount
	rate = captures[0].samplerate

	duration = frames / rate

	time = dds.time_series(rate, duration)
	band = ad9910_sweep_bandwidth(a, b, sysclk=sysclk)

	model_sweep = dds.sweep(time, -band/2, band/2, 0, pulse_duration)
	temporal_freq = (time / pulse_duration)*band - band/2
	spectral_freq = torch.linspace(-rate/2, rate/2, frames)

	#
	# Retain a stretch of the sweep
	#
	indices = time <= pulse_duration
	indices = (temporal_freq >= -retain/2)*(temporal_freq < +retain/2)

	#
	# A narrower stretch for delay estimation to achieve better accuracy
	#
	indices_est = (spectral_freq >= -retain/3)*(spectral_freq < +retain/3)
	est = SpectralDelayEstimator(model_sweep, indices_est)

	#
	# 0 Hz must be skipped
	#
	freq_set = freqs - set( [0] )

	all_x = []
	all_y = []

	sparse_x = []
	sparse_y = []

	RAD2DEG = 180.0/torch.pi

	#
	# Do one at a time
	#
	for freq in sorted(freq_set):
		a_filter_fn = lambda x: x.center_freq == freq and x.channel_number == 1
		b_filter_fn = lambda x: x.center_freq == freq and x.channel_number == 3

		ch_a = filter(a_filter_fn, captures)
		ch_b = filter(b_filter_fn, captures)

		a = [x.iq for x in ch_a]
		b = [x.iq for x in ch_b]
		c = torch.vstack( [ v * u.conj() for u, v in zip(a, b) ] )
		d = c.angle()

		# Continuous but fuzzy estimation
		# Taken at each point
		mampl = d.mean(0)
		lower, _ = d.min(0)
		upper, _ = d.max(0)

		# Sparse (discontinuous) but certain estimation
		# Represents center frequency
		scalar_phase_diff = c.sum(1).angle().mean(0)

		sparse_x.append(freq)
		sparse_y.append(scalar_phase_diff * RAD2DEG)

		# Estimate delay
		# Delay between channel 1 and channel 3 is essentially the same in samples
		# use channel 1 delay
		sample_delay = est.estimate(a[0])

		# Eliminate delay
		mampl = mampl.roll(-round(sample_delay.item()))
		#mampl = roll_lerp(mampl, -sample_delay)

		x = temporal_freq[indices] + freq
		y = mampl[indices]

		all_x.append(x)
		all_y.append(y)

		#spectral.trace(x, y, name=f"Fc = {freq/1000/1000:.2f} MHz")

	x = torch.hstack(all_x)
	y = torch.hstack(all_y) * RAD2DEG
	_, indices = torch.sort(x)
	x = x[indices]
	y = y[indices]

	spectral.trace(x, y, name="Метод 1")
	spectral.trace(sparse_x, sparse_y, name="Метод 2")

def run_v2():
	spectral = display.minmaxplot("Hz")
	spectral.xtitle("Частота")
	spectral.ytitle("Градусы")
	spectral.header("Разность фаз между каналами ВУПа (B - A)")

	#run_v1("cal_2025_02_25/20250225_030146_000_0000_003_000.ISE", spectral)
	#run_v1("cal_2025_02_25/20250225_030635_000_0000_003_000.ISE", spectral)

	#run_v1("cal_2025_02_25/20250225_042408_000_0000_003_000.ISE", spectral)
	#run_v1("cal_2025_02_25/20250225_042938_000_0000_003_000.ISE", spectral)

	run_v1("cal_2025_02_21/a_to_1_b_to_3.ISE", spectral)
	#run_v1("cal_2025_02_21/a_to_3_b_to_1.ISE", spectral)


	z = display.page([spectral])
	z.show()
