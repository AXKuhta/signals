
import argparse
import torch

from src.misc import ad9910_sweep_bandwidth, parse_numeric_expr, roll_lerp
from src.delay import SpectralDelayEstimator
from src.orda import StreamORDA
import src.delay as delay
import src.dds as dds

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--a", help="sweep a parameter")
parser.add_argument("-b", "--b", help="sweep b parameter")
parser.add_argument("-r", "--retain", help="how wide of a stretch should be retained per capture, in Hz")
parser.add_argument("-s", "--sysclk", help="ad9910 sysclk in Hz", default="1000*1000*1000")
parser.add_argument("-d", "--duration", help="pulse duration in s", default="900/1000/1000")
parser.add_argument("filename_ref", help="path to an .ISE/.SPU file with reference signals (DUT bypassed)")
parser.add_argument("filename_dut", help="path to an .ISE/.SPU file with altered signals (signal feeding through the DUT)")
args = parser.parse_args()

def run_v1(fname_ref, fname_dut, a, b, pulse_duration, retain, sysclk):
	#
	# Load all captures
	#
	with open(fname_ref, "rb") as f:
		captures_ref = StreamORDA(f).all_captures()

	with open(fname_dut, "rb") as f:
		captures_dut = StreamORDA(f).all_captures()

	#
	# We have a number of frequencies in the files
	#
	freqs_ref = set( [x.center_freq for x in captures_ref] )
	freqs_dut = set( [x.center_freq for x in captures_dut] )

	assert freqs_ref == freqs_dut, "Frequencies don't match between two files"

	#
	# Prepare model signal
	#
	frames = captures_ref[0].samplecount
	rate = captures_ref[0].samplerate

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
	# Reuse the indices for delay estimation
	# Retained strecth bound to have non-marginal spectral content
	#
	est = SpectralDelayEstimator(model_sweep, indices)

	#
	# 0 Hz must be skipped
	#
	freq_set = freqs_ref - set( [0] )

	def eliminate_time_delay(signal):
		sample_delay = est.estimate(signal)
		return roll_lerp(signal, -sample_delay)

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

		for x_, y_ in zip(x, y):
			print(f"{x_},{y_}")

assert args.a is not None, "please specify --a"
assert args.b is not None, "please specify --b"

a = int(args.a)
b = int(args.b)
retain = parse_numeric_expr(args.retain)
duration = parse_numeric_expr(args.duration)
sysclk = parse_numeric_expr(args.sysclk)

run_v1(args.filename_ref, args.filename_dut, a, b, duration, retain, sysclk)
