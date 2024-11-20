import argparse
import torch

from signals import display, dds, ddc, orda, delay, misc

StreamORDA = orda.StreamORDA

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="one of: tone, pulse, sweep")
parser.add_argument("-a", "--a", help="sweep a parameter")
parser.add_argument("-b", "--b", help="sweep b parameter")
parser.add_argument("-s", "--sysclk", help="ad9910 sysclk in Hz", default="1000*1000*1000")
parser.add_argument("-d", "--duration", help="pulse duration in s", default="900/1000/1000")
parser.add_argument("filename", help="path to an .ISE/.SPU file")
args = parser.parse_args()

def do_sweep(a, b, pulse_duration, filename, sysclk):
	with open(filename, "rb") as f:
		captures = StreamORDA(f).all_captures()

	#
	# We have a number of frequencies and channels in the files
	#
	frequencies = set( [x.center_freq for x in captures] )
	channels = set( [x.channel_number for x in captures] )

	#
	# Prepare model signal
	#
	frames = captures[0].samplecount
	rate = captures[0].samplerate

	duration = frames / rate

	time = dds.time_series(rate, duration)
	band = misc.ad9910_sweep_bandwidth(a, b, duration=pulse_duration, sysclk=sysclk)

	model_sweep = dds.sweep(time, -band/2, band/2, 0, pulse_duration)
	spectrum_m = torch.fft.fft(model_sweep.flip(0).roll(1).conj())

	temporal_freq = (time / pulse_duration)*band - band/2
	spectral_freq = torch.linspace(-rate/2, rate/2, frames)
	indices_allow = torch.where( (spectral_freq >= -band/4) * (spectral_freq <= band/4) )
	index_fcenter = temporal_freq.abs().argmin()

	estimator_a = delay.ConvDelayEstimator(model_sweep)
	estimator_b = delay.SpectralDelayEstimator(model_sweep, indices_allow)

	print("ch,freq,amplitude,phase_offset,delay_a,delay_b")

	for ch in channels:
		for freq in frequencies:
			for capture in filter(lambda x: x.channel_number == ch and x.center_freq == freq, captures):
				phase_offset = torch.dot(capture.iq, model_sweep.conj()).angle()
				amplitude = capture.iq[index_fcenter].abs()
				delay_a = estimator_a.estimate(capture.iq)
				delay_b = estimator_b.estimate(capture.iq)

				fields = [
					f"{ch}",
					f"{freq}",
					f"{amplitude:.3f}",
					f"{phase_offset:.3f}",
					f"{delay_a:.3f}",
					f"{delay_b:.3f}",
				]

				print(",".join(fields))

def parse_numeric_expr(expr):
	token = []
	tokens = []

	for c in expr:
		if c in "*/":
			tokens.append( "".join(token) )
			tokens.append(c)
			token = []
		else:
			token.append(c)

	tokens.append( "".join(token) )
	acc = float( tokens.pop(0) )

	while len(tokens):
		operator = tokens.pop(0)
		operand = float( tokens.pop(0) )

		if operator == "*":
			acc = acc * operand
		elif operator == "/":
			acc = acc / operand
		else:
			assert 0, f"unknown operator {operator}"

	return acc

if args.mode == "sweep":
	assert args.a is not None, "please specify --a"
	assert args.b is not None, "please specify --b"

	a = int(args.a)
	b = int(args.b)
	duration = parse_numeric_expr(args.duration)
	sysclk = parse_numeric_expr(args.sysclk)

	do_sweep(a, b, duration, args.filename, sysclk)
else:
	assert 0, "UNIMP"
