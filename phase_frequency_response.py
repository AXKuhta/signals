
from enum import Enum, auto
from glob import glob
import argparse
import json

import numpy as np

from src.misc import ad9910_sweep_bandwidth, ad9910_inv_sinc, parse_numeric_expr, parse_time_expr, parse_freq_expr, roll_lerp, ddc_cost_mv
from src.delay import SpectralDelayEstimator
from src.display import minmaxplot, page
from src.touchstone import S2PFile
from src.orda import StreamORDA
import src.delay as delay
import src.dds as dds

from src.workflows.v1 import ModelSignalV1
from src.schemas.v1 import *

class Mode(Enum):
	NO_REFERENCE = auto()
	REFERENCED = auto()

class PhaseFrequencyResponsePointsV1:
	"""
	How to carry out phase measurements?
	- DDC and the generator appear to trigger up to 200 ns apart
		- Suspect the DDC's pipeline limits its triggering resolution
		- The pipeline probably runs continuously
		- Does not restart on trigger
		- Trigger merely sets a "capture pending" flag
		- Flag is effective next output of decimator
	- Dechirping will produce up to 500 kHz / 900 us * 200 ns = 111.[1] Hz tone
	- Will amount to 2*pi radian * 111.1 Hz * 900 us to deg = 36 deg travel
	- Where there really should have been 0 deg travel
	- To fight this off two DDC channels must be used
	- They are not 200 ns apart, they are coherent
	- Treating one as incident
	- The other as transmitted
	- Reference run
		- Splitter	incident
		- Thru		transmitted
	- Test run
		- Splitter	incident
		- DUT		transmitted

	So the math is:

		z_n(t) = u_n(t) * v_n(t).conj

	Where:

		u_n(t)	transmitted
		v_n(t)	incident

	What passes for averaging is:

		z(t) = sum_over_n( z_n(t) )
		z(t) = z(t) / |z(t)| --- optional, if one wants 1.0 magnitude

	Associate z(t) samples with freq
	Extract a subset of z(t)

	Then, for a referenced mode:

		arg( z(t) * w(t).conj )
	"""

	def __init__(self, location, idx_a, idx_b, trim=0.05, radians=False):
		with open(f"{location}/preset.json") as f:
			obj = json.load(f)

		preset = JsonDDCAndCalibratorV1.deserialize(obj["ddc-and-calibrator-v1"])

		captures = []

		# No streaming, just load it all
		# Will take up some RAM
		#################################################################################################
		for filename in sorted(glob(f"{location}/*.ISE")):
			with open(filename, "rb") as f:
				for capture in StreamORDA(f).captures:
					if capture.center_freq == 0:
						continue # DDC quirk: 0 Hz must be skipped

					captures.append(capture)

		chan_set = set([x.channel_number for x in captures])

		# Trigger number rewrite
		for i, capture in enumerate(captures):
			capture.trigger_number = i // len(chan_set)

		print("Loaded", len(captures), "captures")
		print("Active channels:", chan_set)

		assert idx_a in chan_set, "Channel not available"
		assert idx_b in chan_set, "Channel not available"

		q_x = []
		q_y = []

		model_x = []
		model_y = []

		signals = []

		for descriptor in preset.signals:
			signals.append( ModelSignalV1(descriptor, preset.ddc) )

		# Onto the actual processing
		# Delay elimination + amplitude + averaging
		#################################################################################################
		for i, signal in enumerate(signals):
			tune = parse_freq_expr(signal.descriptor.tune)

			# Pulse cropping
			start = signal.duration*trim + signal.delay
			stop = signal.duration*(1-trim) + signal.delay

			indices = (signal.time >= start) * (signal.time < stop)

			# Model captures
			x = signal.temporal_freq[indices] + tune
			y = np.abs(signal.iq)[indices]

			model_x.append(x)
			model_y.append(y)

			u_filter_fn = lambda x: x.trigger_number % len(signals) == i and x.channel_number == idx_a
			v_filter_fn = lambda x: x.trigger_number % len(signals) == i and x.channel_number == idx_b

			u_repeats = list(filter(u_filter_fn, captures))
			v_repeats = list(filter(v_filter_fn, captures))

			assert len(u_repeats) == len(v_repeats)
			assert all([x.center_freq == tune for x in u_repeats])
			assert all([x.center_freq == tune for x in v_repeats])

			z_n = []

			# Estimate delay once, for channel a
			# DO NOT estimate delay individually
			# Thay would defy the point
			for u, v in zip(u_repeats, v_repeats):
				delay = signal.est.estimate(u.iq)

				u_ = np.roll(u.iq, -delay)
				v_ = np.roll(v.iq, -delay)

				z_n.append( u_ * v_.conj() )

			z = np.sum(z_n, 0)

			x = signal.temporal_freq[indices] + tune
			y = z[indices]

			q_x.append(x)
			q_y.append(y)

		# Second pass over the data to sort it - this deals with overlap
		#################################################################################################
		x = np.hstack(q_x)
		y = np.hstack(q_y)

		x, indices = np.sort(x), np.argsort(x)
		y = y[indices]

		self.q_x = x
		self.q_y = y
		self.chan_set = chan_set

		self.model_x = np.hstack(model_x)
		self.model_y = np.hstack(model_y)

		self.radians = radians

	def csv(self, mode, filename, reference=None):
		x = []
		y = []
		cols = ["freq_hz"]

		if self.radians:
			factor = 1.0
		else:
			factor = 180.0 / np.pi


		if mode == Mode.NO_REFERENCE:
			assert 0
		elif mode == Mode.REFERENCED:
			assert 0
		else:
			assert 0

		np.savetxt(
			filename,
			np.vstack([x[0]] + y).T,
			comments="",
			delimiter=",",
			header=",".join(cols)
		)

	def display(self, mode, reference=None):
		spectral = minmaxplot("Hz")
		spectral.xtitle("Frequency")

		if self.radians:
			spectral.ytitle("Radians")
			factor = 1.0
		else:
			spectral.ytitle("Degrees")
			factor = 180.0 / np.pi

		if mode == Mode.NO_REFERENCE:
			spectral.trace(self.q_x, np.angle(self.q_y) * factor)
		elif mode == Mode.REFERENCED:
			adjusted = np.angle( self.q_y * reference.q_y.conj() )
			spectral.trace(self.q_x, adjusted * factor)
		else:
			assert 0

		result = page([spectral])
		result.show()


parser = argparse.ArgumentParser(description="Produces a plot or a csv file of phase frequency response, needs special measurement setup.")
parser.add_argument("--dut", help="path to a directory containing captures+metadata with test signals split into bypass + fed into device under test", required=True)
parser.add_argument("--ref", help="path to a directory containing captures+metadata with test signals split into bypass + fed into a thru")
parser.add_argument("--channels", help="specify a channel pair pattern a,b where a is a bypass channel and b is either a thru or a dut channel", required=True)
parser.add_argument("--trim", help="specify the proportion of pulse head and tail to be discarded in time domain to remove transients, defaults to 0.05")
parser.add_argument("--rad", help="use radians instead of degrees", action="store_true")
parser.add_argument("--csv", help="write results into a specified csv file")
args = parser.parse_args()

trim = float(args.trim or "0.05")
idx_a, idx_b = [int(x) for x in args.channels.split(",")]

if args.ref:
	a = PhaseFrequencyResponsePointsV1(args.dut, idx_a, idx_b, trim=trim, radians=args.rad)
	b = PhaseFrequencyResponsePointsV1(args.ref, idx_a, idx_b, trim=trim, radians=args.rad)

	if args.csv:
		a.csv(Mode.REFERENCED, arg.csv, b)
	else:
		a.display(Mode.REFERENCED, b)

else:
	a = PhaseFrequencyResponsePointsV1(args.dut, idx_a, idx_b, trim=trim, radians=args.rad)

	if args.csv:
		a.csv(Mode.NO_REFERENCE, arg.csv)
	else:
		a.display(Mode.NO_REFERENCE)
