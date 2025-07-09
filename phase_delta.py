
from enum import Enum, auto
from glob import glob
import argparse
import json

import numpy as np

from src.misc import ad9910_sweep_bandwidth, ad9910_inv_sinc, parse_numeric_expr, parse_time_expr, parse_freq_expr, roll_lerp, ddc_cost_mv
from src.delay import SpectralDelayEstimator
from src.display import minmaxplot, page
from src.orda import StreamORDA
import src.delay as delay
import src.dds as dds

from src.workflows.v1 import ModelSignalV1
from src.schemas.v1 import *

class PhaseDeltaPointsV1:
	"""
	Application class to determine phase difference between channels

	Assumes two channels were taking the same signal

	Takes:
	- A location with captures+metadata
	- Pair of channel numbers

	Makes:
	- A plot or a csv file

	The pipeline is different from amplitude frequency response:
	- Read the metadata
		- Parse JSON
		- Verify schema
	- Read the captures
		- Glob all .ISEs
		- Discard 0 Hz captures
	- Detect active channels
	- Pool phase-delta-from-frequency points across descriptors
		- Prepare a model signal
		- Find associated captures
		- Establish phase delta across repeated captures
			- For capture pairs grouped by trigger number
				- Estimate delay
				- Eliminate delay
				- Find phase difference
			- Apply averaging
			- Map time to frequency
			- Discard samples outside pulse body
				- Below 0.05*duration
				- Above 0.95*duration
				- To remove transients
		- Apply pooling
			- Join and sort
			- Deals with overlap
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

		# High density phase delta points
		fine_delta_x = []
		fine_delta_y = []
		fine_delta_upper = []
		fine_delta_lower = []

		# Low density
		coarse_delta_x = []
		coarse_delta_y = []

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
			start = signal.duration*trim
			stop = signal.duration*(1-trim)

			indices = (signal.time >= start) * (signal.time < stop)

			# Model captures
			x = signal.temporal_freq[indices] + tune
			y = np.abs(signal.iq)[indices]

			model_x.append(x)
			model_y.append(y)

			# We want 1 vs 3
			u_filter_fn = lambda x: x.trigger_number % len(signals) == i and x.channel_number == idx_a
			v_filter_fn = lambda x: x.trigger_number % len(signals) == i and x.channel_number == idx_b

			u_repeats = list(filter(u_filter_fn, captures))
			v_repeats = list(filter(v_filter_fn, captures))

			assert len(u_repeats) == len(v_repeats)
			assert all([x.center_freq == tune for x in u_repeats])
			assert all([x.center_freq == tune for x in v_repeats])

			# Estimate delay once, for channel a
			# DO NOT estimate delay individually
			# Thay would defy the point
			u_ = []
			v_ = []

			for u, v in zip(u_repeats, v_repeats):
				delay = signal.est.estimate(u.iq)

				u_.append( np.roll(u.iq, -delay) )
				v_.append( np.roll(v.iq, -delay) )

			u = np.vstack(u_)
			v = np.vstack(v_)

			coarse = np.angle( np.sum(u * v.conj(), 1) ).mean(0)
			deltas = np.angle( u * v.conj() )

			delta = deltas.mean(0)
			lower = deltas.min(0)
			upper = deltas.max(0)

			x = signal.temporal_freq[indices] + tune
			y = delta[indices]
			upper = upper[indices]
			lower = lower[indices]

			fine_delta_x.append(x)
			fine_delta_y.append(y)
			fine_delta_upper.append(upper)
			fine_delta_lower.append(lower)

			coarse_delta_x.append(tune)
			coarse_delta_y.append(coarse)


		# Second pass over the data to sort it - this deals with overlap
		#################################################################################################
		x = np.hstack(fine_delta_x)
		y = np.hstack(fine_delta_y)
		upper = np.hstack(fine_delta_upper)
		lower = np.hstack(fine_delta_lower)

		x, indices = np.sort(x), np.argsort(x)
		y = y[indices]
		upper = upper[indices]
		lower = lower[indices]

		self.fine_delta_x = x
		self.fine_delta_y = y
		self.fine_delta_upper = upper
		self.fine_delta_lower = lower

		self.coarse_delta_x = np.hstack(coarse_delta_x)
		self.coarse_delta_y = np.hstack(coarse_delta_y)

		self.chan_set = chan_set
		self.idx_a = idx_a
		self.idx_b = idx_b

		self.radians = radians

		self.model_x = np.hstack(model_x)
		self.model_y = np.hstack(model_y)

	def adc_ch_iterator(self):
		"""
		Iterate through channel numbers associated with respective x, y arrays

		x	test signal frequency in Hz (always the same)
		y	perceived signal level in ADC codes
		"""

		return zip(
			self.chan_set,
			self.adc_ch_x.values(),
			self.adc_ch_y.values()
		)

	def csv(self, filename, fine):
		"""
		Save phase delta to file
		"""

		cols = ["freq_hz"]

		if self.radians:
			cols.append(f"ch{self.idx_a}_minus_ch{self.idx_b}_phase_delta_radians")
			factor = 1.0
		else:
			cols.append(f"ch{self.idx_a}_minus_ch{self.idx_b}_phase_delta_degrees")
			factor = 180.0 / np.pi

		if fine:
			data = np.vstack([
				self.fine_delta_x,
				self.fine_delta_y * factor
			]).T
		else:
			data = np.vstack([
				self.coarse_delta_x,
				self.coarse_delta_y * factor
			]).T

		np.savetxt(
			filename,
			data,
			comments="",
			delimiter=",",
			header=",".join(cols)
		)

	def display(self):
		"""
		Display phase delta visually
		"""

		spectral = minmaxplot("Hz")
		spectral.xtitle("Частота")

		if self.radians:
			spectral.ytitle("Radians")
			factor = 1.0
		else:
			spectral.ytitle("Разность фаз (°)")
			factor = 180.0 / np.pi

		spectral.trace(
			self.fine_delta_x,
			self.fine_delta_y * factor,
			error_band=[self.fine_delta_upper * factor, self.fine_delta_lower * factor],
			name=f"CH{self.idx_a}−CH{self.idx_b} Fine"
		)

		spectral.trace(
			self.coarse_delta_x,
			self.coarse_delta_y * factor,
			name=f"CH{self.idx_a}−CH{self.idx_b} Coarse"
		)

		result = page([spectral])
		result.show()

parser = argparse.ArgumentParser(description="Produces a plot or a csv file of phase difference between two channels, with respect to frequency.")
parser.add_argument("--location", help="path to a directory containing captures+metadata with test signals fed into device under test", required=True)
parser.add_argument("--channels", help="specify a channel pair pattern a,b to be used when obtaining a phase delta; a will have b subtracted from it", required=True)
parser.add_argument("--trim", help="specify the proportion of pulse head and tail to be discarded in time domain to remove transients, defaults to 0.05")
parser.add_argument("--fine", help="[When --csv used] save fine curve instead of coarse, results in a lot more data", action="store_true")
parser.add_argument("--rad", help="use radians instead of degrees", action="store_true")
parser.add_argument("--csv", help="write results into a specified csv file")
args = parser.parse_args()

trim = float(args.trim or "0.05")
idx_a, idx_b = [int(x) for x in args.channels.split(",")]

a = PhaseDeltaPointsV1(args.location, idx_a, idx_b, trim=trim, radians=args.rad)

if args.csv:
	a.csv(args.csv, args.fine)
else:
	a.display()
