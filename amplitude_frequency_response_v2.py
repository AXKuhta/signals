
from glob import glob
import argparse
import json

import numpy as np

from deserializer import Model, Field

from src.misc import ad9910_sweep_bandwidth, parse_numeric_expr, roll_lerp
from src.delay import SpectralDelayEstimator
from src.orda import StreamORDA
import src.delay as delay
import src.dds as dds

#
# We have domain models:
# - JsonDDCAndCalibratorV1
# 	- JsonDDCSettingsV1
# 	- JsonSignalV1
#
# They might as well not exist
# Are for storage representation and have no methods
# Their hierarchy is unsuitable for application tasks
#
# Then we have actual application classes:
# - SignalV1	A self-contained DDC capture model
#
# The class hierarchy becomes inverted to meet
# the application demands
#
# Who makes SignalV1?
#

class JsonDDCSettingsV1(Model):
	config_dir = Field(str)
	samplerate = Field(str)
	frames = Field(int)

class JsonSignalV1(Model):
	tune = Field(str)
	level = Field(str)
	emit = Field(str)

class JsonDDCAndCalibratorV1(Model):
	ddc = Field(JsonDDCSettingsV1)
	signals = Field([JsonSignalV1])

class SignalV1:
	"""
	Joint DDC + Calibrator signal modelling application class

	Self-contained and context-independent
	"""

	# Models have associated DDC settings and the signal descriptor
	descriptor = None
	ddc = None

	# Lack of context regarding other signals can cause duplicate work-
	# most signals are literally the same
	est = None
	time = None
	delay = None
	duration = None
	temporal_freq = None
	spectral_freq = None

	def __init__(self, descriptor, ddc, trim=0.05):
		"""
		Prepares a model signal as DDC would see it

		trim	Portion of pulse head+tail to be discarded so as to remove transients
			Default: 0.05
		"""

		self.ddc = ddc
		self.descriptor = descriptor

		sysclk = parse_freq_expr("1 GHz")

		rate = parse_freq_expr(ddc.samplerate)
		frames = ddc.frames

		capture_duration = frames / rate

		tokens = descriptor.emit.split(" ")

		if tokens[0] == "sweep":
			sweep, offset, offset_unit, duration, duration_unit, freq, freq_unit, a, b = tokens

			delay = parse_time_expr(f"{offset} {offset_unit}")
			duration = parse_time_expr(f"{duration} {duration_unit}")
			center_frequency = parse_freq_expr(f"{freq} {freq_unit}")
			a = int(a)
			b = int(b)

			#
			# Prepare model signal
			# TODO: support off-center tuning
			#
			time = dds.time_series(rate, capture_duration)
			band = ad9910_sweep_bandwidth(a, b, sysclk=sysclk)

			model_sweep = dds.sweep(time, -band/2, band/2, 0, duration)
			temporal_freq = (time / duration)*band - band/2
			spectral_freq = np.linspace(-rate/2, rate/2, frames)

			#
			# Prepare delay estimator
			#
			# We could have some kind of heuristic to decide on the estimator,
			# for example having at least 100 bins occupied to pick SpectralDelayEstimator:
			# bins_occupied = int( torch.unique(freqs - freqs % (rate/frames)).shape[0] )
			#
			# But lets have it simpler:
			# - Sweeps have SpectralDelayEstimator - which assumes contiguous indices_est
			# - Pulses have ConvDelayEstimator

			# Establish the frequencies at truncated head/tail
			start = duration*trim + delay
			stop = duration*(1-trim) + delay

			temporal_indices = (time >= start) * (time < stop)

			min_freq = temporal_freq[temporal_indices].min()
			max_freq = temporal_freq[temporal_indices].max()

			indices_est = (spectral_freq >= min_freq)*(spectral_freq < max_freq)
			est = SpectralDelayEstimator(model_sweep, indices_est)

			self.est = est
			self.time = time
			self.delay = delay
			self.duration = duration
			self.temporal_freq = temporal_freq
			self.spectral_freq = spectral_freq
		else:
			assert 0

	def eliminate_delay(self, iq):
		"""
		Remove time delay from a capture

		TODO: shoud this actually be "fit()"?
		"""
		sample_delay = self.est.estimate(iq)

		# Use simple roll for now
		return np.roll( iq, round(-sample_delay.item()) )


def parse_time_expr(expr, into="s"):
	inv_factors = {
		"s": 1,
		"ms": 1000,
		"us": 1000000,
		"ns": 1000000000
	}

	value, unit = expr.split(" ")

	if "." in value:
		value = float(value)
	else:
		value = int(value)

	inv_factor = inv_factors[ unit.lower() ] // inv_factors[ into.lower() ]

	return value / inv_factor

def parse_freq_expr(expr, into="hz"):
	"""
	Parse a frequency.

	Examples:

	>>> parse_freq_expr("150 MHz")
	150000000
	>>> parse_freq_expr("150.1 MHz")
	150100000.0

	Returns an int whenever possible.
	"""

	factors = {
		"hz": 1,
		"khz": 1000,
		"mhz": 1000000,
		"ghz": 1000000000
	}

	value, unit = expr.split(" ")

	if "." in value:
		value = float(value)
	else:
		value = int(value)

	factor = factors[ unit.lower() ] // factors[ into.lower() ]

	return value * factor

#
# Amplitude Frequency Response the 2nd:
#
# Two modes of operation:
# - Without a reference
# - Adjusted
#
# The reference-free mode:
# - Prep
# 	- Take a directory
#	- Deserialize the preset
#	- Glob the .ISEs
# - Prep 2
#	- Model signals from the preset
#	- Estimate and eliminate delay
# - Frequency Response
#	- Associate time with frequency
#	- Retain, say, 0.9 of time
#		- Previously we've been retaining an explicitly specified frequency band
#			- This creates ambiguity for fixed frequency pulses
#				- They do not really have a band
#	- Into the point array
#	- Sort it
#	- Plot it
#
# Whenever the adjusted mode is used:
# - Two instances of FrequencyResponsePointsV1
# - Assert pulse set is the same
# - That way point array X is absolutely surely the same
# 	- Do not want to deal with resampling
#		- Though it aint that hard
#			- bins `x - x % 0.1`
#			- new X np.unique(...)
#			- gather x[b == 0.1]
#				- for loop
#				- stack
# - 20log10 before plotting
#


class FrequencyResponsePointsV1:
	"""
	Application class to translate a folder of captures+metadata into frequency response points (x, y)
	"""

	def __init__(self, location, trim=0.05):
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
		print(len(chan_set), "channels active")

		# Channel to points mapping
		points = { chan: {"x": [], "y": []} for chan in chan_set }

		signals = []

		for descriptor in preset.signals:
			signals.append( SignalV1(descriptor, preset.ddc) )

		# Onto the actual processing
		# Delay elimination + amplitude + averaging
		#################################################################################################
		for i, signal in enumerate(signals):
			tune = parse_freq_expr(signal.descriptor.tune)

			# Pulse cropping
			start = signal.duration*trim + signal.delay
			stop = signal.duration*(1-trim) + signal.delay

			indices = (signal.time >= start) * (signal.time < stop)

			for channel in chan_set:
				filter_fn = lambda x: x.trigger_number % len(signals) == i and x.channel_number == channel

				repeats = list(filter(filter_fn, captures))

				assert all([x.center_freq == tune for x in repeats])

				a = [signal.eliminate_delay(x.iq) for x in repeats]
				a = np.abs(np.vstack(a))

				mampl = a.mean(0)
				lower, _ = a.min(0), np.argmin(0)
				upper, _ = a.max(0), np.argmax(0)

				x = signal.temporal_freq[indices] + tune
				y = mampl[indices]

				lower = lower[indices]
				upper = upper[indices]

				points[channel]["x"].append(x)
				points[channel]["y"].append(y)

		from src.display import minmaxplot, page

		spectral = minmaxplot("Hz")

		for chan, pts in points.items():
			x = np.hstack(pts["x"])
			y = np.hstack(pts["y"])

			x, indices = np.sort(x), np.argsort(x)
			y = y[indices]

			spectral.trace(x, y)

		result = page([spectral])
		result.show()

FrequencyResponsePointsV1("/media/pop/2e7a55b1-cee8-4dd6-a513-4cd4c618a44e/calibrator_data_v1/calibrator_v1_2025-03-12T06_58_03+00_00")
