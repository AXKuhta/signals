
from glob import glob
import argparse
import json

import torch

from deserializer import Model, Field

from src.misc import ad9910_sweep_bandwidth, parse_numeric_expr, roll_lerp
from src.delay import SpectralDelayEstimator
from src.orda import StreamORDA
import src.delay as delay
import src.dds as dds


class DDCSettingsV1(Model):
	config_dir = Field(str)
	samplerate = Field(str)
	frames = Field(int)

class SignalV1(Model):
	tune = Field(str)
	level = Field(str)
	emit = Field(str)

class DDCAndCalibratorV1SignalModel():
	delay = None
	duration = None

	time = None
	iq = None
	est = None

	temporal_freq = None
	spectral_freq = None

	temporal_indices = None

	def eliminate_delay(self, iq):
		"""
		Remove time delay from a capture

		TODO: shoud this actually be "fit()"?
		"""
		sample_delay = self.est.estimate(iq)

		# Use simple roll for now
		return iq.roll( round(-sample_delay.item()) )

class DDCAndCalibratorV1(Model):
	ddc = Field(DDCSettingsV1)
	signals = Field([SignalV1])

	# Signals have associated models
	#
	# We cannot:
	# - Have SignalV1.model() - lack required information
	#
	# We can:
	# - Have SubclassSignalV1.model()
	#	- Disadvantage: lacks context
	#		- Most signals the same
	#		- Duplicate work
	# - Have modelling happen here
	#	- Where do the results go?
	# 		- Late init attributes in SignalV1
	# 		- Distinct DDCAndCalibratorV1SignalModel
	#
	# Sticking to distinct object
	models = None

	def init_models(self, trim=0.05):
		"""
		Walks the signal list in the preset and prepares a model signal as DDC would have it

		trim	Portion of pulse head+tail to be discarded so as to remove transients
			Default: 0.05
		"""

		sysclk = parse_freq_expr("1 GHz")

		rate = parse_freq_expr(self.ddc.samplerate)
		frames = self.ddc.frames

		capture_duration = frames / rate

		models = []

		# Modelling as DDC would have it
		# TODO: Extract a set of parameters that matter, prep only that, propagate
		for signal in self.signals:
			tokens = signal.emit.split(" ")

			if tokens[0] == "sweep":
				sweep, offset, offset_unit, duration, duration_unit, freq, freq_unit, a, b = tokens

				delay = parse_time_expr(f"{offset} {offset_unit}")
				duration = parse_time_expr(f"{duration} {duration_unit}")
				center_frequency = parse_freq_expr(f"{freq} {freq_unit}")
				a = int(a)
				b = int(b)

				#
				# Prepare model signal
				#
				time = dds.time_series(rate, capture_duration)
				band = ad9910_sweep_bandwidth(a, b, sysclk=sysclk)

				model_sweep = dds.sweep(time, -band/2, band/2, 0, duration)
				temporal_freq = (time / duration)*band - band/2
				spectral_freq = torch.linspace(-rate/2, rate/2, frames)

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

				start = duration*trim + delay
				stop = duration*(1-trim) + delay

				temporal_indices = (time >= start) * (time < stop)

				min_freq = temporal_freq[temporal_indices].min()
				max_freq = temporal_freq[temporal_indices].max()

				indices_est = (spectral_freq >= min_freq)*(spectral_freq < max_freq)
				est = SpectralDelayEstimator(model_sweep, indices_est)

				model = DDCAndCalibratorV1SignalModel()

				model.est = est
				model.temporal_freq = temporal_freq
				model.temporal_indices = temporal_indices

				models.append(model)
			else:
				assert 0

		self.models = models

		print(len(self.models), "models initialized")


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
# Now systematized, streamlined, refined
#
# Two modes of operation:
# - Without a reference
# - Adjusted
#
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
#
# Two mode architecture:
# - A class FrequencyResponsePoints
#	- Could have been a function too
#		- But no access to preset
#	- The constructor takes:
#		- directory?
#		- preset obj + captures?
#	- x
#	- y
#


class FrequencyResponsePointsV1:
	"""
	Application class to translate a folder of captures+metadata into frequency response points (x, y)
	"""

	def __init__(self, location):
		with open(f"{location}/preset.json") as f:
			obj = json.load(f)

		preset = DDCAndCalibratorV1.deserialize(obj["ddc-and-calibrator-v1"])

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

		preset.init_models()

		# Channel to points mapping
		points = { chan: {"x": [], "y": []} for chan in chan_set }

		# Onto the actual processing
		# Delay elimination + amplitude + averaging
		#################################################################################################
		for i, (signal, model) in enumerate(zip(preset.signals, preset.models)):
			for channel in chan_set:
				filter_fn = lambda x: x.trigger_number % len(preset.signals) == i and x.channel_number == channel

				repeats = list(filter(filter_fn, captures))

				assert all([x.center_freq == parse_freq_expr(signal.tune) for x in repeats])

				a = [model.eliminate_delay(x.iq) for x in repeats]
				a = torch.vstack(a).abs()

				mampl = a.mean(0)
				lower, _ = a.min(0)
				upper, _ = a.max(0)

				indices = model.temporal_indices

				x = model.temporal_freq[indices] + parse_freq_expr(signal.tune)
				y = mampl[indices]

				lower = lower[indices]
				upper = upper[indices]

				points[channel]["x"].append(x)
				points[channel]["y"].append(y)

		from src.display import minmaxplot, page

		spectral = minmaxplot("Hz")

		for chan, pts in points.items():
			x = torch.hstack(pts["x"])
			y = torch.hstack(pts["y"])

			x, indices = torch.sort(x)
			y = y[indices]

			spectral.trace(x, y)

		result = page([spectral])
		result.show()

FrequencyResponsePointsV1("/media/pop/2e7a55b1-cee8-4dd6-a513-4cd4c618a44e/calibrator_data_v1/calibrator_v1_2025-03-12T06_58_03+00_00")
