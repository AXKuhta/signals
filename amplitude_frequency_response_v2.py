
from glob import glob
import argparse
import json

import numpy as np

from src.misc import ad9910_sweep_bandwidth, parse_numeric_expr, parse_time_expr, parse_freq_expr, roll_lerp, ddc_cost_mv
from src.delay import SpectralDelayEstimator
from src.display import minmaxplot, page
from src.orda import StreamORDA
import src.delay as delay
import src.dds as dds

#
# What's missing:
# - Referenced mode
#

from src.workflows.v1 import ModelSignalV1

from src.schemas.v1 import *
#from src.models.v1 import *

parser = argparse.ArgumentParser()
parser.add_argument("--dut", help="path to a directory containing captures+metadata with test signals fed through device under test", required=True)
parser.add_argument("--ref", help="path to a directory containing captures+metadata with reference signals (device under test bypassed)")
args = parser.parse_args()

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
# - Extra method FrequencyResponsePointsV1
# - Assert pulse set is the same
# - No reason whatsoever to support different pulse sets
# 	- While possible, do not want to deal with resampling
#		- Though it may be less difficult than it seems
#			- bins `x - x % 0.1`
#			- new X np.unique(...)
#			- gather x[b == 0.1]
#				- for loop
#				- stack
# - Would entail a second instance of FrequencyResponsePointsV1
# - Stick to single instance of FrequencyResponsePointsV1
#


class FrequencyResponsePointsV1:
	"""
	Application class to translate a folder of captures+metadata into frequency response points (x, y)

	The class should support three display scenarios:
	- Raw ADC codes
	- Signal level estimate
	- dB against a model
	- dB against a reference - done elsewhere
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

		# Point storage for:
		# - ADC's perceived level, channel wise, across frequencies
		# - Model level, across frequencies
		adc_ch_x = { chan: [] for chan in chan_set }
		adc_ch_y = { chan: [] for chan in chan_set }

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

			# Actual captures
			# Deal with:
			# - Different channels
			# - Accumulation
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

				adc_ch_x[channel].append(x)
				adc_ch_y[channel].append(y)

		# Second pass over the data to sort it - this deals with overlap
		#################################################################################################
		for chan in chan_set:
			x = np.hstack(adc_ch_x[chan])
			y = np.hstack(adc_ch_y[chan])

			x, indices = np.sort(x), np.argsort(x)
			y = y[indices]

			adc_ch_x[chan] = x
			adc_ch_y[chan] = y

		self.adc_ch_x = adc_ch_x
		self.adc_ch_y = adc_ch_y

		self.model_x = np.hstack(model_x)
		self.model_y = np.hstack(model_y)

	def adc_ch_iterator(self):
		"""
		Iterate through channel numbers associated with respective x, y arrays

		x	test signal frequency in Hz (always the same)
		y	perceived signal level in ADC codes
		"""

		return zip(
			self.adc_ch_x.keys(),
			self.adc_ch_x.values(),
			self.adc_ch_y.values()
		)

	def display_mv(self):
		"""
		Trace the estimate of signal level in mV rms as a function of frequency
		"""

		spectral = minmaxplot("Hz")
		spectral.xtitle("MHz")
		spectral.ytitle("mV")

		for chan, x, y in self.adc_ch_iterator():
			# Postprocessing: voltage scale in mv
			# This also removes the DDC's overall influence on frequency response
			factor = ddc_cost_mv(x)

			spectral.trace(x, y * factor, name=f"Channel {chan}")

		result = page([spectral])
		result.show()

	def display_db_unreferenced(self):
		"""
		Trace the ratio of actual signal level to model signal level
		"""

		spectral = minmaxplot("Hz")
		spectral.xtitle("MHz")
		spectral.ytitle("dB")

		for chan, x, y in self.adc_ch_iterator():
			ratio = 10*np.log10(y / self.model_y)
			spectral.trace(x, ratio, name=f"Channel {chan}")

		result = page([spectral])
		result.show()


x = FrequencyResponsePointsV1(args.dut)
x.display_db_unreferenced()
