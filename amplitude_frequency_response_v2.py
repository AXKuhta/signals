
from glob import glob
import argparse
import json

import numpy as np

from src.misc import ad9910_sweep_bandwidth, parse_numeric_expr, parse_time_expr, parse_freq_expr, roll_lerp, ddc_cost_mv
from src.delay import SpectralDelayEstimator
from src.display import minmaxplot, page
from src.touchstone import S2PFile
from src.orda import StreamORDA
import src.delay as delay
import src.dds as dds

from src.workflows.v1 import ModelSignalV1
from src.schemas.v1 import *

class FrequencyResponsePointsV1:
	"""
	Application class to translate a folder of captures+metadata into frequency response points (x, y)

	The class should support these display/csv scenarios:
	- Raw ADC codes
	- Signal level estimate
	- dB against a model
	- dB against a reference

	The pipeline design, top down:
	- Read the metadata
		- Parse JSON
		- Verify schema
	- Read the captures
		- Glob all .ISEs
		- Discard 0 Hz captures
	- Detect active channels
	- Pool amplitude-from-frequency points across descriptors
		- Prepare a model signal
		- Find associated captures
		- For every active channel
			- Average amplitude across repeated captures
				- Estimate delay
				- Eliminate delay
				- Find amplitude
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

	def __init__(self, location, trim=0.05, attenuation=1.0):
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
		self.chan_set = chan_set

		self.model_x = np.hstack(model_x)
		self.model_y = np.hstack(model_y)

		self.attenuation = attenuation

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

	def write_raw(self, filename):
		"""
		Encode adc codes - that is |q(t)| where q is iq, vs f(t) where f is frequency given time, as csv data
		"""

		x = list(self.adc_ch_x.values())[0]
		y = list(self.adc_ch_y.values())

		table = np.vstack([x] + y).T

		np.savetxt(filename, table, comments="", delimiter=",", header=",".join(["freq_hz"] + [f"ch{v}" for v in self.chan_set]))

	def display_raw(self):
		"""
		Trace adc codes - that is |q(t)| where q is iq, vs f(t) where f is frequency given time
		"""

		spectral = minmaxplot("Hz")
		spectral.xtitle("Frequency")
		spectral.ytitle("ADC code")

		for chan, x, y in self.adc_ch_iterator():
			spectral.trace(x, y, name=f"Channel {chan}")

		result = page([spectral])
		result.show()

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

	def display_db_vs_model(self):
		"""
		Trace the ratio of actual signal level to model signal level

		Tries to compensate for DDC's and DDS's frequency response - but may be imperfect
		"""

		spectral = minmaxplot("Hz")
		spectral.xtitle("MHz")
		spectral.ytitle("dB")

		def inv_sinc(x):
			x = x / 1000 / 1000 / 1000
			return np.pi * x / np.sin(np.pi * x)

		for chan, x, y in self.adc_ch_iterator():
			ratio = 20*np.log10(y / ( self.model_y * self.attenuation * inv_sinc(self.model_x) ) )
			spectral.trace(x, ratio, name=f"Channel {chan}")

		result = page([spectral])
		result.show()

	def display_db_referenced(self, reference):
		"""
		Trace power gain against a reference
		"""

		spectral = minmaxplot("Hz")
		spectral.xtitle("MHz")
		spectral.ytitle("dB")

		assert self.chan_set == reference.chan_set, "Channel set mismatch between datasets"
		assert self.model_x.shape == reference.model_x.shape, "Frequency set mismatch between datasets"

		for chan, x, u, v in zip(
			self.chan_set,
			self.adc_ch_x.values(),
			self.adc_ch_y.values(),
			reference.adc_ch_y.values()
		):
			ratio = 20*np.log10( u / (v * reference.attenuation) )
			spectral.trace(x, ratio, name=f"Channel {chan}")

		result = page([spectral])
		result.show()

parser = argparse.ArgumentParser(description="Produces a plot or a csv file of amplitude frequency response.")
parser.add_argument("--dut", help="path to a directory containing captures+metadata with test signals fed into device under test", required=True)
parser.add_argument("--ref", help="path to a directory containing captures+metadata with reference signals (device under test bypassed)")
parser.add_argument("--trim", help="specify the proportion of pulse head and tail to be discarded in time domain to remove transients, defaults to 0.05")
parser.add_argument("--offset", help="specify how much extra gain or attenuation of reference signals should be factored in, e.g. 0.00498 or \"-46 dB\"")
parser.add_argument("--csv", help="write results into a specified csv file")
parser.add_argument("--model", help="[when no --ref] use an approximate model of reference signals instead of actual reference captures", action="store_true")
parser.add_argument("--raw", help="[when no --ref] display signal level in |iq| adc codes", action="store_true")
parser.add_argument("--mv", help="[when no --ref] display signal level in volts", action="store_true")
args = parser.parse_args()

attenuation = float(args.offset or "1.0")
trim = float(args.trim or "0.05")

if args.dut and args.ref:
	assert not args.model, "--model cannot be used with --ref"
	assert not args.raw, "--raw cannot be used with --ref"
	assert not args.mv, "--mv cannot be used with --ref"

	a = FrequencyResponsePointsV1(args.dut, trim=trim)
	b = FrequencyResponsePointsV1(args.ref, trim=trim, attenuation=attenuation)

	if args.csv:
		assert 0
	else:
		a.display_db_referenced(b)
elif args.dut:
	assert (args.model or args.raw or args.mv), "either --raw or --mv or --model must be specified with no --ref"
	assert not (args.model and args.raw), "--raw and --model are mutually exclusive"
	assert not (args.model and args.mv), "--mv and --model are mutually exclusive"

	a = FrequencyResponsePointsV1(args.dut, trim=trim, attenuation=attenuation)

	if args.csv:
		if args.model:
			assert 0
		else:
			if args.mv:
				assert 0
			else:
				a.write_raw(args.csv)
	else:
		if args.model:
			a.display_db_vs_model()
		else:
			if args.mv:
				a.display_mv()
			else:
				a.display_raw()
else:
	assert 0
