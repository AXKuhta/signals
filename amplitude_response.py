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
	RAW = auto()
	MV = auto()
	MODEL = auto()
	REFERENCED = auto()

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

		# Keep track of what pulses had the most captures
		max_h = 0

		# First pass over the data
		# Pulse modelling + filtering + delay elimination
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

			# Deal with channels and repeated captures
			for channel in chan_set:
				filter_fn = lambda x: x.trigger_number % len(signals) == i and x.channel_number == channel

				repeats = list(filter(filter_fn, captures))

				assert all([x.center_freq == tune for x in repeats])

				x = signal.temporal_freq[indices] + tune
				y = np.vstack(
					[signal.eliminate_delay(x.iq)[indices] for x in repeats]
				)

				adc_ch_x[channel].append(x)
				adc_ch_y[channel].append(y)

				if len(repeats) > max_h:
					max_h = len(repeats)

		# Second pass over the data to pool it, sort and bin it
		# Some pulses may have less captures than others; this is deat with:
		# - By zero padding
		# - And a mask
		#
		# Total memory utilization per channel is in the ballpark of:
		# - Captures:
		#	Five minute session
		#	25 captures per second
		#	4096 samples per capture
		#	complex128 samples = 16 bytes
		#	about 491 MB
		#
		# - Mask:
		#	bools = 1 byte
		#	about 30 MB
		#################################################################################################
		for chan in chan_set:
			# x will stack no problem
			# y needs padding

			x = np.hstack(adc_ch_x[chan]) # Frequency values
			y = [] # Amplitude values
			m = [] # Legitimate/padding mask

			for pulse in adc_ch_y[chan]:
				h, w = pulse.shape
				u = np.zeros([max_h, w], dtype=np.complex128)
				v = np.zeros([max_h, w], dtype=np.bool)

				u[:h] = pulse
				v[:h] = True

				y.append(u)
				m.append(v)

			y = np.abs(np.hstack(y))
			m = np.hstack(m)

			# Sort so frequency is monotonic
			x, indices = np.sort(x), np.argsort(x)
			y = y[:, indices]
			m = m[:, indices]

			# Binning
			roundto = 10000
			repeat = np.round(x/roundto)*roundto
			unique, indices = np.unique(repeat, return_index=True)

			# Digitize assigns index of 0 to values that are out-of-bounds
			# e.g. fall outside the lowest bin
			indices = np.digitize(x, unique)

			assert not np.any(indices == 0), "bin bugcheck"

			mampl = []
			lower = []
			upper = []

			for i in range( 1, len(unique) ):
				ind1 = np.where(indices == i)
				sub1 = y[:, ind1]
				mub1 = m[:, ind1]

				ind2 = np.where(mub1)
				bin = sub1[ind2]

				mampl.append(np.mean(bin))
				lower.append(np.min(bin))
				upper.append(np.max(bin))

			# For 849 frequency points
			# There will be 848 bins
			# Align frequencies to bin centers
			adc_ch_x[chan] = unique[1:] - roundto/2
			adc_ch_y[chan] = mampl

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

	def csv(self, mode, filename, reference=None):
		"""
		Save frequency response to file

		Mode.RAW		Encode adc codes - that is |q(t)| where q is iq, vs f(t) where f is frequency given time, as csv data
		Mode.MV			Save the estimate of signal level in mV rms as a function of frequency, as csv data
		Model.MODEL		Save the ratio of actual signal level to model signal level as csv data;
					Tries to compensate for DDC's and DDS's frequency response-
					but may be imperfect
		Model.REFERENCED	Save power gain against a reference as csv data
		"""

		x = []
		y = []
		cols = []

		if mode == Mode.RAW:
			x = list(self.adc_ch_x.values())
			y = list(self.adc_ch_y.values())
			cols = [f"ch{v}_adc" for v in self.chan_set]

		elif mode == Mode.MV:
			for chan, x_, y_ in self.adc_ch_iterator():
				factor = ddc_cost_mv(x_)

				x.append(x_)
				y.append(y_*factor)
				cols.append(f"ch{chan}_mv")

		elif mode == Mode.MODEL:
			for chan, x_, y_ in self.adc_ch_iterator():
				ratio = 20*np.log10( y_ / ( self.model_y * self.attenuation) )
				x.append(x_)
				y.append(ratio)
				cols.append(f"ch{chan}_db")

		elif mode == Mode.REFERENCED:
			assert self.chan_set == reference.chan_set, "Channel set mismatch between datasets"
			assert self.model_x.shape == reference.model_x.shape, "Frequency set mismatch between datasets"

			for chan, x_, u, v in zip(
				self.chan_set,
				self.adc_ch_x.values(),
				self.adc_ch_y.values(),
				reference.adc_ch_y.values()
			):
				ratio = 20*np.log10( u / (v * reference.attenuation) )
				x.append(x_)
				y.append(ratio)
				cols.append(f"ch{chan}_db")


		assert all([np.all(v == x[0]) for v in x])

		np.savetxt(
			filename,
			np.vstack([x[0]] + y).T,
			comments="",
			delimiter=",",
			header=",".join(["freq_hz"] + cols)
		)

	def display(self, mode, reference=None):
		"""
		Display the frequency response visually

		Mode.RAW 		Trace adc codes - that is |q(t)| where q is iq, vs f(t) where f is frequency given time
		Mode.MV			Trace the estimate of signal level in mV rms as a function of frequency
		Mode.MODEL		Trace the ratio of actual signal level to model signal level;
					Tries to compensate for DDC's and DDS's frequency response-
					but may be imperfect
		Mode.REFERENCED		Trace power gain against a reference
		"""

		spectral = minmaxplot()
		spectral.xtitle("Частота (МГц)")
		spectral.xexponent(False)
		spectral.yexponent(True, scientific=True)

		if mode == Mode.RAW:
			spectral.ytitle("Код АЦП")

			for chan, x, y in self.adc_ch_iterator():
				spectral.trace(x, y, name=f"Канал {chan}")

		elif mode == Mode.MV:
			spectral.ytitle("mV")

			# Postprocessing: voltage scale in mv
			# This also removes the DDC's overall influence on frequency response
			for chan, x, y in self.adc_ch_iterator():
				spectral.trace(x, y * ddc_cost_mv(x), name=f"Канал {chan}")

		elif mode == Mode.MODEL:
			spectral.ytitle("dB")

			for chan, x, y in self.adc_ch_iterator():
				ratio = 20*np.log10( y / (self.model_y * self.attenuation) )
				spectral.trace(x, ratio, name=f"Канал {chan}")

		elif mode == Mode.REFERENCED:
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
				spectral.trace(x, ratio, name=f"Канал {chan}")

		else:
			assert 0

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
		a.csv(Mode.REFERENCED, args.csv, b)
	else:
		a.display(Mode.REFERENCED, b)
elif args.dut:
	assert (args.model or args.raw or args.mv), "either --raw or --mv or --model must be specified with no --ref"
	assert not (args.model and args.raw), "--raw and --model are mutually exclusive"
	assert not (args.model and args.mv), "--mv and --model are mutually exclusive"

	a = FrequencyResponsePointsV1(args.dut, trim=trim, attenuation=attenuation)

	if args.csv:
		if args.model:
			a.csv(Mode.MODEL, args.csv)
		else:
			if args.mv:
				a.csv(Mode.MV, args.csv)
			else:
				a.csv(Mode.RAW, args.csv)
	else:
		if args.model:
			a.display(Mode.MODEL)
		else:
			if args.mv:
				a.display(Mode.MV)
			else:
				a.display(Mode.RAW)
else:
	assert 0
