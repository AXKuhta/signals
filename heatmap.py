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

import matplotlib.pyplot as plt

#
# Heatmap:
# - The output must be a matrix
#
# The program:
# - Extracts the pulse body
#	- So model, etc
# - But does no averaging
#
# - 100k frequencies too much for a picture
# - must be binned
# - so individual captures get binned
# - bins get averaged
# - bin content gets piecewise lerped
#
# bin selection:
# - linspace
# - roundto
#

class FrequencyResponsePointsV1:
	def __init__(self, location, trim=0.05, attenuation=1.0):
		with open(f"{location}/preset.json") as f:
			obj = json.load(f)

		preset = JsonDDCAndCalibratorV1.deserialize(obj["ddc-and-calibrator-v1"])

		captures = []

		# Hack
		self.bins = np.linspace(154*1000*1000, 162*1000*1000, 1024)

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
			capture.trigger_number = i // len(chan_set) + 1

		print("Loaded", len(captures), "captures")
		print(len(chan_set), "channels active")

		# Be careful
		# [ [] ]*1024 references the same list 1024 times
		bins_ch_x = { chan: [ [] for i in range(1025) ] for chan in chan_set } # timestamp[][1025]; [0] for lhs discard, [1024] for rhs discard
		bins_ch_y = { chan: [ [] for i in range(1025) ] for chan in chan_set } # amplitude[][1025]; [0] for lhs discard, [1024] for rhs discard

		signals = []

		for descriptor in preset.signals:
			signals.append( ModelSignalV1(descriptor, preset.ddc) )

		# Iterate over captures not pulses
		for capture in captures:
			signal_idx = (capture.trigger_number - 1) % len(signals)
			signal = signals[signal_idx]

			tune = parse_freq_expr(signal.descriptor.tune)

			# Pulse cropping
			start = signal.duration*trim
			stop = signal.duration*(1-trim)

			indices = (signal.time >= start) * (signal.time < stop)

			x = signal.temporal_freq[indices] + tune
			y = signal.eliminate_delay(capture.iq)[indices]
			y = np.abs(y)

			# We must digitize right here
			ind = np.digitize(x, self.bins)

			# https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function
			uniq, ind2 = np.unique(ind, return_index=True)
			y = np.split(y, ind2[1:])

			assert len(uniq) == len(y)

			x_ = []
			y_ = []

			print(capture.timestamp)

			for bin_idx, bin_content in zip(uniq, y):
				bins_ch_x[capture.channel_number][bin_idx].append( capture.timestamp.timestamp() )
				bins_ch_y[capture.channel_number][bin_idx].append( np.mean(bin_content) )

				#x_.append(self.bins[bin_idx])
				#y_.append(np.mean(bin_content))

			#plt.plot(x_, y_)
			#plt.show()

			#if (capture.trigger_number>200):
			#	break

		# Timestamp and amplitude bins now populated
		# But timestamps are unevenly spaced
		# Heatmap time is evenly spaced
		time_from = bins_ch_x[1][1][1]
		time_to = bins_ch_x[1][1][-1]
		time = np.linspace(time_from, time_to, 4096)

		picture_a = [ [] for i in range(1025) ]
		picture_b = [ [] for i in range(1025) ]

		# Sample heatmap time bin-wise using lerp
		for i in range(1025):
			picture_a[i] = np.interp(time, bins_ch_x[1][i], bins_ch_y[1][i])
			picture_b[i] = np.interp(time, bins_ch_x[3][i], bins_ch_y[3][i])

		fig, (ax1, ax2) = plt.subplots(2)

		ax1.imshow(picture_a)
		ax2.imshow(picture_b)
		plt.show()

#x = FrequencyResponsePointsV1("/media/pop/2e7a55b1-cee8-4dd6-a513-4cd4c618a44e/calibrator_data_field_trip_2/calibrator_v1_2025-07-09T07_21_28+00_00/")
x = FrequencyResponsePointsV1("/media/pop/2e7a55b1-cee8-4dd6-a513-4cd4c618a44e/calibrator_data_field_trip_2/calibrator_v1_2025-07-09T14_04_49+00_00/")
