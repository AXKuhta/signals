
from datetime import datetime, timedelta
from glob import glob
import json

import matplotlib.pyplot as plt
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

"""
calibrator_v1_2025-07-09T07_21_28+00_00		154-162 МГц через систему фазирования
						(Входы ВУПа в обычном порядке)

calibrator_v1_2025-07-09T07_52_22+00_00		154-162 МГц через систему фазирования
						(Входы ВУПа в обратном порядке) 

calibrator_v1_2025-07-09T09_10_07+00_00		154-162 МГц через систему фазирования
						(Входы ВУПа в обратном порядке) 

calibrator_v1_2025-07-09T09_22_32+00_00		154-162 МГц через систему фазирования
						(Входы ВУПа в обычном порядке)  

calibrator_v1_2025-07-09T10_00_50+00_00		154-162 МГц через систему фазирования
						(Входы ВУПа в обычном порядке)  

calibrator_v1_2025-07-09T10_22_12+00_00         154-162 МГц через систему фазирования
						(Входы ВУПа в обратном порядке) 

calibrator_v1_2025-07-09T11_06_52+00_00		154-162 МГц через систему фазирования
						(Входы ВУПа в обратном порядке) 

calibrator_v1_2025-07-09T11_27_10+00_00         154-162 МГц через систему фазирования
						(Входы ВУПа в обычном порядке)

calibrator_v1_2025-07-09T13_20_34+00_00		154-162 МГц через систему фазирования
						(Входы ВУПа в обычном порядке)

calibrator_v1_2025-07-09T13_31_29+00_00		154-162 МГц через систему фазирования
						(Входы ВУПа в обратном порядке)
						Запись ~30 минут

calibrator_v1_2025-07-09T14_04_49+00_00         154-162 МГц через систему фазирования
						(Входы ВУПа в обычном порядке)
						Запись ~25 минут

calibrator_v1_2025-07-09T14_36_15+00_00		154-162 МГц через систему фазирования
						(Входы ВУПа в обратном порядке)
						Запись ~25 минут

calibrator_v1_2025-07-09T15_04_59+00_00		154-162 МГц через систему фазирования
						(Входы ВУПа в обычном порядке)
						Запись ~25 минут

calibrator_v1_2025-07-09T15_37_30+00_00         154-162 МГц через систему фазирования
						(Входы ВУПа в обратном порядке)
						Запись ~25 минут
"""

capdir = "/media/pop/2e7a55b1-cee8-4dd6-a513-4cd4c618a44e/calibrator_data_field_trip_2/"

sessions = [
	{ # Flip 1
		"forward": "calibrator_v1_2025-07-09T07_21_28+00_00",
		"inverse": "calibrator_v1_2025-07-09T07_52_22+00_00",
	},
	{ # Flip 2
		"inverse": "calibrator_v1_2025-07-09T09_10_07+00_00",
		"forward": "calibrator_v1_2025-07-09T09_22_32+00_00"
	},
	{ # Flip 3
		"forward": "calibrator_v1_2025-07-09T10_00_50+00_00",
		"inverse": "calibrator_v1_2025-07-09T10_22_12+00_00"
	},
	{ # Flip 4
		"inverse": "calibrator_v1_2025-07-09T11_06_52+00_00",
		"forward": "calibrator_v1_2025-07-09T11_27_10+00_00"
	},
	{ # Flip 5
		"forward": "calibrator_v1_2025-07-09T13_20_34+00_00",
		"inverse": "calibrator_v1_2025-07-09T13_31_29+00_00"
	},
	{ # Flip 6 + long run
		"inverse": "calibrator_v1_2025-07-09T13_31_29+00_00",
		"forward": "calibrator_v1_2025-07-09T14_04_49+00_00"
	},
	{ # Flip 7 + long run
		"forward": "calibrator_v1_2025-07-09T14_04_49+00_00",
		"inverse": "calibrator_v1_2025-07-09T14_36_15+00_00"
	},
	{ # Flip 8 + long run
		"inverse": "calibrator_v1_2025-07-09T14_36_15+00_00",
		"forward": "calibrator_v1_2025-07-09T15_04_59+00_00"
	},
	{ # Flip 9 + long run
		"forward": "calibrator_v1_2025-07-09T15_04_59+00_00",
		"inverse": "calibrator_v1_2025-07-09T15_37_30+00_00"
	}
]

# This function takes captures and model signals
# Also bins
# Returns values for two channels
def extract_amplitude_response(captures, signals, bins):
	acc_a = np.zeros_like(bins)
	acc_b = np.zeros_like(bins)
	cnt_a = np.zeros_like(bins)
	cnt_b = np.zeros_like(bins)
	n_bins = len(bins)

	trim = 0.05

	for capture in captures:
		filter_fn = lambda x: parse_freq_expr(x.descriptor.tune) == capture.center_freq
		signal = next(filter(filter_fn, signals))

		tune = parse_freq_expr(signal.descriptor.tune)

		# Pulse cropping
		start = signal.duration*trim
		stop = signal.duration*(1-trim)

		indices = (signal.time >= start) * (signal.time < stop)

		x = signal.temporal_freq[indices] + tune
		y = signal.eliminate_delay(capture.iq)[indices]
		y = np.abs(y)

		# Some summation
		# Bad approarch from a numerical standpoint? accumulators get very large values
		ind = np.digitize(x, bins)
		acc = np.bincount(ind, y, n_bins)
		cnt = np.bincount(ind, None, n_bins)

		if capture.channel_number == 1:
			acc_a += acc
			cnt_a += cnt
		elif capture.channel_number == 3:
			acc_b += acc
			cnt_b += cnt
		else:
			assert 0

	# Suppress NaNs in empty bins
	cnt_a[cnt_a == 0] = 1
	cnt_b[cnt_b == 0] = 1

	acc_a /= cnt_a
	acc_b /= cnt_b

	return acc_a, acc_b

def run_v1():
	with open(capdir + sessions[0]["forward"] + "/preset.json") as f:
		obj = json.load(f)

	preset = JsonDDCAndCalibratorV1.deserialize(obj["ddc-and-calibrator-v1"])

	prop_cycle = plt.rcParams['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']

	plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
	plt.rcParams['axes.xmargin'] = 0
	plt.rcParams['axes.ymargin'] = 0

	signals = []

	for descriptor in preset.signals:
		signals.append( ModelSignalV1(descriptor, preset.ddc) )

	bins = np.linspace(153.5*1000*1000, 162.5*1000*1000, 1024)

	ch_a_stream = []
	ch_b_stream = []

	# Captures in time window
	class window:
		captures = []
		since = None
		until = None

	def flush():
		window.since += timedelta(seconds=10)
		window.until += timedelta(seconds=10)
		print(window.since, len(window.captures))

		a, b = extract_amplitude_response(window.captures, signals, bins)
		ch_a_stream.append(a)
		ch_b_stream.append(b)
		window.captures = []

	try:
		# We must flush every 10 seconds
		# For a given a capture, we flush until it is in-window
		for filename in sorted(glob(capdir + "/*/*.ISE")):
			with open(filename, "rb") as f:
				for capture in StreamORDA(f).captures:
					if capture.center_freq == 0: # DDC quirk
						continue

					if not window.until:
						window.since = capture.timestamp
						window.until = capture.timestamp + timedelta(seconds=10)

					while window.until < capture.timestamp:
						flush()

					window.captures.append(capture)
	except KeyboardInterrupt as e:
		pass

	plt.imsave( "channelA.png", np.transpose(ch_a_stream) )
	plt.imsave( "channelB.png", np.transpose(ch_b_stream) )

	fig, (ax1, ax2) = plt.subplots(2)

	ax1.imshow( np.transpose(ch_a_stream) )
	ax2.imshow( np.transpose(ch_b_stream) )

	plt.show()
