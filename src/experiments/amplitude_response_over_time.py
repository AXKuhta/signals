
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

	z = np.arange(8192)

	from functools import cache

	@cache
	def get_ind(signal):
		start = signal.duration*trim
		stop = signal.duration*(1-trim)
		indices = (signal.time >= start) * (signal.time < stop)
		x = signal.temporal_freq[indices] + parse_freq_expr(signal.descriptor.tune)
		return np.digitize(x, bins) # Expensive, so avoid doing it over and over again

	for capture in captures:
		filter_fn = lambda x: parse_freq_expr(x.descriptor.tune) == capture.center_freq
		signal = next(filter(filter_fn, signals))

		tune = parse_freq_expr(signal.descriptor.tune)

		# Pulse cropping
		start = signal.duration*trim
		stop = signal.duration*(1-trim)

		indices = (signal.time >= start) * (signal.time < stop)

		#delay = 2541 #signal.est.estimate(capture.iq)
		#d2 = np.int32(np.round(delay))
		d2 = 2541

		#if delay<2000 or delay>2700:
		#	continue

		#print(delay)

		x = signal.temporal_freq[indices] + tune
		y = capture.iq[z[indices] + d2]
		y = np.abs(y)

		# Some summation
		# Bad approarch from a numerical standpoint? accumulators get very large values
		#ind = np.digitize(x, bins)
		ind = get_ind(signal)
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
	timestamps = []

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
		timestamps.append(window.since)
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

	fig, (ax1, ax2) = plt.subplots(2, figsize=[16, 10])

	im1 = ax1.imshow( np.transpose(ch_a_stream) )
	im2 = ax2.imshow( np.transpose(ch_b_stream) )

	mhz = bins/1000/1000

	ytick_px = np.searchsorted(mhz, np.unique(np.round(mhz)))
	ytick_hz = [f"{x:.1f}" for x in mhz[ytick_px]]

	sl = slice(None, None, 200)

	xtick_px = np.arange( len(timestamps) )[sl]
	xtick_ts = [x.strftime("%H:%M") for x in timestamps[sl] ]

	ax1.set_yticks(ytick_px, ytick_hz)
	ax2.set_yticks(ytick_px, ytick_hz)

	ax1.set_xticks(xtick_px, xtick_ts)
	ax2.set_xticks(xtick_px, xtick_ts)

	#ax1.set_xlabel("Время UT")
	#ax2.set_xlabel("Время UT")
	ax1.set_ylabel("Частота (МГц)")
	ax2.set_ylabel("Частота (МГц)")

	plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")
	plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")

	ax1.grid(True, c="black", alpha=0.5, linewidth=1)
	ax2.grid(True, c="black", alpha=0.5, linewidth=1)

	fig.suptitle("Периодическая оценка АЧХ приемного тракта с чередованием входов ВУПа\nдата и время: 2025-07-09, от 07:23 до 15:39 UT;")
	ax1.set_title("Канал A")
	ax2.set_title("Канал B")

	cbar1 = fig.colorbar(im1, ax=ax1)
	cbar2 = fig.colorbar(im2, ax=ax2)
	cbar1.ax.set_ylabel("Код АЦП")
	cbar2.ax.set_ylabel("Код АЦП")

	plt.show()
	fig.savefig("result.png", dpi=300)
