
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
# CH1-CH3
def extract_phase_delta(captures, signals):
	idx_a = 1
	idx_b = 3
	trim = 0.05

	coarse_delta_x = []
	coarse_delta_y = []

	for i, signal in enumerate(signals):
		tune = parse_freq_expr(signal.descriptor.tune)

		# Pulse cropping
		start = signal.duration*trim
		stop = signal.duration*(1-trim)

		# We want 1 vs 3
		u_filter_fn = lambda x: x.center_freq == tune and x.channel_number == idx_a
		v_filter_fn = lambda x: x.center_freq == tune and x.channel_number == idx_b

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

		coarse_delta_x.append(tune)
		coarse_delta_y.append(coarse)


	coarse_delta_x = np.hstack(coarse_delta_x)
	coarse_delta_y = np.hstack(coarse_delta_y)

	return coarse_delta_x, coarse_delta_y

def run_v1():
	with open(capdir + sessions[0]["forward"] + "/preset.json") as f:
		obj = json.load(f)

	preset = JsonDDCAndCalibratorV1.deserialize(obj["ddc-and-calibrator-v1"])

	signals = []

	prop_cycle = plt.rcParams['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']

	plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
	plt.rcParams['axes.xmargin'] = 0
	plt.rcParams['axes.ymargin'] = 0

	fig, (ax1, ax2) = plt.subplots(2)

	for descriptor in preset.signals:
		signals.append( ModelSignalV1(descriptor, preset.ddc) )

	for session in sessions:
		fwd_captures = []
		inv_captures = []

		for filename in sorted(glob(capdir + session["forward"] + "/*.ISE")):
			with open(filename, "rb") as f:
				for capture in StreamORDA(f).captures:
					fwd_captures.append(capture)

				print(filename)

		for filename in sorted(glob(capdir + session["inverse"] + "/*.ISE")):
			with open(filename, "rb") as f:
				for capture in StreamORDA(f).captures:
					inv_captures.append(capture)

				print(filename)

		if session["forward"] < session["inverse"]:
			print("Forward first")
			since = fwd_captures[-1].timestamp - timedelta(minutes=5)
			until = inv_captures[0].timestamp + timedelta(minutes=5)
		else:
			since = inv_captures[-1].timestamp - timedelta(minutes=5)
			until = fwd_captures[0].timestamp + timedelta(minutes=5)
			print("Inverse first")

		print(since)
		print(until)

		filter_fn = lambda x: (x.timestamp >= since) and (x.timestamp < until)

		fwd_captures = list(filter(filter_fn, fwd_captures))
		inv_captures = list(filter(filter_fn, inv_captures))

		print(len(fwd_captures), "forward captures")
		print(len(inv_captures), "inverse captures")

		x_fwd, y_fwd = extract_phase_delta(fwd_captures, signals)
		x_inv, y_inv = extract_phase_delta(inv_captures, signals)

		rffe_only = 0.5 * (y_fwd + y_inv)
		feed_only = 0.5 * (y_fwd - y_inv)

		timespan = since.strftime("%H:%M") + "..." + until.strftime("%H:%M")

		ax1.plot(x_fwd, rffe_only * (180/np.pi), label=timespan, marker="o")
		ax2.plot(x_fwd, feed_only * (180/np.pi), marker="o")

	fig.suptitle("Периодическая оценка разности фаз между каналами приемного тракта,\nдата и время: 2025-07-09, от 07:23 до 15:39 UT; канал A − канал B")
	ax1.set_title("Разность фаз, обеспеченная несимметричностью каналов приемного тракта")
	ax2.set_title("Разность фаз, обеспеченная особенностями кабелей")

	#ax1.legend()

	ax1.set_xlabel("Частота (МГц)")
	ax2.set_xlabel("Частота (МГц)")
	ax1.set_ylabel("Разность фаз (°)")
	ax2.set_ylabel("Разность фаз (°)")

	ticks_loc = x_fwd
	ticks_txt = x_fwd/1000/1000

	ticks_loc = ticks_loc[::4]
	ticks_txt = ticks_txt[::4]

	ax1.set_xticks(ticks_loc, ticks_txt)
	ax2.set_xticks(ticks_loc, ticks_txt)

	ax1.grid(True)
	ax2.grid(True)

	fig.legend(loc="right")

	plt.show()
