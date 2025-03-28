import numpy as np

from src.orda import StreamORDA
from src.display import page, minmaxplot
import src.delay as delay
import src.dds as dds

def run_v1():
	"""
	DDC integral nonlinearity experiment

	lo	10 MHz, 270 mV to 10 mV
	hi	190 MHz, 350 mV to 10 mV
	"""

	fname_lo = "c:/Data/cal/20250328_054933_000_0000_003_000.ISE"
	fname_hi = "c:/Data/cal/20250328_055300_000_0000_003_000.ISE"

	lo_mv = [270 - i*10 for i in range(27)]
	hi_mv = [350 - i*10 for i in range(35)]

	#
	# Load all captures
	#
	with open(fname_lo, "rb") as f:
		lo_captures = StreamORDA(f).all_captures()

	with open(fname_hi, "rb") as f:
		hi_captures = StreamORDA(f).all_captures()


	#
	# Exclude 0 Hz, leave ch1
	#
	lo_captures = filter(lambda x: x.center_freq != 0 and x.channel_number == 1, lo_captures)
	hi_captures = filter(lambda x: x.center_freq != 0 and x.channel_number == 1, hi_captures)

	lo_codes = []
	lo_codes_min = []
	lo_codes_max = []

	hi_codes = []
	hi_codes_min = []
	hi_codes_max = []

	print("10 MHz:")

	for x, mv in zip(lo_captures, lo_mv):
		amplitude = np.abs(x.iq)
		code = amplitude.mean()
		noise = amplitude.std()
		lo_codes.append( amplitude.mean() )
		lo_codes_min.append( amplitude.min() )
		lo_codes_max.append( amplitude.max() )
		print(mv, code, f"\t3 sigma = {3*noise:.3f}")

	print("190 MHz:")

	for x, mv in zip(hi_captures, hi_mv):
		amplitude = np.abs(x.iq)
		code = amplitude.mean()
		noise = amplitude.std()
		hi_codes.append( amplitude.mean() )
		hi_codes_min.append( amplitude.min() )
		hi_codes_max.append( amplitude.max() )
		print(mv, code, f"\t3 sigma = {3*noise:.3f}")

	lo_cost = (lo_codes[0] - lo_codes[-1]) / 260
	hi_cost = (hi_codes[0] - hi_codes[-1]) / 340

	print("lo cost per mv", lo_cost)
	print("hi cost per mv", hi_cost)

	def r(x):
		z = [] + x
		z.reverse()
		return z

	#
	# The INL plot
	#
	import matplotlib.pyplot as plt

	plt.title("DDC INL")
	plt.plot(lo_mv, np.array(lo_mv) - np.array(lo_codes)/lo_cost, marker="o", label="10 MHz")
	plt.plot(hi_mv, np.array(hi_mv) - np.array(hi_codes)/hi_cost, marker="o", label="190 MHz")
	plt.fill(lo_mv + r(lo_mv), np.array(lo_mv + r(lo_mv)) - np.array(lo_codes_max + r(lo_codes_min))/lo_cost, alpha=0.5)
	plt.fill(hi_mv + r(hi_mv), np.array(hi_mv + r(hi_mv)) - np.array(hi_codes_max + r(hi_codes_min))/hi_cost, alpha=0.5)
	plt.xlabel("mV in")
	plt.ylabel("mV error (absolute)")
	plt.legend()
	plt.show()


	plt.title("DDC INL")
	plt.plot(lo_mv, np.array(lo_mv) / (np.array(lo_codes)/lo_cost), marker="o", label="10 MHz")
	plt.plot(hi_mv, np.array(hi_mv) / (np.array(hi_codes)/hi_cost), marker="o", label="190 MHz")
	plt.fill(lo_mv + r(lo_mv), np.array(lo_mv + r(lo_mv)) / (np.array(lo_codes_max + r(lo_codes_min))/lo_cost), alpha=0.5)
	plt.fill(hi_mv + r(hi_mv), np.array(hi_mv + r(hi_mv)) / (np.array(hi_codes_max + r(hi_codes_min))/hi_cost), alpha=0.5)
	plt.xlabel("mV in")
	plt.ylabel("mV error (ratio real/anticipated)")
	plt.legend()
	plt.show()
