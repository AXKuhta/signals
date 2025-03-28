import numpy as np

from src.orda import StreamORDA
from src.display import page, minmaxplot
from src.misc import ddc_cost_mv
import src.delay as delay
import src.dds as dds

def run_v1():
	"""
	DDC cost model test

	fname	190 MHz, 350 mV to 10 mV
	"""

	fname = "c:/Data/cal/20250328_055300_000_0000_003_000.ISE"

	mv = [350 - i*10 for i in range(35)]

	with open(fname, "rb") as f:
		captures = StreamORDA(f).all_captures()

	#
	# Exclude 0 Hz, leave ch1
	#
	captures = filter(lambda x: x.center_freq != 0 and x.channel_number == 1, captures)

	codes = []
	codes_min = []
	codes_max = []

	for x, _ in zip(captures, mv):
		amplitude = np.abs(x.iq)
		codes.append( amplitude.mean() )
		codes_min.append( amplitude.min() )
		codes_max.append( amplitude.max() )

	def r(x):
		z = [] + x
		z.reverse()
		return z

	#
	# The INL plot
	#
	import matplotlib.pyplot as plt

	plt.title("DDC voltage scale model test")
	plt.plot(mv, np.array(mv) - np.array(codes)*ddc_cost_mv(190*1000*1000), marker="o", label="190 MHz")
	plt.fill(mv + r(mv), np.array(mv + r(mv)) - np.array(codes_max + r(codes_min))*ddc_cost_mv(190*1000*1000), alpha=0.5)
	plt.xlabel("mV in")
	plt.ylabel("mV error (absolute)")
	plt.legend()
	plt.show()
