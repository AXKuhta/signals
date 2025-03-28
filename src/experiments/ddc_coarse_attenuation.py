import numpy as np

from src.orda import StreamORDA
from src.display import page, minmaxplot
import src.delay as delay
import src.dds as dds

def run_v1():
	"""
	A coarse-grained test of the DDC's attenuation

	Referenced to 10 MHz at 270 mV

	Frequency table was generated using:

	```
	for i in range(190):
		print(f"157000 {10000+i*1000} 0 900 0 2150 21550")
	```

	"""

	fname = "c:/Data/cal/20250328_073446_000_0000_003_000.ISE"

	freqs = [ 10*1000*1000 + i*1000*1000 for i in range(181)]

	#
	# Load all captures
	#
	with open(fname, "rb") as f:
		captures = StreamORDA(f).all_captures()

	#
	# Exclude 0 Hz and frequencies near 100 MHz, leave ch1
	#
	banned = [0, 99*1000*1000, 100*1000*1000, 101*1000*1000]

	captures = filter(lambda x: x.center_freq not in banned and x.channel_number == 1, captures)
	freqs = list(filter(lambda x: x not in banned, freqs))

	codes = []
	codes_min = []
	codes_max = []

	for x, freq in zip(captures, freqs):
		amplitude = np.abs(x.iq)
		codes.append( amplitude.mean() )
		codes_min.append( amplitude.min() )
		codes_max.append( amplitude.max() )

	def r(x):
		z = [] + x
		z.reverse()
		return z

	import matplotlib.pyplot as plt

	plt.title("DDC amplitude frequency response")
	plt.plot(np.array(freqs)/1000/1000, np.array(codes), label="270 mV")
	plt.fill(np.array(freqs + r(freqs))/1000/1000, np.array(codes_max + r(codes_min)), alpha=0.5)
	plt.xlabel("Frequency")
	plt.ylabel("Code")
	plt.legend()
	plt.show()

	plt.title("DDC amplitude frequency response")
	plt.plot(np.array(freqs)/1000/1000, np.array(codes) * (1/codes[0]), label="270 mV")
	plt.xlabel("Frequency")
	plt.ylabel("Code")
	plt.legend()
	plt.show()

	import json

	print(json.dumps({
		"x": freqs,
		"y": list(np.array(codes) * (1/codes[0])),
	}))
