import numpy as np

from src.orda import StreamORDA
from src.display import page, minmaxplot
import src.delay as delay
import src.dds as dds

def run_v1():
	"""
	AD9910's output level modelling

	a	250 mV (Trusted), 10 MHz to 390 MHz with 1 MHz step, from G4-218/1
	b	250 mV (Assumed), 10 MHz to 390 MHz with 1 MHz step, from AD9910
	"""

	fname_a = "c:/Data/cal/20250402_060325_000_0000_003_000.ISE"
	#fname_b = "c:/Data/cal/20250402_052804_000_0000_003_000.ISE" # AD9910 normal
	#fname_b = "c:/Data/cal/20250402_072437_000_0000_003_000.ISE" # AD9910 lowpass removed
	#fname_b = "c:/Data/cal/20250402_075219_000_0000_003_000.ISE" # AD9910 lowpass restored
	fname_b = "c:/Data/cal/20250402_080854_000_0000_003_000.ISE"  # Longer cable

	#
	# Load all captures
	#
	with open(fname_a, "rb") as f:
		a_captures = StreamORDA(f).all_captures()

	with open(fname_b, "rb") as f:
		b_captures = StreamORDA(f).all_captures()


	#
	# Exclude 0 Hz and frequencies in proximity of multiples of Fs/2, leave ch1
	#
	banned = [
		0,
		99*1000*1000,
		100*1000*1000,
		101*1000*1000,
		199*1000*1000,
		200*1000*1000,
		201*1000*1000,
		299*1000*1000,
		300*1000*1000,
		301*1000*1000
	]

	a_captures = filter(lambda x: x.center_freq not in banned and x.channel_number == 1, a_captures)
	b_captures = filter(lambda x: x.center_freq not in banned and x.channel_number == 1, b_captures)

	x = []
	y = []
	min = []
	max = []

	# Compensate for sinc rolloff
	sysclk = 1000*1000*1000

	def inv_sinc(x):
		x = x / 1000 / 1000 / 1000
		return np.pi * x / np.sin(np.pi * x)

	# We assume b-records have lower amplitude than a-records
	for a, b in zip(a_captures, b_captures):
		ratio = inv_sinc(a.center_freq) * np.abs(b.iq) / np.abs(a.iq)

		x.append( a.center_freq )
		y.append( ratio.mean() )
		min.append( ratio.min() )
		max.append( ratio.max() )

	def r(x):
		z = [] + x
		z.reverse()
		return z

	#
	# Ratio plot
	#
	import matplotlib.pyplot as plt

	plt.title("Ratio")
	#plt.plot( np.array(x)/1000/1000, 10*np.log10(y), label="|b|/|a|")
	#plt.fill( np.array(x + r(x))/1000/1000, 10*np.log10(max + r(min)), alpha=0.5)
	plt.plot( np.array(x)/1000/1000, y, label="|b|/|a|")
	plt.fill( np.array(x + r(x))/1000/1000, max + r(min), alpha=0.5)
	plt.xlabel("MHz")
	plt.ylabel("Value")
	plt.legend()
	plt.show()
