
import numpy as np

from src.delay import SpectralDelayEstimator
from src.misc import ad9910_sweep_bandwidth, parse_freq_expr, parse_time_expr
import src.dds as dds

#
# We have domain models:
# - JsonDDCAndCalibratorV1
# 	- JsonDDCSettingsV1
# 	- JsonSignalV1
#
# They might as well not exist
# Are for storage representation and have no methods
# Their hierarchy is unsuitable for application tasks
#
# Then we have actual application classes:
# - SignalV1	A self-contained DDC capture model
#
# The class hierarchy becomes inverted to meet
# the application demands
#
# Who makes SignalV1?
#

class ModelSignalV1:
	"""
	Joint DDC + Calibrator signal modelling application class

	Self-contained and context-independent
	"""

	# Models have associated DDC settings and the signal descriptor
	descriptor = None
	ddc = None

	# Lack of context regarding other signals can cause duplicate work-
	# most signals are literally the same
	est = None
	time = None
	delay = None
	duration = None
	temporal_freq = None
	spectral_freq = None

	def __init__(self, descriptor, ddc, trim=0.05):
		"""
		Prepares a model signal as DDC would see it

		trim	Portion of pulse head+tail to be discarded so as to remove transients
			Default: 0.05
		"""

		self.ddc = ddc
		self.descriptor = descriptor

		sysclk = parse_freq_expr("1 GHz")

		rate = parse_freq_expr(ddc.samplerate)
		frames = ddc.frames

		capture_duration = frames / rate

		tokens = descriptor.emit.split(" ")

		if tokens[0] == "sweep":
			sweep, offset, offset_unit, duration, duration_unit, freq, freq_unit, a, b = tokens

			delay = parse_time_expr(f"{offset} {offset_unit}")
			duration = parse_time_expr(f"{duration} {duration_unit}")
			center_frequency = parse_freq_expr(f"{freq} {freq_unit}")
			a = int(a)
			b = int(b)

			#
			# Prepare model signal
			# TODO: support off-center tuning
			#
			time = dds.time_series(rate, capture_duration)
			band = ad9910_sweep_bandwidth(a, b, sysclk=sysclk)

			model_sweep = dds.sweep(time, -band/2, band/2, 0, duration)
			temporal_freq = (time / duration)*band - band/2
			spectral_freq = np.linspace(-rate/2, rate/2, frames)

			#
			# Prepare delay estimator
			#
			# We could have some kind of heuristic to decide on the estimator,
			# for example having at least 100 bins occupied to pick SpectralDelayEstimator:
			# bins_occupied = int( torch.unique(freqs - freqs % (rate/frames)).shape[0] )
			#
			# But lets have it simpler:
			# - Sweeps have SpectralDelayEstimator - which assumes contiguous indices_est
			# - Pulses have ConvDelayEstimator

			# Establish the frequencies at truncated head/tail
			start = duration*trim + delay
			stop = duration*(1-trim) + delay

			temporal_indices = (time >= start) * (time < stop)

			min_freq = temporal_freq[temporal_indices].min()
			max_freq = temporal_freq[temporal_indices].max()

			indices_est = (spectral_freq >= min_freq)*(spectral_freq < max_freq)
			est = SpectralDelayEstimator(model_sweep, indices_est)

			self.est = est
			self.time = time
			self.delay = delay
			self.duration = duration
			self.temporal_freq = temporal_freq
			self.spectral_freq = spectral_freq
		else:
			assert 0

	def eliminate_delay(self, iq):
		"""
		Remove time delay from a capture

		TODO: shoud this actually be "fit()"?
		"""
		sample_delay = self.est.estimate(iq)

		# Use simple roll for now
		return np.roll( iq, round(-sample_delay.item()) )
