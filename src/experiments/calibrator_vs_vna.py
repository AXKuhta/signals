
from enum import Enum, auto
from glob import glob
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

def run_v2():
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

			# Onto the actual processing
			# Delay elimination + amplitude + averaging
			#################################################################################################
			for i, signal in enumerate(signals):
				tune = parse_freq_expr(signal.descriptor.tune)

				# Pulse cropping
				start = signal.duration*trim + signal.delay
				stop = signal.duration*(1-trim) + signal.delay

				indices = (signal.time >= start) * (signal.time < stop)

				# Model captures
				x = signal.temporal_freq[indices] + tune
				y = np.abs(signal.iq)[indices]

				model_x.append(x)
				model_y.append(y)

				# Actual captures
				# Deal with:
				# - Different channels
				# - Accumulation
				for channel in chan_set:
					filter_fn = lambda x: x.trigger_number % len(signals) == i and x.channel_number == channel

					repeats = list(filter(filter_fn, captures))

					assert all([x.center_freq == tune for x in repeats])

					a = [signal.eliminate_delay(x.iq) for x in repeats]
					a = np.abs(np.vstack(a))

					mampl = a.mean(0)
					lower, _ = a.min(0), np.argmin(0)
					upper, _ = a.max(0), np.argmax(0)

					x = signal.temporal_freq[indices] + tune
					y = mampl[indices]

					lower = lower[indices]
					upper = upper[indices]

					adc_ch_x[channel].append(x)
					adc_ch_y[channel].append(y)

			# Second pass over the data to sort it - this deals with overlap
			#################################################################################################
			for chan in chan_set:
				x = np.hstack(adc_ch_x[chan])
				y = np.hstack(adc_ch_y[chan])

				x, indices = np.sort(x), np.argsort(x)
				y = y[indices]

				adc_ch_x[chan] = x
				adc_ch_y[chan] = y

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

			spectral = minmaxplot("Hz")
			spectral.xtitle("Частота")

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

	#a = FrequencyResponsePointsV1("/media/pop/32/calibrator_data_v1/calibrator_v1_2025-04-15T07_46_55+00_00") # Attenuator 5 dB
	#a = FrequencyResponsePointsV1("/media/pop/32/calibrator_data_v1/calibrator_v1_2025-04-15T07_56_09+00_00") # HPF
	a = FrequencyResponsePointsV1("/media/pop/32/calibrator_data_v1/calibrator_v1_2025-04-15T07_59_26+00_00") # LPF
	b = FrequencyResponsePointsV1("/media/pop/32/calibrator_data_v1/calibrator_v1_2025-04-15T07_44_00+00_00") # Thru

	spectral = minmaxplot("Hz")
	spectral.xtitle("Частота")
	spectral.ytitle("dB")
	spectral.yrange([-10, 0])

	# Stick to channel 1
	x = a.adc_ch_x[1]
	u = a.adc_ch_y[1]
	v = b.adc_ch_y[1]

	ratio = 20*np.log10( u / v )
	spectral.trace(x, ratio, name=f"Калибратор")

	fname_vna = "/media/pop/32/VNA/demoboard/lpf.s2p" # LPF
	#fname_vna = "/media/pop/32/VNA/demoboard/hpf.s2p" # HPF
	#fname_vna = "/media/pop/32/VNA/demoboard/minus_5db.s2p" # Attenuator 5 dB

	with open(fname_vna, "rb") as f:
		vna = S2PFile(f)

	vna_x = vna.freqs
	vna_y = 20*np.log10( np.abs(vna.s21) )

	spectral.trace(vna_x, vna_y, name="VNA")

	#err_y = np.interp(vna_x, x, y) - np.array(vna_y)
	#spectral.trace(vna_x, err_y, name="Calibrator error", hidden=True)

	result = page([spectral])
	result.show()

#
# Testing a combined filter+amplifier
# consisting of
# - a SXBP-157+
# - a PHA-13HLN+
#
# https://www.minicircuits.com/pdfs/SXBP-157+.pdf
# https://www.minicircuits.com/pdfs/PHA-13HLN+.pdf
#
def run_v1():
	# AD9910 PCBZ no R43 installed
	fname_ddc = "cal_2024_10_29/20241029_074522_000_0000_003_000.ISE" # Feeding signal directly into the DDC
	fname_box_b = "cal_2024_10_29/20241029_075143_000_0000_003_000.ISE" # Feeding through channel B box
	fname_box_a = "cal_2024_10_29/20241029_075859_000_0000_003_000.ISE" # Feeding through channel A box

	# AD9910 PCBZ with 100 ohms R43 installed
	# https://ez.analog.com/dds/w/documents/3680/ad9910-evb-r43-resistor-value
	# https://www.ti.com/lit/an/slaa399/slaa399.pdf
	fname_ddc = "cal_2024_11_12/20241112_074749_000_0000_003_000.ISE" # Feeding signal directly into the DDC
	fname_box_b = "cal_2024_11_12/20241112_075337_000_0000_003_000.ISE" # Feeding through channel B box
	fname_box_a = "cal_2024_11_12/20241112_075025_000_0000_003_000.ISE" # Feeding through channel A box

	#
	# Load all captures
	#
	with open(fname_ddc, "rb") as f:
		captures_ref = StreamORDA(f).all_captures()

	with open(fname_box_b, "rb") as f:
		captures_dut = StreamORDA(f).all_captures()

	#
	# We have a number of frequencies in the files
	#
	freqs_ref = set( [x.center_freq for x in captures_ref] )
	freqs_dut = set( [x.center_freq for x in captures_dut] )

	assert freqs_ref == freqs_dut

	#
	# Prepare model signal
	#
	frames = captures_ref[0].samplecount
	rate = captures_ref[0].samplerate

	duration = frames / rate

	time = dds.time_series(rate, duration)
	band = ad9910_sweep_bandwidth(77, 1)
	pulse_duration = 900/1000/1000

	model_sweep = dds.sweep(time, -band/2, band/2, 0, pulse_duration)
	spectrum_m = torch.fft.fft(model_sweep.flip(0).roll(1).conj())

	temporal_freq = (time / pulse_duration)*band - band/2
	spectral_freq = torch.linspace(-rate/2, rate/2, frames)

	def eliminate_time_delay(signal):
		spectrum_s = torch.fft.fft(signal)
		spectrum_c = spectrum_s * spectrum_m

		# Time delay estimation
		offset = 1 # [hack] set to 1 for noisy signals
		f_shift = torch.fft.fftshift( (spectrum_c.angle().diff() - offset) % -torch.pi + offset )*frames/(2*torch.pi)
		f_indices = torch.where( (spectral_freq >= -band/4) * (spectral_freq <= band/4) )
		sample_delay = -f_shift[f_indices].mean()

		# Time delay elimination
		spectrum_e = spectrum_s * torch.fft.fftshift( delay.delay_in_freq(sample_delay, frames) )
		signal = torch.fft.ifft(spectrum_e)

		return signal

	#
	# Snip full pulse duration (a bit over 4 MHz)
	# Or snip exactly 4 MHz
	#
	indices = time <= pulse_duration
	indices = (temporal_freq >= -2*1000*1000)*(temporal_freq < 2*1000*1000)

	spectral = minmaxplot("Hz")
	spectral.yrange([-50, 50])
	spectral.ytitle("dB")
	spectral.xtitle("Частота")
	spectral.footer("Сравнение калибратора и VNA при измерении АЧХ одного из усилителей ВУПа")

	#
	# 0 Hz must be skipped
	#
	freq_set = freqs_ref - set( [0] )

	lst_x = []
	lst_y = []
	lst_lower = []
	lst_upper = []

	#
	# Do one at a time
	#
	for freq in sorted(freq_set):
		filter_fn = lambda x: x.center_freq == freq and x.channel_number == 1
		ref = filter(filter_fn, captures_ref)
		dut = filter(filter_fn, captures_dut)

		a = [eliminate_time_delay(x.iq) for x in ref]
		b = [eliminate_time_delay(x.iq) for x in dut]
		c = torch.vstack( [ v.abs() / u.abs() for u, v in zip(a, b) ] )

		mampl = c.mean(0)
		lower, _ = c.min(0)
		upper, _ = c.max(0)

		mampl = 20*torch.log10(mampl)
		lower = 20*torch.log10(lower)
		upper = 20*torch.log10(upper)

		x = temporal_freq[indices] + freq
		y = mampl[indices]

		lower = lower[indices]
		upper = upper[indices]

		#spectral.trace(x, y, error_band=(lower, upper), name=f"{freq/1000/1000} MHz")

		lst_x.append(x)
		lst_y.append(y)
		lst_lower.append(lower)
		lst_upper.append(upper)

	x = torch.hstack(lst_x)
	y = torch.hstack(lst_y)
	lower = torch.hstack(lst_lower)
	upper = torch.hstack(lst_upper)

	#spectral.trace(x, y, error_band=(lower, upper), name="Attenuation")
	spectral.trace(x, y, name="Calibrator")

	fname_vna_box_a = "VNA/channel_a_1000pts.s2p"
	fname_vna_box_b = "VNA/channel_b_1000pts.s2p"

	with open(fname_vna_box_b, "rb") as f:
		vna = S2PFile(f)

	vna_x = vna.freqs
	vna_y = 20*torch.log10( vna.s21.abs() )

	spectral.trace(vna_x, vna_y, name="VNA")

	err_y = np.interp(vna_x, x, y) - np.array(vna_y)

	spectral.trace(vna_x, err_y, name="Calibrator error", hidden=True)

	disp = page([spectral])
	disp.show()
	input()
