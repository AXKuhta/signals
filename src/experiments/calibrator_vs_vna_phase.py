
import torch

from src.orda import StreamORDA
from src.touchstone import S2PFile
from src.display import page, minmaxplot
import src.delay as delay
import src.dds as dds

#
# AD9910 sweep calculations
#
def ad9910_sweep_bandwidth(a, b, duration=900/1000/1000, sysclk=1000*1000*1000):
	fstep = sysclk / 2**32
	steps = sysclk / 4 / b * duration

	assert steps % 1 == 0

	return a * fstep * (steps - 1)

#
# Testing a combined filter+amplifier
# consisting of
# - a SXBP-157+
# - a PHA-13HLN+
#
# https://www.minicircuits.com/pdfs/SXBP-157+.pdf
# https://www.minicircuits.com/pdfs/PHA-13HLN+.pdf
#
#
def run_v1():
	# Reference signal
	# AD9910 -> SMA-SMA -> SMA Tee -> SMA-LEMO -> DDC CHANNEL 3
	#                              -> SMA-BNC -> BNC-SMA -> Coupler -> SMA-LEMO -> DDC CHANNEL 1
	#
	fname_ddc = "cal_2024_11_13/20241113_062312_000_0000_003_000.ISE"

	# Feeding signal through channel A box
	# AD9910 -> SMA-SMA -> SMA Tee -> SMA-LEMO -> DDC CHANNEL 3
	#                              -> SMA-BNC -> BNC-SMA -> CHANNEL A AMP -> SMA-LEMO -> DDC CHANNEL 1
	#
	fname_box_a = "cal_2024_11_13/20241113_062936_000_0000_003_000.ISE"

	# Feeding signal through channel B box
	# AD9910 -> SMA-SMA -> SMA Tee -> SMA-LEMO -> DDC CHANNEL 3
	#                              -> SMA-BNC -> BNC-SMA -> CHANNEL B AMP -> SMA-LEMO -> DDC CHANNEL 1
	#
	fname_box_b = "cal_2024_11_13/20241113_063644_000_0000_003_000.ISE"

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

	time = dds.time_series(rate, duration) - 20/1000/1000
	band = ad9910_sweep_bandwidth(77, 1)
	pulse_duration = 900/1000/1000

	model_sweep = dds.sweep(time, -band/2, band/2, 0, pulse_duration, clip=False)

	temporal_freq = (time / pulse_duration)*band - band/2
	spectral_freq = torch.linspace(-rate/2, rate/2, frames)

	#
	# Snip full pulse duration (a bit over 4 MHz)
	# Or snip exactly 4 MHz
	#
	indices = time <= pulse_duration
	indices = (temporal_freq >= -2*1000*1000)*(temporal_freq < 2*1000*1000)

	spectral = minmaxplot("Hz")
	spectral.ytitle("Радианы")
	spectral.xtitle("Частота")
	spectral.footer("Сравнение калибратора и VNA при измерении ФЧХ одного из усилителей в ВУПе<br>Использованы два канала DDC - один для опорного сигнала, второй для сигнала после усилителя<br>Провода учтены")


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
		inpt_filter_fn = lambda x: x.center_freq == freq and x.channel_number == 3
		thru_filter_fn = lambda x: x.center_freq == freq and x.channel_number == 1

		ref_inpt = filter(inpt_filter_fn, captures_ref)
		ref_thru = filter(thru_filter_fn, captures_ref)
		dut_inpt = filter(inpt_filter_fn, captures_dut)
		dut_thru = filter(thru_filter_fn, captures_dut)

		a = [ x.iq for x in ref_inpt ]
		b = [ x.iq for x in ref_thru ]
		delta_phi_ref = torch.vstack( [ v*u.conj() for u, v in zip(a, b) ] ).mean(0)

		a = [ x.iq for x in dut_inpt ]
		b = [ x.iq for x in dut_thru ]
		delta_phi_dut = torch.vstack( [ v*u.conj() for u, v in zip(a, b) ] ).mean(0)

		phase = (delta_phi_dut * delta_phi_ref.conj()).angle()

		x = temporal_freq[indices] + freq
		y = phase[indices]

		lst_x.append(x)
		lst_y.append(y)


	x = torch.hstack(lst_x)
	y = torch.hstack(lst_y)
	#lower = torch.hstack(lst_lower)
	#upper = torch.hstack(lst_upper)

	spectral.trace(x, y, name="Calibrator")

	fname_vna_box_a = "VNA/channel_a_1000pts.s2p"
	fname_vna_box_b = "VNA/channel_b_1000pts.s2p"

	with open(fname_vna_box_b, "rb") as f:
		vna = S2PFile(f)

	spectral.trace(vna.freqs, vna.s21.angle(), name="VNA")

	disp = page([spectral])
	disp.show()
	input()
