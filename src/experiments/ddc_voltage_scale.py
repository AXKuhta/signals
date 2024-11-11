
import torch

from src.orda import StreamORDA
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

def run_v1():
	"""
	Testing the DDC accross a wide range of frequencies and signal levels

	cal_2024_10_23/20241023_052834_000_0000_003_000.ISE	4 MHz sweeps, ASF=16337 FSC=102 (200 mV)
	cal_2024_10_29/20241029_062948_000_0000_003_000.ISE	Tones from a signal generator (200 mV)
	"""

	#
	# Load all captures
	#
	with open("cal_2024_10_23/20241023_052834_000_0000_003_000.ISE", "rb") as f:
		sweeps = StreamORDA(f).all_captures()

	with open("cal_2024_10_29/20241029_062948_000_0000_003_000.ISE", "rb") as f:
		tones = StreamORDA(f).all_captures()

	#
	# We have a number of frequencies in the file
	#
	freqs = set( [x.center_freq for x in sweeps] )

	#
	# Prepare model signal
	#
	frames = sweeps[0].samplecount
	rate = sweeps[0].samplerate

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

	indices = time <= pulse_duration
	indices = (temporal_freq > -100) * (temporal_freq < 100) # Take one point

	spectral = minmaxplot("Hz")
	spectral.ytitle("Код АЦП")
	spectral.xtitle("Частота")
	spectral.footer("Примечание: генератор сигналов создавал непрерывный сигнал, калибратор создавал ЛЧМ импульсы")

	#
	# 0 Hz must be skipped
	#
	freq_set = freqs - {0}

	tone_x = []
	tone_y = []

	#
	# Do tones
	#
	for freq in sorted(freq_set):
		filter_fn = lambda x: x.center_freq == freq and x.channel_number == 1
		caps = filter(filter_fn, tones)

		a = [x.iq.abs().mean() for x in caps]
		b = torch.vstack(a)

		mampl = b.mean(0)
		lower, _ = b.min(0)
		upper, _ = b.max(0)

		tone_x.append(freq)
		tone_y.append(b[0][0])

	spectral.trace(tone_x, tone_y, name=f"Signal generator")

	x = []
	y = []

	#
	# Do sweeps
	#
	for freq in sorted(freq_set):
		filter_fn = lambda x: x.center_freq == freq and x.channel_number == 1
		caps = filter(filter_fn, sweeps)

		a = [eliminate_time_delay(x.iq) for x in caps]
		b = torch.vstack([x.abs() for x in a])

		mampl = b.mean(0)
		lower, _ = b.min(0)
		upper, _ = b.max(0)

		x.append( temporal_freq[indices] + freq )
		y.append( mampl[indices] )

		#spectral.trace(x, y, error_band=(lower[indices], upper[indices]), name=f"{freq/1000/1000}MHz")

	x = torch.hstack(x)
	y = torch.hstack(y)
	spectral.trace(x, y, name="Calibrator")

	disp = page([spectral])
	disp.show()
	input()
