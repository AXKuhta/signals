import torch

from .misc import roll_lerp

def time_series(samplerate, duration):
	"""
	Produces a time series, e.g.:
	time_series(4, 1) = tensor([0.0000, 0.2500, 0.5000, 0.7500])

	samplerate		SPS in Hz
	duration		length in seconds

	Double precision makes quite a difference
	"""

	return torch.arange(samplerate*duration, dtype=torch.double) / samplerate

def sine(time, freq, phase_offset=0.0, offset=0.0, duration=0.0):
	"""
	Produces a sine wave

	time			time series tensor
	freq			frequency in Hz
	phase_offset	starting phase in degrees
	offset			fill offset
	duration		fill time

	If duration is 0.0, then time[-1] is used as duration, filling the entire span of time
	"""

	real = torch.cos(freq*2*torch.pi*time + phase_offset/180.0*torch.pi)
	imag = torch.sin(freq*2*torch.pi*time + phase_offset/180.0*torch.pi)

	if duration:
		assert time[0] == 0, "Please supply time with 0 at origin"
		assert duration % time[1] < 1e-10, f"Please supply duration that is divisible by dt"
		mask = roll_lerp( (time<duration)*1, offset/time[1] )
		real *= mask
		imag *= mask

	return torch.complex(real, imag)

def sweep(time, f1, f2, offset=0.0, duration=0.0, clip=True):
	"""
	Produces a sweep

	time			time series tensor
	f1				start frequency
	f2				end frequency
	offset			fill offset
	duration		fill time

	If duration is 0.0, then time[-1] is used as duration, filling the entire span of time
	"""

	duration = duration if duration else time[-1]

	delta = f2 - f1
	x = time - offset

	base = f1*2*torch.pi*x
	swp = delta*torch.pi*x*x / duration

	real = torch.cos(base + swp)
	imag = torch.sin(base + swp)

	if clip:
		assert time[0] == 0, "Please supply time with 0 at origin"
		assert duration % time[1] < 1e-10, f"Please supply duration that is divisible by dt"
		mask = roll_lerp( (time<duration)*1, offset/time[1] )
		real *= mask
		imag *= mask

	return torch.complex(real, imag)

def psk(time, freq, code = [0, 0, 1, 0, 1]):
	"""
	Produces a PSK code

	Frequency for perfectly timed shifts:
	freq = samples / len(code) (Where samples = samplerate * duration)

	Can be halved or doubled safely.
	"""

	samples = time.shape[0]
	elements = len(code)

	assert samples % elements == 0

	element_duration = samples // elements
	code_mask = []

	for element in code:
	    code_mask += [element]*element_duration

	phase_offset = torch.tensor(code_mask) * torch.pi

	real = torch.cos(freq*2*torch.pi*time + phase_offset)
	imag = torch.sin(freq*2*torch.pi*time + phase_offset)

	return torch.complex(real, imag)

def rotator(phase_offset):
	"""
	Phase rotation helper

	phase offset	phase in degrees
	"""
	return torch.e**( (phase_offset/180.0*torch.pi) * 1j )
