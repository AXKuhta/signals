from math import floor, ceil
import torch

#
# AD9910 sweep calculations
#
def ad9910_sweep_bandwidth(a, b, duration=900/1000/1000, sysclk=1000*1000*1000):
	fstep = sysclk / 2**32
	steps = sysclk / 4 / b * duration

	assert steps % 1 == 0

	return a * fstep * (steps - 1)

#
# Roll with lerp
# roll_lerp(torch.tensor([1,0,0,0]), 1) = tensor([0, 1, 0, 0])
# roll_lerp(torch.tensor([1,0,0,0]), 0.5) = tensor([0.5000, 0.5000, 0.0000, 0.0000])
#
def roll_lerp(x, shift):
	shift = float(shift)
	a = floor(shift)
	b = ceil(shift)
	u = x.roll(a)
	v = x.roll(b)
	w = shift - a

	return torch.lerp(
		u.double(),
		v.double(),
		w
	)
