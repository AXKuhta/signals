from math import floor, ceil
import numpy as np

#
# AD9910 sweep calculations
#
def ad9910_sweep_bandwidth(a, b, duration=900/1000/1000, sysclk=1000*1000*1000):
	fstep = sysclk / 2**32
	steps = sysclk / 4 / b * duration

	assert steps % 1 == 0

	return a * fstep * (steps - 1)

def lerp(u, v, w):
	return (1 - w) * u + v * w

#
# Roll with lerp
# roll_lerp(torch.tensor([1,0,0,0]), 1) = tensor([0, 1, 0, 0])
# roll_lerp(torch.tensor([1,0,0,0]), 0.5) = tensor([0.5000, 0.5000, 0.0000, 0.0000])
#
def roll_lerp(x, shift):
	shift = float(shift)
	a = floor(shift)
	b = ceil(shift)
	u = np.roll(x, a)
	v = np.roll(x, b)
	w = shift - a

	return lerp(u, v, w)

#
# Parses expressions like 1000*1000*1000 or 1/1000/1000
#
def parse_numeric_expr(expr):
	token = []
	tokens = []

	for c in expr:
		if c in "*/":
			tokens.append( "".join(token) )
			tokens.append(c)
			token = []
		else:
			token.append(c)

	tokens.append( "".join(token) )
	acc = float( tokens.pop(0) )

	while len(tokens):
		operator = tokens.pop(0)
		operand = float( tokens.pop(0) )

		if operator == "*":
			acc = acc * operand
		elif operator == "/":
			acc = acc / operand
		else:
			assert 0, f"unknown operator {operator}"

	return acc
