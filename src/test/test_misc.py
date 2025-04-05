import numpy as np

from src import misc

def test_downsample():
	x = np.linspace(0, 100, 8192)

	x, l = misc.downsample(x, x, 5)

	assert x[0] == 0
	assert x[1] == 5

	assert np.all(l[0] < 2.5)
	assert np.all(l[1] >= 2.5) and np.all(l[1] < 7.5)
