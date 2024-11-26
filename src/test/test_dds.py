from src import dds

def test_sine_energy_1():
	samplerate = 5*1000*1000
	samples = 8192
	duration = samples/samplerate
	pulse_duration = 900/1000/1000
	frequency = 1*1000*1000

	x = dds.time_series(samplerate, duration)
	y = dds.sine(x, frequency, duration=pulse_duration, offset=1/samplerate)

	assert y.abs().sum() == samplerate*pulse_duration

def test_sine_energy_2():
	samplerate = 5*1000*1000
	samples = 8192
	duration = samples/samplerate
	pulse_duration = 900/1000/1000
	frequency = 1*1000*1000

	x = dds.time_series(samplerate, duration)
	y = dds.sine(x, frequency, duration=pulse_duration, offset=.25/samplerate)

	assert y.abs().sum() == samplerate*pulse_duration

def test_sweep_energy_1():
	samplerate = 5*1000*1000
	samples = 8192
	duration = samples/samplerate
	pulse_duration = 900/1000/1000
	frequency = 1*1000*1000

	x = dds.time_series(samplerate, duration)
	y = dds.sweep(x, -frequency/2, +frequency/2, duration=pulse_duration, offset=1/samplerate)

	assert y.abs().sum() == samplerate*pulse_duration

def test_sweep_energy_2():
	samplerate = 5*1000*1000
	samples = 8192
	duration = samples/samplerate
	pulse_duration = 900/1000/1000
	frequency = 1*1000*1000

	x = dds.time_series(samplerate, duration)
	y = dds.sweep(x, -frequency/2, +frequency/2, duration=pulse_duration, offset=.25/samplerate)

	assert y.abs().sum() == samplerate*pulse_duration
