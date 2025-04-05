from math import floor, ceil
import numpy as np

def downsample(x, y, roundto=0.1):
	"""
	Sample rate conversion

	Rounds every x to a factor of `roundto` then groups `y` by it

	Behavior undefined if x is not monotonic (i.e. x is not sorted)
	"""

	# https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function
	repeat = np.round(x/roundto)*roundto
	unique, indices = np.unique(repeat, return_index=True)

	return unique, np.split(y, indices[1:])


#
# AD9910 sweep calculations
#
def ad9910_sweep_bandwidth(a, b, duration=900/1000/1000, sysclk=1000*1000*1000):
	fstep = sysclk / 2**32
	steps = sysclk / 4 / b * duration

	assert steps % 1 == 0

	return a * fstep * (steps - 1)

#
# AD9910 full scale current
#
def ad9910_fsc_i(fsc):
	return (86.4 / 10000) * (1 + (fsc/96.0))


# AD9910 is a current steering DAC
# Its peak output voltage is V = IR
#
# Where:
# R	the load in ohms
# I	output current
#
# While the current is:
# I = (asf/16383) * i(fsc)
#
# Where:
# fsc	full scale current code
# asf	amplitude scale factor code
#
# Peak voltage is then converted into rms
#
# To determine the DAC load see AD9910 PCBZ schematics
# and https://www.ti.com/lit/an/slaa399/slaa399.pdf
#
def ad9910_vrms_v0(asf, fsc):
	"""
	AD9910 output voltage estimate (no R43 installed)
	"""

	return ad9910_fsc_i(fsc) * (asf/16383) * (50/3) * 2**-.5

def ad9910_vrms_v1(asf, fsc):
	"""
	AD9910 output voltage estimate (R43 is 100 ohms)
	"""

	return ad9910_fsc_i(fsc) * (asf/16383) * (12.5) * 2**-.5

def ad9910_best_asf_fsc_v0(mv_rms):
	"""
	A pair of (asf, fsc) values that represent a given output voltage

	Assumes no R43 installed

	Tries to maximize asf

	Does not factor in sinc rolloff or lowpass filter losses
	"""

	for fsc in range(256):
		cost = ad9910_vrms_v0(1, fsc)
		asf = round(mv_rms / cost)

		if asf <= 16383:
			return asf, fsc

	assert 0

def ad9910_best_asf_fsc_v1(mv_rms):
	"""
	A pair of (asf, fsc) values that represent a given output voltage

	Assumes R43 is 100 ohms

	Tries to maximize asf

	Does not factor in sinc rolloff or lowpass filter losses
	"""

	for fsc in range(256):
		cost = ad9910_vrms_v1(1, fsc)
		asf = round(mv_rms / cost)

		if asf <= 16383:
			return asf, fsc

	assert 0

def ad9910_inv_sinc(x, sysclk=1000*1000*1000):
	"""
	AD9910 sinc rolloff compensation

	Takes:
	x	frequency array in Hz
	"""

	x = x / sysclk
	return np.pi * x / np.sin(np.pi * x)

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


def parse_angle_expr(expr, into="deg"):
	"""
	Parse an angle.
	"""

	value, unit = expr.split(" ")

	if "." in value:
		value = float(value)
	else:
		value = int(value)

	# FIXME: add radians.
	assert unit.lower() == "deg"

	return value

def parse_time_expr(expr, into="s"):
	"""
	Parse a quantity of time.

	Examples:
	>>> parse_time_expr("900 us", into="ns")
	900000.0
	"""

	inv_factors = {
		"s": 1,
		"ms": 1000,
		"us": 1000000,
		"ns": 1000000000
	}

	value, unit = expr.split(" ")

	if "." in value:
		value = float(value)
	else:
		value = int(value)

	inv_factor = inv_factors[ unit.lower() ] / inv_factors[ into.lower() ]

	return value / inv_factor

def parse_volt_expr(expr, into="v"):
	"""
	Parse a voltage.

	Examples:
	>>> parse_time_expr("60 mv", into="mv")
	60.0
	"""

	inv_factors = {
		"v": 1,
		"mv": 1000,
		"uv": 1000000
	}

	value, unit = expr.split(" ")

	if "." in value:
		value = float(value)
	else:
		value = int(value)

	inv_factor = inv_factors[ unit.lower() ] / inv_factors[ into.lower() ]

	return value / inv_factor

def parse_freq_expr(expr, into="hz"):
	"""
	Parse a frequency.

	Examples:

	>>> parse_freq_expr("150 MHz")
	150000000
	>>> parse_freq_expr("150.1 MHz")
	150100000.0

	Returns an int whenever possible.
	"""

	factors = {
		"hz": 1,
		"khz": 1000,
		"mhz": 1000000,
		"ghz": 1000000000
	}

	value, unit = expr.split(" ")

	if "." in value:
		value = float(value)
	else:
		value = int(value)

	factor = factors[ unit.lower() ] // factors[ into.lower() ]

	return value * factor

def ddc_cost_mv(freq):
	"""
	DDC voltage scale model

	Returns a factor that transforms codes into a voltage in mv

	Takes a scalar or an array
	"""

	# Temporarily disable
	#assert np.all(np.greater_equal(freq, 10*1000*1000))
	#assert np.all(np.less_equal(freq, 190*1000*1000))

	# Estimated attenuation at different frequencies
	x = [10000000, 11000000, 12000000, 13000000, 14000000, 15000000, 16000000, 17000000, 18000000, 19000000, 20000000, 21000000, 22000000, 23000000, 24000000, 25000000, 26000000, 27000000, 28000000, 29000000, 30000000, 31000000, 32000000, 33000000, 34000000, 35000000, 36000000, 37000000, 38000000, 39000000, 40000000, 41000000, 42000000, 43000000, 44000000, 45000000, 46000000, 47000000, 48000000, 49000000, 50000000, 51000000, 52000000, 53000000, 54000000, 55000000, 56000000, 57000000, 58000000, 59000000, 60000000, 61000000, 62000000, 63000000, 64000000, 65000000, 66000000, 67000000, 68000000, 69000000, 70000000, 71000000, 72000000, 73000000, 74000000, 75000000, 76000000, 77000000, 78000000, 79000000, 80000000, 81000000, 82000000, 83000000, 84000000, 85000000, 86000000, 87000000, 88000000, 89000000, 90000000, 91000000, 92000000, 93000000, 94000000, 95000000, 96000000, 97000000, 98000000, 102000000, 103000000, 104000000, 105000000, 106000000, 107000000, 108000000, 109000000, 110000000, 111000000, 112000000, 113000000, 114000000, 115000000, 116000000, 117000000, 118000000, 119000000, 120000000, 121000000, 122000000, 123000000, 124000000, 125000000, 126000000, 127000000, 128000000, 129000000, 130000000, 131000000, 132000000, 133000000, 134000000, 135000000, 136000000, 137000000, 138000000, 139000000, 140000000, 141000000, 142000000, 143000000, 144000000, 145000000, 146000000, 147000000, 148000000, 149000000, 150000000, 151000000, 152000000, 153000000, 154000000, 155000000, 156000000, 157000000, 158000000, 159000000, 160000000, 161000000, 162000000, 163000000, 164000000, 165000000, 166000000, 167000000, 168000000, 169000000, 170000000, 171000000, 172000000, 173000000, 174000000, 175000000, 176000000, 177000000, 178000000, 179000000, 180000000, 181000000, 182000000, 183000000, 184000000, 185000000, 186000000, 187000000, 188000000, 189000000, 190000000]

	y = [0.9999999999999999, 0.99885645260956, 0.9974311105808427, 0.9960074326132003, 0.9959995957265321, 0.9945199269582464, 0.9931164428086602, 0.9930318965884148, 0.9915079622449041, 0.9901181873924185, 0.9900430984442214, 0.988740992389765, 0.9874614568427765, 0.986353389665892, 0.9851689683303827, 0.9836927822170861, 0.9853209269735335, 0.9841034462167191, 0.982810619203339, 0.9815565473366106, 0.9815701086549344, 0.9802461778357372, 0.978874920090806, 0.9787134869851454, 0.9771000286455911, 0.9754839910415343, 0.9751371632950762, 0.9735128867372135, 0.9746384093334733, 0.9731239053059902, 0.9728606011090216, 0.9700676857633155, 0.9685984176348341, 0.9670035196969338, 0.9656203340511008, 0.9644255609213808, 0.9607012331474989, 0.9598330710146519, 0.9591394381964459, 0.9584390634156943, 0.9572005810757539, 0.9561888334248698, 0.9549204828746382, 0.9537279865547517, 0.9526635338528965, 0.9515583949242803, 0.9503488800402982, 0.9490163533404018, 0.9476536284228815, 0.9462987279383445, 0.9450360548149699, 0.943737584736124, 0.9424299189611258, 0.9412146244872983, 0.9401338169305344, 0.9393246272111541, 0.9386787156064863, 0.9379808724800318, 0.9373312268238583, 0.9365562751817701, 0.9357571589377098, 0.9349227375190478, 0.9342616862901365, 0.9338292114419029, 0.9336482143450352, 0.933611194857378, 0.9326580029936277, 0.9313991516676466, 0.9298443646749819, 0.9283729789044997, 0.9295534351139678, 0.92835446927447, 0.9270982218404487, 0.926892593543602, 0.9251950263393294, 0.9231938025894587, 0.9222218424494494, 0.9199952052489219, 0.9190629957707299, 0.9170211878626539, 0.9162345863695777, 0.9143474093565246, 0.9125945368654796, 0.9108915596300637, 0.9093009629548233, 0.9076891404767856, 0.9061342723323658, 0.904520016629948, 0.902662476213928, 0.8961197667067925, 0.8940340114183803, 0.8920815454752704, 0.8903912405776376, 0.8889668314689522, 0.8890982613464282, 0.8882403809729126, 0.8875491000647173, 0.8869714142840108, 0.8864357891660607, 0.8858366971098425, 0.8862392269953182, 0.885333881686853, 0.8843383744904366, 0.8834028969304208, 0.8825085485886415, 0.8817435121388636, 0.8821422502403826, 0.8812827334991449, 0.8803582154019441, 0.8794834517066238, 0.8786475864634031, 0.8777959960792948, 0.8780992309204145, 0.87687337094516, 0.8754666171940914, 0.8736692379554024, 0.8714120353662658, 0.8688511866483286, 0.8661119246733212, 0.8656868640867897, 0.8630237950172297, 0.8604380636707294, 0.8579154793518183, 0.8554607951913495, 0.8530819731731784, 0.8527789330539277, 0.8503699943618414, 0.8478264127613055, 0.8453837289629789, 0.8424109478586121, 0.8390686983247789, 0.8366571722362992, 0.8331268856126783, 0.8299362252793588, 0.8271373329334677, 0.8248086904671876, 0.8228249552355557, 0.8263348772941208, 0.8211050943547433, 0.8201847461157544, 0.8195471510301751, 0.8191820113451452, 0.8190575178810601, 0.8200793101586158, 0.8199686528065141, 0.8196036312768826, 0.8191081266345129, 0.8187291632961053, 0.8206152381986576, 0.8205350408399525, 0.820656730099591, 0.8208589522740561, 0.8211180771736049, 0.8223905139354533, 0.8225658615319551, 0.8226592962835253, 0.822609464652916, 0.8223759728757932, 0.8228961710844651, 0.8217153932301856, 0.8199186074692855, 0.8176170192781329, 0.815145650613588, 0.8138547491153507, 0.8114879554057552, 0.809035122032575, 0.8064871208286136, 0.8038831979142498, 0.8021678635198015, 0.7992507216906526, 0.7961493802353964, 0.7927815888852087, 0.7889685033832529, 0.786679354549012, 0.7818457667047843, 0.7769211767446592, 0.7723417341751034, 0.7681740046722978]

	# Perceived ADC codes at 10 MHz
	# hi = 270 mV
	# lo = 10 mV
	#
	# These two form a straight line
	# which should be respective of
	# the ADC's results for any voltage
	# given ideal ADC
	#
	hi = 31660.292390033334
	lo = 1150.4921917960041

	# The real ADC is not ideal and is less sensitive at higher frequencies
	fade = np.interp(freq, x, y)

	mv_per_code = 260/(hi*fade - lo*fade)

	return mv_per_code
