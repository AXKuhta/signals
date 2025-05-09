from .deserializer import Schema, Field
from .v1 import *

class JsonDDCSettingsV2(JsonDDCSettingsV1):
	pass

# Here, chip means one of concatenated sub-pulses
# https://its.ntia.gov/media/31078/DavisRadar_waveforms.pdf
class JsonPulseChipV2(Schema):
	hold = Field(str)
	amplitude = Field(float)
	frequency = Field(str)
	phase = Field(str)

class JsonSignalV2(Schema):
	tune = Field(str)
	level = Field(str)
	emit = Field([JsonPulseChipV2])

class JsonDDCAndCalibratorV2(Schema):
	ddc = Field(JsonDDCSettingsV2)
	signals = Field([JsonSignalV2])
