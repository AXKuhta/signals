from .deserializer import Schema, Field

class JsonDDCSettingsV1(Schema):
	config_dir = Field(str)
	samplerate = Field(str)
	frames = Field(int)

class JsonSignalV1(Schema):
	tune = Field(str)
	level = Field(str)
	emit = Field(str)

class JsonDDCAndCalibratorV1(Schema):
	ddc = Field(JsonDDCSettingsV1)
	signals = Field([JsonSignalV1])
