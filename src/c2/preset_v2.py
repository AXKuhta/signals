
import json

from .preset_v1 import PresetInterpreterDDCAndCalibratorV1
from src.misc import parse_angle_expr, parse_time_expr, parse_freq_expr
from src.schemas.v2 import *

class PresetInterpreterDDCAndCalibratorV2(PresetInterpreterDDCAndCalibratorV1):
	"""
	An interpreter for the V2 preset format.


	{ "ddc-and-calibrator-v2": {
		"ddc": {
			"config_dir": "c:/workprogs/active",
			"samplerate": 5*1000*1000,
			"frames": 8192,
		},
		"signals": [
			{ "tune": "154 MHz", "level": "350 mV", "emit": [
				{ "hold": "900 us", "amplitude": 1.0, "frequency": { "start": "154 MHz", "stop": "156 MHz" }, "phase": "0 deg" }
			] }
		]
	} }
	"""
	def __init__(self, preset):
		self.preset = JsonDDCAndCalibratorV2.deserialize(preset)
		self.prep_metadata()
		self.prep_ddc_config()
		self.prep_ddc_frequency_table()
		self.prep_calibrator_command_sequence()

	def prep_calibrator_command_sequence(self):
		signals = self.preset.signals
		sequence = []

		sequence.append("seq stop")
		sequence.append("seq reset")

		for signal in signals:
			sequence.append("set_level " + signal.level)
			sequence.append("seq json " + self.translate_to_json_payload(signal) )

		self.calibrator_command_sequence = sequence

	def translate_to_json_payload(self, signal):
		"""
		Takes: array of sequence elements, of pulse segments
		Makes: a JSON payload
		"""

		P_0 = 0b00010000
		P_1 = 0b00100000
		P_2 = 0b01000000

		PROFILE0 = (0)
		PROFILE1 = (P_0)
		PROFILE2 = (P_1)
		PROFILE3 = (P_1 | P_0)
		PROFILE4 = (P_2)
		PROFILE5 = (P_2 | P_0)
		PROFILE6 = (P_2 | P_1)
		PROFILE7 = (P_2 | P_1 | P_0)

		PROFILES_ = [
			PROFILE0,
			PROFILE1,
			PROFILE2,
			PROFILE3,
			PROFILE4,
			PROFILE5,
			PROFILE6,
			PROFILE7
		]

		GRAY = [
			0b000,
			0b001,
			0b011,
			0b010,
			0b110,
			0b111,
			0b101,
			0b100
		]

		logic_level_sequence = []

		# For now lets do without intricate profile register allocation
		profiles = [ {"asf": 0, "ftw": 0, "pow": 0} ]*8

		def insert(hold_ns, asf, ftw, pow):
			idx = GRAY.pop(0)

			profiles[idx] = {
				"asf": asf,
				"ftw": ftw,
				"pow": pow
			}

			if hold_ns is not None:
				logic_level_sequence.append({
					"hold_ns": hold_ns,
					"state": PROFILES_[idx]
				})

		# Parking profile
		insert(None, 0, 0, 0)

		fstep = 1000*1000*1000 / 2**32

		for chip in signal.emit:
			hold_ns = round( parse_time_expr( chip.hold, into="ns" ) )
			asf = round(16383 * float( chip.amplitude ))
			ftw = round(parse_freq_expr(chip.frequency) / fstep)
			pow = round(65535 * parse_angle_expr(chip.phase, into="deg") / 360.0 )

			insert(hold_ns, asf, ftw, pow)

		payload = {"v2": {
			"profiles": profiles,
			"logic_level_sequence": logic_level_sequence
		}}

		print(json.dumps(payload))

		return json.dumps(payload)
