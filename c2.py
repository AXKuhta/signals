from datetime import datetime, timezone
from socket import *
import argparse
import asyncio
import json
import os

from src.misc import parse_angle_expr, parse_time_expr, parse_freq_expr

#
# The calibrator command and control tool prototype
#
# Feature wishlist:
# - Device status check
#	- Calibrator USB Serial backend
#	- Calibrator TCP/IP backend
# - Preset parsing
# - Translation
#	- DDC backend
#		- frequency2.cfg
#		- radar_config.ini
#	- AD9910 backend
#		- A sequence of commands
# - Runtime
#	- Action
#		- Sequence execution
#		- Start prompt
#	- Monitoring
#		- Reply log
#

#
# Pressing questions:
# - Are domain models worth it here? (As opposed to dicts):
#	- Having custom __repr__ borderline useless on JSON
#	- Input validation useful -IF- the models are used to build presets programmatically
#		- Yeah preset construction is required
#			- Imagine specifying 30+ sweeps by hand
#	- Extraneous attribute detection useful
#	- How to deserialize a weakly-typed format?
# - The architecture:
#	- Should anticipate future integration:
#		- A web app backend may want to use this
#		- What should be the affordances?
# 	- The interfaces:
#		- An interpreter will have:
#			- __init__()
#			- run()
#		- What should there be besides an interpreter?
#

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="path to a .json preset file")
args = parser.parse_args()

class Calibrator:
	"""
	There may be two of those in the future.
	"""

	def __init__(self, ip="10.15.15.250", port=80, log_location="log.txt"):
		self.sock = socket(AF_INET, SOCK_STREAM)
		print(f"Calibrator: trying {ip}:{port}")
		self.sock.connect( (ip, port) )
		print("Calibrator: connected")

		self.log = open(log_location, "w")
		self.responses = []
		self.acc = ""

		while self.rx() != "> ":
			pass

	def tx(self, command):
		command = command + "\n"
		self.log.write(command)
		self.sock.send(command.encode())

	def rx(self):
		while not self.responses:
			data = self.sock.recv(16383).decode()
			self.log.write(data)

			for c in data:
				self.acc += c

				if c == "\n" or self.acc.endswith("> "):
					self.responses.append(self.acc)
					self.acc = ""

		return self.responses.pop(0)

	def wait(self, command, flow_control="> "):
		self.tx(command)
		response = ""

		# Treat command prompt invitations as flow control
		while not response.endswith(flow_control):
			response += self.rx()

	def close(self):
		self.sock.close()
		self.log.close()

class PresetInterpreterDDCAndCalibratorV1:
	"""
	An interpreter for the V1 preset format.

	{ "ddc-and-calibrator-v1": {
		"ddc": {
			"config_dir": "c:/workprogs/active",
			"samplerate": 5*1000*1000,
			"frames": 8192,
		},
		"signals": [
			{ "tune": "154 MHz", "level": "350 mV", "emit": "sweep 0 us 900 us 154 MHz 77 1" }
		]
	} }

	This should be sufficient for signal fitting.

	Upgrade path to flexible structure signals:

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

	Top-level versioning - the alternative would be versioning "ddc" or "signals"

	We feel there would only be a limited benefit vs. added complexity with fine grained versioning
	Specifically, the added complexity would be in having interpreter dispatch inside an interpreter
	"""

	def __init__(self, init):
		self.init = init
		self.prep_metadata()
		self.prep_ddc_config()
		self.prep_ddc_frequency_table()
		self.prep_calibrator_command_sequence()

	##############################################################################
	# Storage + metadata
	##############################################################################

	def prep_metadata(self):
		ts_cls = datetime.now(timezone.utc)
		ts_str = ts_cls.replace(microsecond=0).isoformat().replace(":", "_")
		dirname = f"calibrator_v1_{ts_str}"
		dirpath = f"c:/calibrator_data_v1/{dirname}"

		self.preset = json.dumps( {"ddc-and-calibrator-v1": self.init}, indent=2 )
		self.dirpath = dirpath

	def write_metadata(self):
		os.mkdir(self.dirpath)

		with open(f"{self.dirpath}/preset.json", "w") as f:
			f.write(self.preset)

	##############################################################################
	# DDC
	##############################################################################
	def prep_ddc_config(self):
		lines = []

		lines.append("[Общие настройки]")
		lines.append("Путь к настройкам приемника для НР=C:/DDC4_Settings/test_cal.ini")
		lines.append("Путь к настройкам приемника для SPU=C:/DDC4_Settings/fm416x250m_ddc4_spu.ini")
		lines.append(f"Путь к директории для данных={self.dirpath}")
		lines.append("IP адрес контроллера синтезаторов=192.168.1.63")
		lines.append("Порт контроллера синтезаторов=69")
		lines.append("Включить multicast=1")
		lines.append("Multicast-группа=234.5.6.7")
		lines.append("Multicast-порт=25000")
		lines.append("Multicast-интерфейс=10.15.15.1")
		lines.append("Включить watchdog=0")
		lines.append("Watchdog адрес=192.168.0.1")
		lines.append("Watchdog порт=9010")
		lines.append("Старый Thunderbolt=1")
		lines.append("Режим калибратора=0")
		lines.append("Время из ОС=1")
		lines.append("[Настройки режима НР]")
		lines.append("НР прореживание канал 0=1")
		lines.append("НР прореживание канал 1=1")
		lines.append("НР прореживание канал 2=1")
		lines.append("НР прореживание канал 3=1")
		lines.append("Режим синхронизации=1")
		lines.append("[Настройки пассивных наблюдений]")
		lines.append("Трекинг радиоисточников=0")
		lines.append("[Настройки режима SPU]")
		lines.append("Путь к файлу с режимами для спутников=C:/DDC4_Settings/spu_modes.cfg")

		self.ddc_config = "\r\n".join(lines)

	def write_ddc_config(self):
		path = self.init.get("ddc").get("config_dir")

		with open(f"{path}/radar_config.ini", "w", encoding="cp1251") as f:
			f.write(self.ddc_config)

	def prep_ddc_frequency_table(self):
		signals = self.init.get("signals")
		entries = []

		for signal in signals:
			tune_khz = parse_freq_expr(signal.get("tune"), "khz")
			entries.append(f"157000 {tune_khz} 0 900 0 2150 21550\r\n")

		self.ddc_frequency_table = "".join(entries)

	def write_ddc_frequency_table(self):
		path = self.init.get("ddc").get("config_dir")

		with open(f"{path}/frequency2.cfg", "w") as f:
			f.write(self.ddc_frequency_table)

	##############################################################################
	# Calibrator
	##############################################################################
	def prep_calibrator_command_sequence(self):
		signals = self.init.get("signals")
		sequence = []

		sequence.append("seq stop")
		sequence.append("seq reset")

		for signal in signals:
			sequence.append(f"set_level " + signal.get("level"))
			sequence.append(f"seq " + signal.get("emit"))

		self.calibrator_command_sequence = sequence

	def write_calibrator_command_sequence(self):
		for command in self.calibrator_command_sequence:
			self.cal_link.wait(command)

	def run(self):
		"""
		What must happen here:
		- Prep the storage
			- A new directory
		- Prep the DDC
			- Write frequency table
			- Write output location
		- Prep the calibrator
			- Connect
			- Send commands
		- Display the start prompt
		- Monitoring
		- Shutdown

		Frequency table and command sequence construction can happen eagerly or lazily-
		having it happen eagerly unlocks a kind of dry run capability
		"""

		self.write_metadata()

		self.cal_link = Calibrator(log_location=f"{self.dirpath}/log.txt")

		self.write_ddc_config()
		self.write_ddc_frequency_table()
		self.write_calibrator_command_sequence()

		print("NOW WAITING FOR DDC - Press start")

		# Wait does not obey normal flow control
		self.cal_link.tx("wait")

		while self.cal_link.rx() != "Running\n":
			pass

		class Break(Exception):
			pass

		# At the moment just two useful metrics here:
		# - Calibrator's uptime in ms
		# - Calibrator's peak heap usage in bytes
		#
		# What is lacking:
		# - Trigger count
		#
		# Some ways this could evolve:
		# - [EARLIER] Active polling
		#	- Running commands and waiting for responses
		#	- Relies on existing CLI infra
		#	- Not readily machine-readable
		#	- Incurs some round trip latency
		# - [PRESENTLY] Inline event stream:
		#	- Running a special command that streams telemetry
		#	- Too, relies on existing infra
		#	- The stream could be machine readable
		#	- But precludes command execution until a stop
		#		- Stop occurs when c2 disconnects
		#		- Stop occurs when DDC stop detected
		# - Separate event stream:
		#	- A separate TCP server
		#	- Machine readable event stream
		#	- Could allow multiple clients
		async def monitor():
			async def break_on_input():
				print("NOW RUNNING - Press enter to stop")
				await asyncio.to_thread(input)
				raise Break()

			async def calibrator_polls():
				prev_ms = [None]

				def poll():
					msg = self.cal_link.rx()

					if msg == "Stop\n":
						print("DDC stop detected - press enter")
						raise Break()

					time, event = msg.strip().split("\t")
					time_ms = int(time) / 216 / 1000

					delta = time_ms - (prev_ms[0] or 0.0)
					prev_ms[0] = time_ms

					print(f"+{delta:.1f}ms\t", event)

				while True:
					await asyncio.to_thread(poll)

			try:
				await asyncio.gather(
					calibrator_polls(),
					break_on_input()
				)
			except Break as e:
				print("BRK")

		try:
			asyncio.run(monitor())
		except KeyboardInterrupt as e:
			print("E-stop")
		finally:
			pass

		# Link close will cause sequencer stop
		self.cal_link.close()

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
	def __init__(self, init):
		self.init = init
		self.prep_metadata()
		self.prep_ddc_config()
		self.prep_ddc_frequency_table()
		self.prep_calibrator_command_sequence()

	def prep_calibrator_command_sequence(self):
		signals = self.init.get("signals")
		sequence = []

		sequence.append("seq stop")
		sequence.append("seq reset")

		for signal in signals:
			sequence.append("set_level " + signal.get("level"))
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

		for chip in signal.get("emit"):
			hold_ns = round( parse_time_expr( chip.get("hold"), into="ns" ) )
			asf = round(16383 * float( chip.get("amplitude") ))
			ftw = round(parse_freq_expr(chip.get("frequency")) / fstep)
			pow = round(65535 * parse_angle_expr(chip.get("phase"), into="deg") / 360.0 )

			insert(hold_ns, asf, ftw, pow)

		payload = {"v2": {
			"profiles": profiles,
			"logic_level_sequence": logic_level_sequence
		}}

		print(json.dumps(payload))

		return json.dumps(payload)

# Interpreter dispatch
interpreters = {
	"ddc-and-calibrator-v1": PresetInterpreterDDCAndCalibratorV1,
	"ddc-and-calibrator-v2": PresetInterpreterDDCAndCalibratorV2
}

with open(args.filename) as f:
	obj = json.load(f)

assert len(obj) == 1, "Malformed preset"

for k, v in interpreters.items():
	if k in obj:
		interpreter_str = k
		interpreter_cls = v

x = interpreter_cls( obj[interpreter_str] )
x.run()
