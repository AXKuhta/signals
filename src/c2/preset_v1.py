
from datetime import datetime, timezone
import json
import os

from .calibrator import Calibrator
from src.misc import parse_freq_expr
from src.schemas.v1 import *

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

	def __init__(self, preset):
		self.preset = JsonDDCAndCalibratorV1.deserialize(preset)
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

		self.preset_json = json.dumps( {"ddc-and-calibrator-v1": self.preset.serialize() }, indent=2 )
		self.dirpath = dirpath

	def write_metadata(self):
		os.mkdir(self.dirpath)

		with open(f"{self.dirpath}/preset.json", "w") as f:
			f.write(self.preset_json)

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
		path = self.preset.ddc.config_dir

		with open(f"{path}/radar_config.ini", "w", encoding="cp1251") as f:
			f.write(self.ddc_config)

	def prep_ddc_frequency_table(self):
		signals = self.preset.signals
		entries = []

		for signal in signals:
			tune_khz = parse_freq_expr(signal.tune, "khz")
			entries.append(f"157000 {tune_khz} 0 900 0 2150 21550\r\n")

		self.ddc_frequency_table = "".join(entries)

	def write_ddc_frequency_table(self):
		path = self.preset.ddc.config_dir

		with open(f"{path}/frequency2.cfg", "w") as f:
			f.write(self.ddc_frequency_table)

	##############################################################################
	# Calibrator
	##############################################################################
	def prep_calibrator_command_sequence(self):
		signals = self.preset.signals
		sequence = []

		sequence.append("seq stop")
		sequence.append("seq reset")

		for signal in signals:
			sequence.append(f"set_level " + signal.level)
			sequence.append(f"seq " + signal.emit)

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
