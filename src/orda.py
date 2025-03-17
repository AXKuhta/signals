
from datetime import datetime, timezone
import struct

import numpy as np

#
# Loader for the bespoke ORDA .ISE/.SPU format used at Irkutsk Incoherent Scatter Radar
#

class ORDACap:
	"""
	A capture and its metadata

	Stores:
	- Trigger number
	- Channel number
	- Timestamp
	- Center frequency (Hz)
	- Samplerate (Hz)
	- Samplecount
	- I/Q samples (complex128)
	"""

	trigger_number = None
	channel_number = None
	timestamp = None
	center_freq = None
	samplerate = None
	samplecount = None
	iq = None

	def __init__(self, trigger_number, channel_number, timestamp, center_freq, samplerate, samplecount, iq_bytes):
		self.trigger_number = trigger_number
		self.channel_number = channel_number
		self.timestamp = timestamp
		self.center_freq = center_freq
		self.samplerate = samplerate
		self.samplecount = samplecount

		imag, real = np.frombuffer(iq_bytes, dtype=np.int16).reshape(2, self.samplecount) / 1.0
		self.iq = real + 1j*imag

	@property
	def basic(self):
		"""
		Basic unpack interface

		Returns, in that order:
		- Trigger number
		- Channel number
		- Center frequency (Hz)
		- I/Q samples

		Example:

		```
		i, ch, fc, iq = blk.basic
		```

		"""

		yield self.trigger_number
		yield self.channel_number
		yield self.center_freq
		yield self.iq

	def __repr__(self):
		return f"ORDACap(trigger_number={self.trigger_number}, channel={self.channel_number}, ...)"

	def __str__(self):
		return (
			f"ORDACap("
			f"\n\ttrigger_number={self.trigger_number},"
			f"\n\tchannel={self.channel_number},"
			f"\n\ttimestamp={self.timestamp},"
			f"\n\tcenter_freq={self.center_freq},"
			f"\n\tsamplerate={self.samplerate},"
			f"\n\tsamplecount={self.samplecount},"
			f"\n\tiq={self.iq}"
			f"\n)"
		)

class StreamORDA:
	def __init__(self, fd):
		self.type = None
		self.data = None
		self.center_freq = None
		self.samplerate = None
		self.samples = None
		self.channel = None
		self.timestamp = None
		self.ch_blocks = [0, 0, 0, 0]
		self.fd = fd

	def parse_superheader(self, type, header):
		fields = len(header) / 4
		pairs = {}

		assert fields % 1 == 0

		for i in range( int(fields) ):
			k_lo, k_hi, v_lo, v_hi = header[i*4:(i+1)*4]

			k = k_hi*256 + k_lo
			v = v_hi*256 + v_lo

			pairs[k] = v

		# Useful
		# print(pairs, type)

		if type == 3:
			self.samplerate = pairs[30] * 1000
			self.samples = pairs[3]

		if type == 1:
			self.channel = pairs[7]
			self.center_freq = (pairs[17]*65536 + pairs[16]) * 1000
			self.timestamp = datetime(
				year = pairs[9],
				month = pairs[10] // 256,
				day = pairs[10] & 0xFF,
				hour = pairs[11] & 0xFF,
				minute = pairs[11] // 256,
				second = pairs[12],
				microsecond = pairs[13] * 1000,
				tzinfo = timezone.utc
			)

	def advance(self):
		header = self.fd.read(9)

		if len(header) == 0:
			return False

		orda_magic, type, size = struct.unpack("<4sBI", header)
		data = self.fd.read(size)

		assert orda_magic == b"ORDA"
		assert len(data) == size

		self.type = type
		self.data = data

		return True

	def read_capture(self):
		while self.advance():
			if self.type == 3: # Global header
				self.parse_superheader(self.type, self.data)
			elif self.type == 2: # I/Q Samples
				result = ORDACap(
					trigger_number=self.ch_blocks[self.channel],
					channel_number=self.channel,
					timestamp=self.timestamp,
					center_freq=self.center_freq,
					samplerate=self.samplerate,
					samplecount=self.samples,
					iq_bytes=bytearray(self.data)
				)

				self.ch_blocks[self.channel] += 1

				return result
			elif self.type == 1: # Local header
				self.parse_superheader(self.type, self.data)
			else:
				pass

		#print(f"orda_stream block count: {self.ch_blocks}")

		return None

	@property
	def captures(self):
		"""
		Iterator for capture objects

		Example:

		```
		with open("xxxxxxx.ISE", "rb") as f:
			for capture in StreamORDA(f).captures:
				print( str(capture) )
		```

		"""

		while True:
			result = self.read_capture()

			if result:
				yield result
			else:
				return

	@property
	def basics(self):
		"""
		Iterator for basic unpack interface

		Example:

		```
		with open("xxxxxxx.ISE", "rb") as f:
			for trig, chan, freq, iq in StreamORDA(f).basics:
				print(trig, chan, freq)
		```

		"""
		while True:
			result = self.read_capture()

			if result:
				yield result.basic
			else:
				return

	def all_captures(self):
		"""
		Returns a list of captures, no iterators

		Example:

		```
		with open("xxxxxxx.ISE", "rb") as f:
			captures = StreamORDA(f).all_captures():
		```

		"""
		return list(self.captures)
