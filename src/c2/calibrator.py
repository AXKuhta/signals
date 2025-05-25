
from socket import *

class Calibrator:
	"""
	There may be two of those in the future.
	"""

	def __init__(self, ip="192.168.0.12", port=80, log_location="log.txt"):
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
