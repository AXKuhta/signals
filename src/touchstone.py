import numpy as np

#
# Loader for s2p files
#
class S2PFile:
	def __init__(self, fd):
		assert fd.mode == "rb"

		freqs = []
		s11 = []
		s21 = []
		s12 = []
		s22 = []

		for line in fd:
			if line[0] in [b"!"[0], b"#"[0]]:
				continue

			columns = line.replace(b"\t", b"  ").split(b"  ")
			numbers = [float(x) for x in columns]

			freq, s11re, s11im, s21re, s21im, s12re, s12im, s22re, s22im = numbers

			freqs.append(freq)
			s11.append( s11re + 1j*s11im )
			s21.append( s21re + 1j*s21im )
			s12.append( s12re + 1j*s12im )
			s22.append( s22re + 1j*s22im )

		self.freqs = freqs
		self.s11 = np.array(s11)
		self.s21 = np.array(s21)
		self.s12 = np.array(s12)
		self.s22 = np.array(s22)
