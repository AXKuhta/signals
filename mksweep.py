
import argparse
import json

from src.misc import parse_freq_expr, parse_time_expr, parse_volt_expr, pretty_freq

parser = argparse.ArgumentParser()
parser.add_argument("--start", help="start frequency e.g. \"10 MHz\"", required=True)
parser.add_argument("--stop", help="stop frequency e.g. \"390 MHz\"", required=True)
parser.add_argument("--step", help="step frequency e.g. \"1 MHz\"", required=True)
parser.add_argument("--delay", help="trigger delay, defaults to \"0 us\"")
parser.add_argument("--duration", help="pulse duration, defaults to \"900 us\"")
parser.add_argument("--a", help="ad9910 a parameter", required=True)
parser.add_argument("--b", help="ad9910 b parameter, usually 1", required=True)
parser.add_argument("--level", help="signal level e.g. \"60 mV\"", required=True)
args = parser.parse_args()

delay = args.delay or "0 us"
duration = args.duration or "900 us"

parse_time_expr(delay)
parse_time_expr(duration)

start = parse_freq_expr(args.start)
stop = parse_freq_expr(args.stop)
step = parse_freq_expr(args.step)

assert (stop - start) % step == 0.0, "Please specify a band that is divisible by frequency step"

a = int(args.a)
b = int(args.b)

parse_volt_expr(args.level)

ddc = {
      "config_dir": "c:/workprogs/calibrator_mode/",
      "samplerate": "5 MHz",
      "frames": 8192
}

signals = []

freq = start

while freq <= stop:
	signals.append({
		"tune": f"{pretty_freq(freq)}",
		"level": args.level,
		"emit": f"sweep {delay} {duration} {pretty_freq(freq)} {a} {b}"
	})

	freq += step

preset = {
	"ddc-and-calibrator-v1": {
		"ddc": ddc,
		"signals": signals
	}
}

print(json.dumps(preset, indent=2))
