import argparse
import asyncio
import json

from src.c2 import PresetInterpreterDDCAndCalibratorV1, PresetInterpreterDDCAndCalibratorV2

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="path to a .json preset file")
args = parser.parse_args()

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
