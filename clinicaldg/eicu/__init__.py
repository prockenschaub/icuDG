import yaml
from types import SimpleNamespace

with open("config.yml", "r") as stream:
    Constants = yaml.safe_load(stream)

Constants = SimpleNamespace(**Constants['eicu'])