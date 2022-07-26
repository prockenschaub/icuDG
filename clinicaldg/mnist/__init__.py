import yaml
from types import SimpleNamespace

# Load the configuration from the yaml file
with open("config.yml", "r") as stream:
    Constants = yaml.safe_load(stream)

Constants = SimpleNamespace(**Constants['mnist'])