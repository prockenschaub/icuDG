from pathlib import Path
import yaml
from types import SimpleNamespace

# Load the configuration from the yaml file
with open("config.yml", "r") as stream:
    Constants = yaml.safe_load(stream)
Constants = Constants['eicu']

# Change to path objects to ensure compatibility with original ClinicalDG code
# https://github.com/MLforHealth/ClinicalDG/tree/72154a87a6d36416c0dac36e7a846b1194c7f39c
Constants['eicu_dir'] = Path(Constants['eicu_dir'])
Constants['benchmark_dir'] = Path(Constants['benchmark_dir'])  

# Convert to namespace to allow accessing configurations via .
Constants = SimpleNamespace(**Constants)