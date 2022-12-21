import argparse

import os
import subprocess

import sys
sys.path.append(".")

from src import config

parser = argparse.ArgumentParser(description=".")
parser.add_argument("-c", "--config", metavar="CONFIG", type=str,
                    required=True, help="Config filename.")
args = parser.parse_args()

config_ = getattr(config, args.config)

env = os.environ.copy()
if "datasphere" in args.config:
    env["PATH"] += ":/home/jupyter/.local/bin"

data_config = config_.DataConfig()
data_dir = os.path.abspath(data_config.data_dir)
checkpoint_path = os.path.abspath("./resources/chkpoints")
results_path = os.path.abspath("./resources/results")

os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)
os.chdir("./resources")

subprocess.run(["../bin/download.sh", data_dir,
                checkpoint_path, results_path], env=env)
