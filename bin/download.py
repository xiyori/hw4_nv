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

train_config = config_.TrainConfig()
data_dir = os.path.abspath(train_config.data_dir)
checkpoint_path = os.path.abspath("./resources/chkpoints")
results_path = os.path.abspath("./resources/results")

os.makedirs(results_path, exist_ok=True)
os.chdir("./resources")

subprocess.run(["../bin/download.sh", data_dir,
                checkpoint_path, results_path], env=env)
