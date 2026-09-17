import os
import tempfile
import time

import numpy as np
import ray
import ray.train
import torch
from datasets import load_dataset
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from transformers import GPT2Config, GPT2LMHeadModel

# Enable smoke test to run this tutorial quickly.
SMOKE_TEST = True

# Reduce Ray Data verbosity
ray.data.DataContext.get_current().enable_progress_bars = False
ray.data.DataContext.get_current().print_on_execution_start = False

