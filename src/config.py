"""Reads params.yaml (path can be overridden with the PARAMS_FILE environment variable)."""
import os
import random

import numpy as np
import yaml


def load_params():
    with open(os.environ.get("PARAMS_FILE", "params.yaml")) as f:
        return yaml.safe_load(f)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def flatten(d, prefix=""):
    """{'a': {'b': 1}} -> {'a.b': 1}  (used to log parameters to MLflow)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out
