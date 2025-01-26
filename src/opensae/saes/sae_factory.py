__description__ = """
    Implement Factory Class for SAE Models
"""

import os
import sys

import json
from pathlib import Path
from collections import OrderedDict

import torch
import transformers


from ..config_utils import PretrainedSaeConfig


from . import (
    OpenSae,
)


SAE_CLASS_MAP = OrderedDict(
    [
        ("OpenSae", OpenSae),
    ]
)


class AutoSae:
    def __init__(self):
        pass
    
    @staticmethod
    def from_pretrained(pretrained_sae_path: str, **kwargs):
        config_path = Path(pretrained_sae_path) / "config.json"
        sae_config = json.loads(config_path.read_text(), object_pairs_hook=OrderedDict)
        
        architecture = sae_config["architectures"][0]
        
        if architecture in SAE_CLASS_MAP:
            sae_class = SAE_CLASS_MAP[architecture]
        else:
            raise ValueError(f"Unrecognized SAE architecture {architecture}")

        return sae_class.from_pretrained(pretrained_sae_path, **kwargs)