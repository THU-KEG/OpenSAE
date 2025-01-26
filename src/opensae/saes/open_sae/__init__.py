__description__ = """
    Implement OpenSAE --- the Native SAE model trained with OpenSAE
"""

from .configuration_open_sae import OpenSaeConfig
from .modeling_open_sae import OpenSae

__all__ = [
    "OpenSaeConfig",
    "OpenSae",
]