import torch
from ...config_utils import PretrainedSaeConfig



class OpenSaeConfig(PretrainedSaeConfig):
    def __init__(
        self,
        hidden_size: int = 4096,
        feature_size: int = 131072,
        input_normalize: bool = True,
        input_hookpoint: str = "layers.0",
        output_hookpoint: str = "layers.0",
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        activation: str = "topk",
        dtype: torch.dtype | None = None,
        k: int | None = 128,
        multi_topk: int | None = 4,
        jumprelu_delta: float | None = 0.5,
        normalize_decoder: bool = True,
        **kwargs
    ):
        super().__init__(
            hidden_size      = hidden_size,
            feature_size     = feature_size,
            input_normalize  = input_normalize,
            input_hookpoint  = input_hookpoint,
            output_hookpoint = output_hookpoint,
            model_name       = model_name,
            activation       = activation,
            dtype            = dtype,
            **kwargs
        )
        
        self.normalize_decoder = normalize_decoder

        if activation == "topk":
            assert k is not None and k > 0, "k must be greater than 0 when using topk activation"
            self.k = k
            
            self.multi_topk = multi_topk
            
            if self.multi_topk is not None and self.multi_topk < 1e-3:
                self.multi_topk = None
            
            if self.multi_topk:
                assert self.multi_topk * k < feature_size, "multi_topk * k must be less than num_latents"
    
        elif activation == "jumprelu":
            assert self.jumprelu_delta is not None, "jumprelu_delta must be provided when using jumprelu activation"
            self.jumprelu_delta = jumprelu_delta