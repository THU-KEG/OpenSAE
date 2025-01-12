import torch
from ...config_utils import PretrainedSaeConfig



class OpenSaeConfig(PretrainedSaeConfig):
    def __init__(
        self,
        hidden_size: int = 4096,
        feature_size: int = 262144,
        input_normalize: bool = True,
        normalize_shift_back: bool = False,
        input_normalize_eps: float = 1e-5,
        input_hookpoint: str = "layers.0",
        output_hookpoint: str = "layers.0",
        model_name: str = "meta-llama/meta-llama-3.1-8b",
        activation: str = "topk",
        torch_dtype: torch.dtype | None = None,
        # sparse activation related
        k: int | None = 128,
        jumprelu_theta: float | None = 0.5,
        # decoder related
        normalize_decoder: bool = True,
        decoder_impl: str = "triton",
        # loss related
        multi_topk_multiplier: int | None = 4,
        auxk_alpha: float | None = 1e-2,
        l1_coef: float | None = None,
        **kwargs
    ):
        super().__init__(
            hidden_size      = hidden_size,
            feature_size     = feature_size,
            input_normalize      = input_normalize,
            normalize_shift_back = normalize_shift_back,
            input_normalize_eps  = input_normalize_eps,
            input_hookpoint  = input_hookpoint,
            output_hookpoint = output_hookpoint,
            model_name       = model_name,
            activation       = activation,
            torch_dtype      = torch_dtype,
            **kwargs
        )


        if activation == "topk":
            assert k is not None and k > 0, "k must be greater than 0 when using topk activation"
            self.k = k
            
            self.multi_topk = multi_topk_multiplier
            
            if self.multi_topk is not None and self.multi_topk < 1e-3:
                self.multi_topk = None
            
            if self.multi_topk:
                assert self.multi_topk * k < feature_size, "multi_topk * k must be less than num_latents"
    
        elif activation == "jumprelu":
            assert self.jumprelu_theta is not None, "jumprelu_theta must be provided when using jumprelu activation"
            self.jumprelu_theta = jumprelu_theta

        self.normalize_decoder = normalize_decoder
        assert decoder_impl in ["triton", "torch"], "decoder_impl must be either 'triton' or 'torch'"
        self.decoder_impl = decoder_impl

        self.auxk_alpha = auxk_alpha
        self.l1_coef = l1_coef