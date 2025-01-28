from dataclasses import dataclass
from simple_parsing import Serializable, list_field, field

from multiprocessing import cpu_count


@dataclass
class SaeConfig(Serializable):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features."""
    
    multi_topk: bool = False
    """Use Multi-TopK loss."""
    
    l1_coef: float | None = None
    """L1 regularization coefficient."""
    
    input_normalize: bool = True
    """Normalize the input hidden states to have unit norm."""
    
    shift_back: bool = False


@dataclass
class TrainConfig(Serializable):
    seed: int = 42
    """Random seed to use for torch, numpy and dataloader."""

    dp_size: int = 1
    """Data Parallelism size. This means that the same SAE is copied across `dp_size` devices."""
    
    fsdp: bool = False
    """Whether to Use Fully Sharded Data Parallelism."""
    
    mp_size: int = 1
    """Model Parallelism size. This means that we will train `mp_size` SAEs in parallel."""

    local_batch_size: int = 8
    """Batch size measured in sequences without gradient accumulation."""
    
    global_batch_size: int = 512
    """Batch size measured in sequences after gradient accumulation."""

    grad_acc_steps: int | None = None
    """Number of steps over which to accumulate gradients."""

    micro_acc_steps: int = 1
    """Chunk the activations into this number of microbatches for SAE training."""

    lr: float | None = None
    """Base LR. If None, it is automatically chosen based on the number of latents."""
    
    min_lr_ratio: float = 0.02
    """Minimum LR as a ratio of the base LR."""

    lr_scheduler_type: str = "linear"
    """Use a warmup-stable-decay LR schedule."""

    lr_warmup_ratio: float | None = None
    """Fraction of steps to use for LR warmup."""

    lr_warmup_steps: int | None = None
    """Number of steps to use for LR warmup."""
    
    lr_stable_steps: int | None = None
    """Number of steps to use for LR stable phase."""
    
    lr_decay_ratio: float | None = None
    """Fraction of steps to use for LR decay phase."""
    
    lr_decay_steps: int | None = None
    """Number of steps to use for LR decay phase."""
    
    auxk_alpha: float = 0.0
    """Weight of the auxiliary loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    hookpoint: str | None = None
    """List of hookpoints to train SAEs on."""

    distribute_modules: bool = False
    """Store a single copy of each SAE, instead of copying them across devices."""

    save_every: int = 1000
    """Save SAEs every `save_every` steps."""
    
    save_at_init: bool = False
    """Save SAEs at initialization."""
    
    save_dir: str = "output/ckpts"
    
    load_dir: str = None
    
    adam_in_8bit: bool = False
    """Use 8-bit gradients for the Adam optimizer."""
    
    spike_detection_start: int = 10
    
    spike_detection_window_size: int = 5
    
    spike_detection_threshold_ratio: float = 1.05
    
    varlen: bool = True
    
    k_scheduler: str = "constant"
    
    k_scheduler_factor: int = 4
    """If use k scheduler, the training process will start at k = k_scheduler_factor * k"""
    
    k_scheduler_step_ratio: float = 0.25
    """The ratio of the total steps to keep k stable"""
    
    early_exit_inference_layer_num: int | None = None
    """The number of layer to use for early exit inference."""
    

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_project: str | None = None
    wandb_id: str | None = None
    wandb_log_frequency: int = 1

    def __post_init__(self):
        assert not (
            self.grad_acc_steps and self.global_batch_size
        ), "Cannot specify both `grad_acc_steps` and `global_batch_size`."
        
        if self.global_batch_size:
            assert self.global_batch_size % (self.local_batch_size * self.dp_size) == 0, "Global batch size must be divisible by local batch size * dp_size"
            self.grad_acc_steps = self.global_batch_size // (self.local_batch_size * self.dp_size)
            
        if self.run_name is None:
            self.run_name = "default_run"
        if self.wandb_project is None:
            self.wandb_project = self.run_name
        if self.wandb_id is None:
            self.wandb_id  = self.run_name
            
        # lr_warmup_step and lr_warmup_ratio are mutually exclusive
        assert not (self.lr_warmup_steps and self.lr_warmup_ratio), "Cannot specify both `lr_warmup_steps` and `lr_warmup_ratio`."
        
        assert self.spike_detection_window_size <= self.spike_detection_start
        
        assert self.k_scheduler in ["constant", "linear", "piecewise", "random", "discrete_linear"]
        
        assert self.k_scheduler_step_ratio > 0 and self.k_scheduler_step_ratio < 1
        
        assert self.lr_scheduler_type in ["linear", "cosine", "wsd"]


@dataclass
class ModelConfig(Serializable):

    model: str = "EleutherAI/pythia-160m"
    
    auto_model_class: str = "AutoModel"         # "AutoModel" or "AutoModelForCausalLM"
    
    trust_remote_code: bool = True

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    def __post_init__(self):
        assert self.auto_model_class in ["AutoModel", "AutoModelForCausalLM"]
        if self.auto_model_class == "AutoModel":
            from transformers import AutoModel
            self.auto_model_class = AutoModel
        else:
            from transformers import AutoModelForCausalLM
            self.auto_model_class = AutoModelForCausalLM

@dataclass
class DataConfig(Serializable):
    
    dataset: str = "togethercomputer/RedPajama-Data-1T-Sample"

    split: str = "train"
    """Dataset split to use for training."""

    data_preprocessing_num_proc: int = field(
        default_factory=lambda: cpu_count() // 2,
    )
    """Number of processes to use for preprocessing data"""
    
    trust_remote_code: bool = True
    
    ctx_len: int = 2048
    """Context length to use for training."""