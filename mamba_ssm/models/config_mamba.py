from dataclasses import dataclass, field


@dataclass
class MambaConfig:

    d_model: int = 12#2560dim
    n_layer: int = 4#64
    vocab_size: int = 1000#50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
