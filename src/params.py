from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""
    model_id: Optional[str] = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "The model identifier on huggingface.co/models"}
    )
    model_config_override: Optional[str] = field(
        default=None, 
        metadata={"help": "Override model configuration key from predefined configs"}
    )

@dataclass
class TrainingArguments(HFTrainingArguments):
    """Arguments pertaining to the training configuration."""
    
    # General Training
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)
    max_seq_length: int = field(
        default=None,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). If None, will use model's default."}
    )

    # Model Freezing & Optimizations
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Freeze the base model layers."}
    )
    disable_flash_attn2: bool = field(
        default=False,
        metadata={"help": "Disable Flash Attention 2."}
    )
    use_liger: bool = field(
        default=True,
        metadata={"help": "Enable Liger kernel optimizations for supported models."}
    )

    # Quantization Config
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use for quantization (e.g., 4, 8, 16)."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    
    # LoRA & DoRA Config
    use_lora: bool = field(default=False, metadata={"help": "Enable LoRA."})
    use_dora: bool = field(default=False, metadata={"help": "Enable DoRA (Weight-Decomposed Low-Rank Adaptation)."})
    lora_rank: int = field(default=64, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_weight_path: str = field(default="", metadata={"help": "Path to LoRA weights."})
    lora_bias: str = field(default="none", metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"})
    lora_namespan_exclude: str = field(
        default=None, 
        metadata={"help": "List of namespan to exclude for LoRA, as a string representation of a list."}
    )
    num_lora_modules: int = field(
        default=-1, 
        metadata={"help": "Number of LoRA modules to apply. -1 means all linear layers."}
    )

@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    
    # General Data Config
    data_path: str = field(
        default=None, 
        metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = field(
        default=False, 
        metadata={"help": "Whether to perform preprocessing lazily."}
    )