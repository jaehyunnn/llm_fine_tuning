from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class ModelArchitecture(Enum):
    LLAMA = "llama"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    PHI = "phi"
    QWEN = "qwen"


@dataclass
class ModelConfig:
    """Base configuration for all models."""
    model_name: str
    architecture: ModelArchitecture
    supports_flash_attention: bool = True
    supports_gradient_checkpointing: bool = True
    default_max_length: int = 2048
    
    # Model-specific parameters
    freeze_base_model: bool = False
    use_cache: bool = False
    
    # LoRA configuration
    lora_target_modules: Optional[List[str]] = None
    lora_exclude_modules: Optional[List[str]] = None
    
    # Quantization support
    supports_4bit: bool = True
    supports_8bit: bool = True
    quantization_skip_modules: Optional[List[str]] = None



@dataclass 
class TextOnlyModelConfig(ModelConfig):
    """Configuration for text-only language models."""
    
    # Text-specific settings
    supports_system_messages: bool = True
    default_system_message: str = ""
    chat_template: Optional[str] = None


# Predefined configurations for popular models
MODEL_CONFIGS = {
    # Text-only models
    "llama-2-7b": TextOnlyModelConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        architecture=ModelArchitecture.LLAMA,
        default_max_length=4096,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ),
    "llama-2-13b": TextOnlyModelConfig(
        model_name="meta-llama/Llama-2-13b-hf", 
        architecture=ModelArchitecture.LLAMA,
        default_max_length=4096,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ),
    "mistral-7b": TextOnlyModelConfig(
        model_name="mistralai/Mistral-7B-v0.1",
        architecture=ModelArchitecture.MISTRAL,
        default_max_length=32768,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ),
    "phi-3-mini": TextOnlyModelConfig(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        architecture=ModelArchitecture.PHI,
        default_max_length=4096,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ),
    "gemma-7b": TextOnlyModelConfig(
        model_name="google/gemma-7b",
        architecture=ModelArchitecture.GEMMA,
        default_max_length=8192,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ),
    "qwen2-7b": TextOnlyModelConfig(
        model_name="Qwen/Qwen2-7B-Instruct",
        architecture=ModelArchitecture.QWEN,
        default_max_length=32768,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ),
    "qwen3-0.6b": TextOnlyModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        architecture=ModelArchitecture.QWEN,
        default_max_length=32768,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ),
}


def get_model_config(model_id: str) -> ModelConfig:
    """Get model configuration by model ID or path."""
    # First check if it's a predefined config
    for key, config in MODEL_CONFIGS.items():
        if model_id.endswith(key) or config.model_name == model_id:
            return config
    
    # Try to infer from model name
    model_lower = model_id.lower()
    
    
    # Text-only models
    if "llama" in model_lower:
        return TextOnlyModelConfig(
            model_name=model_id,
            architecture=ModelArchitecture.LLAMA,
            default_max_length=4096
        )
    elif "mistral" in model_lower:
        return TextOnlyModelConfig(
            model_name=model_id,
            architecture=ModelArchitecture.MISTRAL,
            default_max_length=32768
        )
    elif "phi" in model_lower:
        return TextOnlyModelConfig(
            model_name=model_id,
            architecture=ModelArchitecture.PHI,
            default_max_length=4096
        )
    elif "gemma" in model_lower:
        return TextOnlyModelConfig(
            model_name=model_id,
            architecture=ModelArchitecture.GEMMA,
            default_max_length=8192
        )
    elif "qwen" in model_lower:
        return TextOnlyModelConfig(
            model_name=model_id,
            architecture=ModelArchitecture.QWEN,
            default_max_length=32768
        )
    
    # Default fallback
    return TextOnlyModelConfig(
        model_name=model_id,
        architecture=ModelArchitecture.LLAMA,  # Most common architecture
        default_max_length=2048
    )