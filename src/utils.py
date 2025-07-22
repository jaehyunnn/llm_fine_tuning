from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig, AutoProcessor, AutoConfig, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
import warnings
import os
import json

def disable_torch_init() -> None:
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def is_lora_model(model_path: str | Path) -> bool:
    """
    Check if a model directory contains LoRA adapter files.
    
    Args:
        model_path: Path to the model directory.
        
    Returns:
        True if the directory contains LoRA adapter files.
    """
    model_dir = Path(model_path)
    return (model_dir / 'adapter_config.json').exists() and (model_dir / 'adapter_model.safetensors').exists()

def _prepare_model_load_kwargs(
    load_8bit: bool, 
    load_4bit: bool,
    use_flash_attn: bool,
    device: str,
    device_map: str
) -> Dict[str, Any]:
    """Prepares a dictionary of keyword arguments for model loading."""
    kwargs = {"device_map": device_map if device == "cuda" else {"": device}}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['_attn_implementation'] = 'flash_attention_2'
    
    return kwargs

def _load_lora_model(
    model_path: str,
    model_base: str,
    model_name: str,
    load_kwargs: Dict[str, Any]
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Loads a base model, applies LoRA weights, and merges them."""
    lora_config = AutoConfig.from_pretrained(model_path)
    if hasattr(lora_config, 'quantization_config'):
        del lora_config.quantization_config
        
    processor = AutoProcessor.from_pretrained(model_base)
    
    print(f'Loading {model_name} from base model: {model_base}...')
    model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_config, **load_kwargs)
    
    token_num, token_dim = model.lm_head.out_features, model.lm_head.in_features
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype))
        model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype))

    print(f'Loading additional {model_name} weights...')
    non_lora_weights_path = Path(model_path) / 'non_lora_state_dict.bin'
    if non_lora_weights_path.exists():
        base_model_trainables = torch.load(str(non_lora_weights_path), map_location='cpu')
        
        cleaned_state_dict = {}
        for k, v in base_model_trainables.items():
            if k.startswith("base_model."):
                k = k[11:]
            if k.startswith("model."):
                k = k[6:]
            cleaned_state_dict[k] = v
        model.load_state_dict(cleaned_state_dict, strict=False)

    print(f'Loading LoRA weights from {model_path}...')
    model = PeftModel.from_pretrained(model, model_path)

    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    
    return processor, model

def _load_standard_model(
    model_path: str, 
    load_kwargs: Dict[str, Any]
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Loads a standard Hugging Face model."""
    print(f"Loading model from {model_path} as a standard model since adapter files were not found.")
    config_path = Path(model_path) / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Model Architecture: {config['architectures'][0]}")

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **load_kwargs)
    return processor, model

def load_pretrained_model(
    model_path: str, 
    model_base: Optional[str] = None, 
    model_name: Optional[str] = None, 
    load_8bit: bool = False, 
    load_4bit: bool = False, 
    device_map: str = "auto", 
    device: str = "cuda", 
    use_flash_attn: bool = False,
    **kwargs
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Loads a pretrained model, handling both standard and LoRA-merged models.

    Args:
        model_path: Path to the model to load.
        model_base: Path to the base model (required for LoRA).
        model_name: Name of the model for logging purposes.
        load_8bit: Whether to load in 8-bit mode.
        load_4bit: Whether to load in 4-bit mode.
        device_map: Device map for model loading.
        device: Device to load the model on if not 'cuda'.
        use_flash_attn: Whether to use Flash Attention 2.

    Returns:
        A tuple containing the processor and the loaded model.
    """
    load_kwargs = _prepare_model_load_kwargs(load_8bit, load_4bit, use_flash_attn, device, device_map)
    model_name = model_name or get_model_name_from_path(model_path)
    
    is_lora = is_lora_model(model_path)
    
    if is_lora:
        if model_base is None:
            raise ValueError('A `model_base` must be provided to load a LoRA model.')
        processor, model = _load_lora_model(model_path, model_base, model_name, load_kwargs)
    else:
        processor, model = _load_standard_model(model_path, load_kwargs)

    print(f'✅ Model "{model_name}" successfully Loaded!')
    return processor, model

def get_model_name_from_path(model_path: str) -> str:
    """
    Extracts a model name from a file path.

    If the path ends with 'checkpoint-xxxxx', it combines the parent directory
    name with the checkpoint name. Otherwise, it returns the final directory name.

    Args:
        model_path: The file path to the model.

    Returns:
        The extracted model name.
    """
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return f"{model_paths[-2]}_{model_paths[-1]}"
    else:
        return model_paths[-1]