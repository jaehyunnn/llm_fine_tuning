"""
Utility functions for training models with DeepSpeed ZeRO-3 and PEFT (LoRA).
"""
import torch
import logging
from typing import Dict, Iterator, Tuple

# Defer importing `transformers` to reduce dependencies.
try:
    import transformers
except ImportError:
    transformers = None

logger = logging.getLogger(__name__)


def is_deepspeed_zero3_enabled() -> bool:
    """Checks if DeepSpeed ZeRO-3 is enabled."""
    try:
        # Use a little trick to check if DeepSpeed is enabled, as it hides its global state.
        from deepspeed import zero
        return zero.is_zero_supported_optimizer(None) and zero.get_zero_stage() == 3
    except Exception:
        return False


def gather_deepspeed_parameter(param: torch.Tensor) -> torch.Tensor:
    """
    Gathers a parameter partitioned by DeepSpeed ZeRO-3 and returns it as a CPU tensor.
    If not using DeepSpeed, it simply detaches the parameter and moves it to the CPU.

    Args:
        param (torch.Tensor): The parameter to gather.

    Returns:
        torch.Tensor: The gathered parameter on the CPU.
    """
    if not is_deepspeed_zero3_enabled() or not hasattr(param, "ds_id"):
        # If it's not a partitioned parameter, just move it to the CPU.
        return param.detach().cpu()
    
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    
    if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
        logging.warning(f"Cannot gather parameter(id: {param.ds_id}). (Status: {param.ds_status})")

    # Gather the parameter using the `GatheredParameters` context manager.
    with zero.GatheredParameters([param]):
        return param.data.detach().cpu()


def get_peft_state_dict(
    named_parameters: Iterator[Tuple[str, torch.Tensor]],
    bias: str = "none"
) -> Dict[str, torch.Tensor]:
    """
    Extracts the state_dict for a PEFT (LoRA) model. Supports DeepSpeed ZeRO-3 environments.

    Args:
        named_parameters: An iterator of the model's `named_parameters()`.
        bias: The type of bias to include.
              - "none": Includes only LoRA parameters.
              - "all": Includes all parameters containing "lora_" or "bias".
              - "lora_only": Includes only parameters of LoRA layers and their corresponding biases.

    Returns:
        Dict[str, torch.Tensor]: The PEFT state_dict on the CPU.
    """
    if bias == "none":
        params_to_save = {k: v for k, v in named_parameters if "lora_" in k}
    elif bias == "all":
        params_to_save = {k: v for k, v in named_parameters if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        params_to_save = {}
        lora_params = {k: v for k, v in named_parameters if "lora_" in k}
        lora_bias_names = {k.split("lora_")[0] + "bias" for k in lora_params}
        
        # Find the bias parameters that correspond to the LoRA parameters.
        bias_params = {k: v for k, v in named_parameters if k in lora_bias_names}
        
        params_to_save.update(lora_params)
        params_to_save.update(bias_params)
    else:
        raise NotImplementedError(f"Unsupported bias option: {bias}")

    return {k: gather_deepspeed_parameter(v) for k, v in params_to_save.items()}


def get_base_model_state_dict(
    named_parameters: Iterator[Tuple[str, torch.Tensor]],
    require_grad_only: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Extracts the state_dict of the base model, excluding PEFT parameters. Supports DeepSpeed ZeRO-3 environments.

    Args:
        named_parameters: An iterator of the model's `named_parameters()`.
        require_grad_only: Whether to include only parameters with `requires_grad=True`.

    Returns:
        Dict[str, torch.Tensor]: The base model state_dict on the CPU.
    """
    params_to_save = {k: v for k, v in named_parameters if "lora_" not in k}
    if require_grad_only:
        params_to_save = {k: v for k, v in params_to_save.items() if v.requires_grad}

    return {k: gather_deepspeed_parameter(v) for k, v in params_to_save.items()}


def save_hf_trainer_model(trainer: "transformers.Trainer", output_dir: str) -> None:
    """
    Safely saves a model from a HuggingFace Trainer, supporting both DeepSpeed and standard environments.

    When using DeepSpeed, it calls `save_model` to correctly save the distributed model.
    In a standard environment, it moves the state_dict to the CPU to prevent OOM errors.

    Args:
        trainer (transformers.Trainer): The Trainer instance containing the model to save.
        output_dir (str): The directory path to save the model to.
    """
    if trainer.deepspeed:
        # DeepSpeed handles gathering and saving parameters from all nodes.
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    if trainer.args.should_save:
        # Move the model's state_dict to the CPU to avoid OOM errors.
        state_dict = trainer.model.state_dict()
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        
        # Use the internal `_save` function to pass the prepared cpu_state_dict.
        trainer._save(output_dir, state_dict=cpu_state_dict)
        trainer.model.config.save_pretrained(output_dir)
