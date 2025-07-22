import os
import torch
import ast
import pathlib
import importlib

from typing import List, Tuple, Union, Dict, Any

from transformers import HfArgumentParser, PreTrainedModel, PreTrainedTokenizer
from src.model import ModelLoader, get_model_config, ModelConfig
from src.train.trainer import SFTTrainer
from src.dataset import get_sft_dataset
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.train.utils import get_peft_state_dict, get_base_model_state_dict, save_hf_trainer_model

local_rank = None

# Maps model identifiers to their corresponding Liger optimization functions.
# The list is checked in order, so more specific identifiers should come first.
# ["Model Identifier", "Function Name in liger_kernel.transformers"]
LIGER_OPTIMIZATION_MAP: List[Tuple[Union[str, Tuple[str, ...]], str]] = [
    ("llama4", "apply_liger_kernel_to_llama4"),
    ("mistral", "apply_liger_kernel_to_mistral"),
    ("gemma3", "apply_liger_kernel_to_gemma3_text"),
    ("qwen3", "apply_liger_kernel_to_qwen3"),
    ("glm4", "apply_liger_kernel_to_glm4"),
]

def print_rank0(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def apply_liger_optimizations(model_config: ModelConfig, use_liger: bool) -> None:
    """
    Dynamically applies Liger kernel optimizations for supported models.

    This function compares the model name against LIGER_OPTIMIZATION_MAP and
    applies the corresponding kernel patch if a match is found.
    To support a new model, simply update the `LIGER_OPTIMIZATION_MAP`.
    """
    if not use_liger:
        return

    model_name_lower = model_config.model_name.lower()
    optimization_to_apply = None
    
    # Iterate through the mapping table to find the appropriate optimization function.
    for identifier, function_name in LIGER_OPTIMIZATION_MAP:
        is_match = False
        if isinstance(identifier, tuple):
            # If the identifier is a tuple, all keywords must be present.
            if all(keyword in model_name_lower for keyword in identifier):
                is_match = True
        elif isinstance(identifier, str):
            # If the identifier is a string, check for its presence.
            if identifier in model_name_lower:
                is_match = True
        
        if is_match:
            optimization_to_apply = (identifier, function_name)
            break

    if not optimization_to_apply:
        print_rank0(f"Liger optimizations are not available for model '{model_config.model_name}'.")
        return

    applied_for, function_name = optimization_to_apply
    try:
        # Dynamically import the 'liger_kernel.transformers' module.
        liger_transformers = importlib.import_module("liger_kernel.transformers")
        # Find the required optimization function by name within the module.
        optimization_func = getattr(liger_transformers, function_name)

        # Call the optimization function (assuming all have the same signature).
        optimization_func(fused_linear_cross_entropy=False)
        print_rank0(f"Successfully applied Liger kernel for '{applied_for}' model.")

    except ImportError:
        print_rank0("liger_kernel not found. Skipping optimizations.")
    except AttributeError:
        print_rank0(f"Optimization function '{function_name}' not found in liger_kernel.")
    except Exception as e:
        print_rank0(f"An error occurred while applying Liger optimizations: {e}")

def configure_training_args(training_args: TrainingArguments, model_config: ModelConfig) -> None:
    """Configure and validate training arguments."""
    
    # LoRA validation
    if training_args.use_lora:
        if not training_args.freeze_llm:
            raise ValueError("If `use_lora` is True, `freeze_llm` must also be True.")
        
        # Set up LoRA exclusions
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []
    
    # Set default sequence length
    if training_args.max_seq_length is None:
        training_args.max_seq_length = model_config.default_max_length
        print_rank0(f"Set max_seq_length to {model_config.default_max_length} from model config")
        
def create_data_module(model_config: ModelConfig, processor: PreTrainedTokenizer, data_args: DataArguments) -> Dict[str, Any]:
    """Create data module based on model type."""
    
    return get_sft_dataset(
        model_id=model_config.model_name,
        processor=processor,
        data_args=data_args
    )

def parse_arguments() -> Tuple[ModelArguments, DataArguments, TrainingArguments]:
    """Parse command-line arguments for model, data, and training."""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args

def get_and_print_model_config(model_args: ModelArguments) -> ModelConfig:
    """Get model configuration based on arguments and print it."""
    if model_args.model_config_override:
        from src.models.model_config import MODEL_CONFIGS
        if model_args.model_config_override not in MODEL_CONFIGS:
            raise ValueError(f"Model config override '{model_args.model_config_override}' not found")
        model_config = MODEL_CONFIGS[model_args.model_config_override]
        model_config.model_name = model_args.model_id
    else:
        model_config = get_model_config(model_args.model_id)
    
    print_rank0(f"Using model configuration: {model_config.model_name}")
    print_rank0(f"Architecture: {model_config.architecture}")
    return model_config

def load_model_and_tokenizer(
    model_config: ModelConfig, 
    training_args: TrainingArguments
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizer]:
    """Load model, tokenizer, and processor."""
    model_loader = ModelLoader(model_config, training_args)
    model, tokenizer, processor = model_loader.load_model_and_tokenizer()
    
    print_rank0(f"Loaded model: {model.__class__.__name__}")
    print_rank0(f"Model device: {next(model.parameters()).device}")
    print_rank0(f"Model dtype: {next(model.parameters()).dtype}")
    return model, tokenizer, processor

def start_training_or_resume(trainer: SFTTrainer, training_args: TrainingArguments) -> None:
    """Run training, resuming from checkpoint if available."""
    checkpoint_dir = pathlib.Path(training_args.output_dir)
    if list(checkpoint_dir.glob("checkpoint-*")):
        print_rank0("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

def finalize_and_save_model(trainer: SFTTrainer, model: PreTrainedModel, tokenizer_or_processor: PreTrainedTokenizer, training_args: TrainingArguments) -> None:
    """Save the final trained model."""
    model.config.use_cache = True
    
    if training_args.use_lora:
        lora_state_dict = get_peft_state_dict(
            model.named_parameters(), training_args.lora_namespan_exclude
        )
        base_model_weights = get_base_model_state_dict(
            model.named_parameters(), require_grad_only=False
        )
        
        if training_args.local_rank in [0, -1]:
            output_dir = training_args.output_dir
            model.config.save_pretrained(output_dir)
            model.save_pretrained(output_dir, state_dict=lora_state_dict)
            tokenizer_or_processor.save_pretrained(output_dir)
            torch.save(base_model_weights, os.path.join(output_dir, "base_model_state_dict.bin"))
    else:
        save_hf_trainer_model(trainer, output_dir=training_args.output_dir)

def train() -> None:
    """Main training function."""
    global local_rank
    
    model_args, data_args, training_args = parse_arguments()
    local_rank = training_args.local_rank
    
    model_config = get_and_print_model_config(model_args)
    configure_training_args(training_args, model_config)
    
    use_liger = getattr(training_args, 'use_liger', False)
    apply_liger_optimizations(model_config, use_liger)
    
    model, tokenizer, processor = load_model_and_tokenizer(model_config, training_args)
    data_module = create_data_module(model_config, processor, data_args)

    tokenizer_or_processor = processor if processor is not None else tokenizer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer_or_processor,
        args=training_args,
        **data_module
    )

    start_training_or_resume(trainer, training_args)
    finalize_and_save_model(trainer, model, tokenizer_or_processor, training_args)
    
    print_rank0("✅ Training successfully completed !")

if __name__ == "__main__":
    train()