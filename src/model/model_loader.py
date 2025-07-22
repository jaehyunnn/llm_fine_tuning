import torch
import warnings
from typing import Optional, Union, Dict, Any
from transformers import (
    AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM,
    BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .model_config import ModelConfig, TextOnlyModelConfig


class ModelLoader:
   
    def __init__(self, model_config: ModelConfig, training_args):
        self.model_config = model_config
        self.training_args = training_args
        self.model = None
        self.tokenizer = None
        self.processor = None
        
    def load_model_and_tokenizer(self):
        """Load model, tokenizer, and processor based on model configuration."""
        # Load tokenizer/processor
        self._load_tokenizer_and_processor()
        
        # Prepare quantization config
        bnb_config = self._prepare_quantization_config()
        
        # Load model
        self._load_model(bnb_config)
        
        # Apply custom configurations
        self._apply_model_configurations()
        
        # Apply LoRA if enabled
        if self.training_args.use_lora:
            self._apply_lora()
            
        return self.model, self.tokenizer, self.processor
    
    def _load_tokenizer_and_processor(self):
        """Load tokenizer and processor based on model type."""
       
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize processor with model and tokenizer references
        class Processor:
            def __init__(self):
                self.model = None
                self.tokenizer = None
            
            def save_pretrained(self, save_directory):
                """Save the tokenizer (processor delegates to tokenizer)"""
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(save_directory)
        
        self.processor = Processor()
        self.processor.model = None  # Will be set after model is loaded
        self.processor.tokenizer = self.tokenizer
    
    def _prepare_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Prepare quantization configuration."""
        if self.training_args.bits not in [4, 8]:
            return None
            
        compute_dtype = self._get_compute_dtype()
        skip_modules = self.model_config.quantization_skip_modules or []
        
        return BitsAndBytesConfig(
            load_in_4bit=self.training_args.bits == 4,
            load_in_8bit=self.training_args.bits == 8,
            llm_int8_skip_modules=skip_modules,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.training_args.double_quant,
            bnb_4bit_quant_type=self.training_args.quant_type,
        )
    
    def _load_model(self, bnb_config: Optional[BitsAndBytesConfig]):
        """Load the model with appropriate class and configuration."""
        compute_dtype = self._get_compute_dtype()
        
        model_kwargs = {
            "torch_dtype": compute_dtype,
            "device_map": {"": self.training_args.device} if bnb_config else None,
            "quantization_config": bnb_config,
            "trust_remote_code": True
        }
        
        # Add attention implementation if supported
        if (self.model_config.supports_flash_attention and 
            not self.training_args.disable_flash_attn2):
            model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            model_kwargs["attn_implementation"] = "sdpa"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name, **model_kwargs
        )
        
        # Update processor with loaded model
        if hasattr(self, 'processor') and self.processor is not None:
            self.processor.model = self.model
    
    def _apply_model_configurations(self):
        """Apply model-specific configurations."""
        self.model.config.use_cache = self.model_config.use_cache
        
        # Configure gradient checkpointing
        if (self.training_args.gradient_checkpointing and 
            self.model_config.supports_gradient_checkpointing):
            self.model.enable_input_require_grads()
            self.training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        
        # Prepare for quantized training
        if self.training_args.bits in [4, 8]:
            self.model.config.torch_dtype = self._get_compute_dtype()
            self.model = prepare_model_for_kbit_training(
                self.model, 
                use_gradient_checkpointing=self.training_args.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": True}
            )
        
        self._configure_text_model_components()
    
    def _configure_text_model_components(self):
        """Configure text-only model components."""
        # For text models, freeze_llm controls the entire model
        if self.training_args.freeze_llm:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def _apply_lora(self):
        """Apply LoRA configuration to the model."""
        target_modules = self._get_lora_target_modules()
        
        if not target_modules:
            raise ValueError(f"No LoRA target modules found for {self.model_config.architecture}")
        
        peft_config = LoraConfig(
            r=self.training_args.lora_rank,
            lora_alpha=self.training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.training_args.lora_dropout,
            bias=self.training_args.lora_bias,
            use_dora=getattr(self.training_args, 'use_dora', False)
        )
        
        # Apply precision settings
        if self.training_args.bits == 16:
            if self.training_args.bf16:
                self.model.to(torch.bfloat16)
            elif self.training_args.fp16:
                self.model.to(torch.float16)
        
        self.model = get_peft_model(self.model, peft_config)
    
    def _get_lora_target_modules(self):
        """Get LoRA target modules based on model configuration and training settings."""
        base_modules = self.model_config.lora_target_modules or []
        exclude_modules = self.model_config.lora_exclude_modules or []
        
        # Add exclusions from training args
        if hasattr(self.training_args, 'lora_namespan_exclude'):
            exclude_modules.extend(self.training_args.lora_namespan_exclude or [])
        
        # Find all linear modules if no specific targets provided
        if not base_modules:
            base_modules = self._find_target_linear_names(exclude_modules)
        
        # Filter out excluded modules
        target_modules = [
            module for module in base_modules 
            if not any(ex in module for ex in exclude_modules)
        ]
        
        return target_modules
    
    def _find_target_linear_names(self, exclude_modules):
        """Find all linear layer names in the model."""
        linear_cls = torch.nn.modules.Linear
        embedding_cls = torch.nn.modules.Embedding
        lora_module_names = []
        
        for name, module in self.model.named_modules():
            if any(ex_keyword in name for ex_keyword in exclude_modules):
                continue
            if isinstance(module, (linear_cls, embedding_cls)):
                lora_module_names.append(name)
        
        # Limit modules if specified
        if hasattr(self.training_args, 'num_lora_modules') and self.training_args.num_lora_modules > 0:
            lora_module_names = lora_module_names[-self.training_args.num_lora_modules:]
        
        return lora_module_names
    
    def _get_compute_dtype(self):
        """Get the compute data type from training arguments."""
        if self.training_args.fp16:
            return torch.float16
        elif self.training_args.bf16:
            return torch.bfloat16
        else:
            return torch.float32
    
    @staticmethod
    def _set_requires_grad(parameters, requires_grad: bool):
        """Set requires_grad for a collection of parameters."""
        for param in parameters:
            param.requires_grad = requires_grad