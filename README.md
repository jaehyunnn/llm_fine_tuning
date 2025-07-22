# LLM Fine-Tuning

A comprehensive framework for fine-tuning of Large Language Models with support for distributed training, quantization, and parameter-efficient fine-tuning techniques.

## Features

- **Full Fine-Tuning**: Complete model parameter optimization
- **Parameter-Efficient Fine-Tuning**: LoRA and DoRA support
- **Quantization**: 4-bit and 8-bit quantization with BitsAndBytes
- **Distributed Training**: DeepSpeed ZeRO integration (Stage 2/3)
- **Flash Attention**: Optimized attention mechanisms
- **Multiple Architectures**: Support for Llama, Mistral, Gemma, Qwen, and more
- **Performance Optimizations**: Liger kernel optimizations

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Training

1. **Prepare your dataset** in JSON format:
```json
[
  {
    "conversations": [
      {
        "role": "user", 
        "content": "Your question here"
      },
      {
        "role": "assistant",
        "content": "Assistant response"
      }
    ]
  }
]
```

2. **Run training**:
```bash
sh script/train.sh
```

## Configuration

### Training Script (`script/train.sh`)

Key parameters you can modify:

- `MODEL_NAME`: Hugging Face model identifier (e.g., "Qwen/Qwen3-0.6B")
- `JSON_DATA_PATH`: Path to your training dataset
- `GLOBAL_BATCH_SIZE`: Total batch size across all devices
- `EPOCHS`: Number of training epochs
- `OUTPUT_DIR`: Directory to save checkpoints and final model

### DeepSpeed Configuration

- `script/zero2.json`: ZeRO Stage 2 configuration
- `script/zero3.json`: ZeRO Stage 3 configuration (current default)

### Fine-Tuning Options

**Full Fine-Tuning** (default):
```bash
--use_lora False
--freeze_llm False
```

**LoRA Fine-Tuning**:
```bash
--use_lora True
--freeze_llm True
--lora_rank 64
--lora_alpha 16
```

**Quantized Training**:
```bash
--bits 4  # or 8
--double_quant True
```

## Supported Models

- **Llama**: Llama 2, Llama 3, Code Llama
- **Mistral**: Mistral 7B, Mixtral
- **Gemma**: Gemma 2B, 7B
- **Qwen**: Qwen2.5, Qwen3
- **Phi**: Phi-3 series
- And more...

## Performance Features

### Flash Attention
Automatically enabled for supported models to reduce memory usage and increase speed.

### Liger Kernel Optimizations
```bash
--use_liger True
```

### Gradient Checkpointing
```bash
--gradient_checkpointing True
```

## Output Structure

After training, you'll find:

```
output/fft_YYYYMMDD_HHMMSS/
├── config.json              # Model configuration
├── pytorch_model.bin         # Model weights
├── tokenizer.json           # Tokenizer files
├── training_args.bin        # Training arguments
├── trainer_state.json       # Training state
└── checkpoint-*/            # Intermediate checkpoints
```

## Memory Requirements

### Estimated VRAM Usage (Qwen 0.6B):

- **Full Fine-Tuning (bf16)**: ~8-12GB
- **LoRA (4-bit)**: ~4-6GB
- **ZeRO Stage 3**: Scales across multiple GPUs

### For Larger Models:

- Use ZeRO Stage 3 with multiple GPUs
- Enable gradient checkpointing
- Use 4-bit quantization
- Reduce batch size

## Advanced Usage

### Custom Model Configuration

Override model settings by specifying `model_config_override` in your training arguments.

### Multiple GPU Training

```bash
# Modify in train.sh
NUM_DEVICES=4
BATCH_PER_DEVICE=2
```

### Resuming Training

The framework automatically resumes from the latest checkpoint if found in the output directory.

### Monitoring

Training metrics are logged to TensorBoard:
```bash
tensorboard --logdir output/
```

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce batch size, enable gradient checkpointing, or use quantization
2. **DeepSpeed errors**: Check DeepSpeed configuration and ensure proper environment setup
3. **Model loading issues**: Verify model name and internet connection

### Environment Setup

For optimal performance:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{llm_fine_tuning,
  title={LLM Fine-Tuning},
  year={2024},
  url={https://github.com/jaehyunnn/llm_fine_tuning}
}
```