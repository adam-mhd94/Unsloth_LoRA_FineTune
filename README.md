# LLM Fine-Tuning with Unsloth 🚀
This repository contains the implementation and fine-tuning of large language models using Unsloth and LoRA (Low-Rank Adaptation).

## Unsloth

### Key Features ✨

- **2 to 5 times faster training** with Unsloth optimizations.
- **70% memory reduction** (supports 4-bit QLoRA).
- **Full integration** with Hugging Face libraries.
- Support for **LLaMA, Mistral, Phi architectures**, and custom models.
- Capability to train on **resource-limited hardware** (consumer GPUs).


### Installation & Setup 💻

#### Prerequisites

- Python 3.8 or higher.
- **NVIDIA GPU** with CUDA 11.8+ support.
- At least **16GB VRAM** for 7B parameter models.

#### Installing Dependencies

To install the required dependencies, run the following command:

```bash
pip install unsloth
```

If you want to install the latest nightly build of Unsloth, use this command:

```bash
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```
For installing the correct version of **Unsloth**, please refer to its [repository](https://github.com/unslothai/unsloth).

### ⚙️Fine-tuning DeepSeek R1

Fine-tuning reasoning models is still an emerging field. However, since DeepSeek's distilled models are based on Llama and Qwen architectures, they are fully compatible with Unsloth right away.

Simply update the model names to the correct ones. For example, replace 'unsloth/Meta-Llama-3.1-8B' with 'unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit.

## How to Run 🚀

### Basic Command
```bash
python Unsloth_FineTune.py \
    --model_path_name "/path/to/model" \
    --data_path "/path/to/dataset.json" \
    --output_dir "outputs" \
    --save_model_path "lora_model"
```
### Full Example with Advanced Parameters
```bash
python Unsloth_FineTune.py \
    --model_path_name "models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B" \
    --data_path "data/finetune_data.json" \
    --max_seq_length 2048 \
    --dtype float16 \
    --load_in_4bit \
    --lora_r 16 \
    --lora_alpha 16 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --epochs 1 \
    --output_dir "outputs" \
    --save_model_path "lora_model"
```

### Parameter Descriptions ⚙️
Parameter	Default Value	Description
### Hyperparameters

| Parameter                         | Description                                                       | Default Value             |
| ---------------------------------- | ----------------------------------------------------------------- | ------------------------- |
| `--model_path_name`                | Path to the model from Hugging Face Hub or local directory        | Required                  |
| `--data_path`                      | Path to the dataset JSON file                                     | Required                  |
| `--max_seq_length`                 | Maximum input sequence length (adjust based on the model)         | 2048                      |
| `--dtype`                          | Computation precision (float16, bfloat16, float32)                | bfloat16                  |
| `--load_in_4bit`                   | Enable 4-bit quantization for memory efficiency                    | Disabled                  |
| `--lora_r`                         | LoRA matrix rank (recommended: 8 to 64)                           | 8                         |
| `--lora_alpha`                     | LoRA scaling factor (usually 2x `lora_r`)                         | 16                        |
| `--batch_size`                     | Training batch size per GPU (adjust based on VRAM)                | 2                         |
| `--gradient_accumulation_steps`    | Number of gradient accumulation steps                             | 4                         |
| `--learning_rate`                  | Initial learning rate (for LoRA: 1e-4 to 3e-4)                    | 2e-4                      |
| `--epochs`                         | Number of training epochs                                          | 1                         |
| `--output_dir`                     | Directory to save logs and checkpoints                             | "outputs"                 |
| `--save_model_path`                | Final path to save the trained model                               | "lora_model"              |




