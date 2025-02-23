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

### Installing Dependencies

To install the required dependencies, run the following commands:

pip install unsloth
Also get the latest nightly Unsloth!
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

For installing the correct version of **Unsloth**, please refer to its [repository](https://github.com/unslothai/unsloth).




