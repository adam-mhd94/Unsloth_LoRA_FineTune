"""
Fine-tuning script for language models using Unsloth and Hugging Face libraries,
featuring LoRA adaptation, mixed precision training, and optimized resource management.
"""
# pip install unsloth
# Also get the latest nightly Unsloth!
# pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

import argparse
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from typing import Tuple, Dict, Any
import os
import logging


# Configure logging
def setup_logging(log_level: str = "INFO") -> None:
    """Initialize logging configuration."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized successfully")


def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments

    Raises:
        ValueError: If invalid parameter values are provided
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model using Unsloth with LoRA adaptation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_path_name",
        type=str,
        required=True,
        help="Path to the pre-trained model directory or Hugging Face hub identifier",
    )
    model_group.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for model input",
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16" if is_bfloat16_supported() else "float16",
        help="Data type for model weights",
    )
    model_group.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Enable 4-bit quantization for memory-efficient training",
    )

    # LoRA configuration
    lora_group = parser.add_argument_group("LoRA Configuration")
    lora_group.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="Rank of the LoRA decomposition matrices",
    )
    lora_group.add_argument(
        "--lora_alpha",
        type=float,
        default=16.0,
        help="Scaling factor for LoRA weights",
    )

    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training dataset (JSON format)",
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device training batch size",
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of steps for gradient accumulation",
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate for optimizer",
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    train_group.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (overrides epochs)",
    )
    train_group.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warmup steps for learning rate scheduler",
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for training logs and checkpoints",
    )
    output_group.add_argument(
        "--save_model_path",
        type=str,
        default="lora_model",
        help="Path to save the trained LoRA adapter",
    )

    args = parser.parse_args()
    validate_arguments(args)
    return args


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""

    if not os.path.isfile(args.data_path):
        raise ValueError(f"Data file {args.data_path} not found")

    if args.lora_r <= 0:
        raise ValueError("LoRA rank must be a positive integer")

    if args.lora_alpha <= 0:
        raise ValueError("LoRA alpha must be a positive float")


def load_model(
    model_path_name: str,
    max_seq_length: int,
    dtype: str,
    load_in_4bit: bool,
    lora_r: int,
    lora_alpha: float,
) -> Tuple[Any, Any]:
    """
    Load and configure the base model with LoRA adaptation.

    Args:
        model_path_name: Path or name of the pre-trained model
        max_seq_length: Maximum input sequence length
        dtype: Data type for model weights
        load_in_4bit: Enable 4-bit quantization
        lora_r: LoRA matrix rank
        lora_alpha: LoRA scaling factor

    Returns:
        Tuple: (model, tokenizer) pair
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading model from %s", model_path_name)

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path_name,
        max_seq_length=max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    logger.info("Model loaded successfully with LoRA configuration")
    return model, tokenizer


def prepare_dataset(data_path: str, tokenizer: Any) -> Any:
    """
    Load and preprocess training data.

    Args:
        data_path: Path to training data file
        tokenizer: Model tokenizer for text processing

    Returns:
        Dataset: Preprocessed training dataset
    """

    logger = logging.getLogger(__name__)
    logger.info("Loading dataset from %s", data_path)

    def formatting_prompts_func(examples: Dict[str, Any]) -> Dict[str, Any]:
        """Format examples into instructional prompts."""
        prompt_template = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}" 
        )
        ### For DeepSeek Finetuning (reasoning dataset)
        # prompt_template = (
        #     "Below is an instruction that describes a task, paired with an input that provides further context. "
        #     "Write a response that appropriately completes the request.\n\n"
        #     "### Instruction:\n{instruction}\n\n"
        #     "### Input:\n{input}\n\n"
        #     "### Response:\n <think>\n{think}\n</think>\n{output}"
        # )

        texts = [
            prompt_template.format(
                instruction=inst,
                input=inp,
                output=out,
            ) + tokenizer.eos_token
            for inst, inp, out in zip(
                examples["instruction"],
                examples["input"],
                examples["output"]
            )
        ]
        return {"text": texts, }
    pass

    try:
        dataset = load_dataset("json", data_files=data_path, split="train")
        dataset = dataset.map(
            formatting_prompts_func,
            batched=True,
            batch_size=32,
            num_proc=os.cpu_count() // 2,
            load_from_cache_file=False
        )
        logger.info("Dataset prepared successfully (samples: %d)", len(dataset))
        return dataset
    except Exception as e:
        logger.error("Failed to load dataset: %s", str(e))
        raise


def setup_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    """
    Configure training parameters.

    Args:
        args: Parsed command-line arguments

    Returns:
        TrainingArguments: Configured training parameters
    """
    return TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=args.output_dir,
        report_to="none",
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
    )


def main() -> None:
    """Main training workflow."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        args = parse_args()
        model, tokenizer = load_model(
            args.model_path_name,
            args.max_seq_length,
            args.dtype,
            args.load_in_4bit,
            args.lora_r,
            args.lora_alpha,
        )

        dataset = prepare_dataset(args.data_path, tokenizer)
        if args.max_steps is None:
            args.max_steps = len(
                dataset) // (args.batch_size * args.gradient_accumulation_steps)
        training_args = setup_training_arguments(args)

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            dataset_num_proc=os.cpu_count() // 2,
            packing=False,
            args=training_args,
        )

        # Start training
        logger.info("Starting training with %d samples", len(dataset))
        trainer.train()
        logger.info("Training completed successfully")

        # Save results
        # This ONLY saves the LoRA adapters, and not the full model.
        logger.info("Saving model to %s", args.save_model_path)
        model.save_pretrained(args.save_model_path)
        tokenizer.save_pretrained(args.save_model_path)

        # Merge to 16bit
        # model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)

        # Merge to 4bit
        # model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)

        # To finetune and auto export to Ollama
        # Save to 8bit Q8_0
        # model.save_pretrained_gguf("model", tokenizer,)

        # Save to 16bit GGUF
        # model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")

        # Save to q4_k_m GGUF
        # model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")

        logger.info("Model saved successfully")

    except Exception as e:
        logger.error("Training failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()

# python Unsloth_FineTune.py \
#     --model_path_name "/path/to/model or model name" \
#     --data_path "/path/to/dataset.json" \
#     --max_seq_length 2048 \
#     --dtype float16 \
#     --load_in_4bit \
#     --lora_r 16 \
#     --lora_alpha 16 \
#     --batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 2e-4 \
#     --epochs 1 \
#     --output_dir "outputs" \
#     --save_model_path "lora_model"
