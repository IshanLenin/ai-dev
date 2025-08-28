import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from dotenv import load_dotenv
from huggingface_hub import login

def main():
    """
    Main function to run the Llama 3 fine-tuning pipeline.
    """
    # --- 1. Load Configuration and Secrets ---
    print("Loading configuration...")
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    model_id = "meta-llama/Meta-Llama-3-8B"
    dataset_path = "dataset.jsonl"
    new_model_name = "llama-3-8b-presidency-gpt"

    if not HF_TOKEN:
        print("FATAL: HF_TOKEN not found in .env file. Please create a .env file with your Hugging Face token.")
        return

    # --- 2. Log in to Hugging Face ---
    print("Logging in to Hugging Face...")
    login(token=HF_TOKEN)
    print("Successfully logged in.")

    # --- 3. Configure and Load Model (QLoRA) ---
    print("Configuring and loading the base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Base model and tokenizer loaded successfully.")

    # --- 4. Load Custom Dataset ---
    try:
        print(f"Loading custom dataset from {dataset_path}...")
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        print("Custom dataset loaded successfully.")
        print(f"First example: {dataset[0]}")
    except FileNotFoundError:
        print(f"FATAL: '{dataset_path}' not found. Please ensure your dataset is in the same directory.")
        return

    # --- 5. Configure PEFT with LoRA ---
    print("Configuring LoRA adapters for training...")
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print("LoRA adapters configured.")

    # --- 6. Set Up and Run Training ---
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("\nStarting the fine-tuning process... This will take some time.")
    trainer.train()
    print("Fine-tuning complete!")

    # --- 7. Save the Fine-Tuned Model Adapter ---
    print(f"Saving model adapter to '{new_model_name}'...")
    trainer.save_model(new_model_name)
    print("Model adapter saved successfully.")

if __name__ == "__main__":
    main()
