import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
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
        print("FATAL: HF_TOKEN not found. Please check your .env file.")
        return

    # --- 2. Log in to Hugging Face ---
    print("Logging in to Hugging Face...")
    login(token=HF_TOKEN)
    print("Successfully logged in.")

    # --- 3. Load Model in 16-bit Precision ---
    print("Loading base model and tokenizer in 16-bit precision...")
    
    # We now load the model in bfloat16, which is very efficient on modern GPUs.
    # This completely avoids the need for the bitsandbytes library.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, # Use 16-bit precision
        device_map="auto",
        token=HF_TOKEN
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Base model loaded successfully.")

    # --- 4. Load Custom Dataset ---
    try:
        print(f"Loading custom dataset from {dataset_path}...")
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        print("Custom dataset loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL: '{dataset_path}' not found.")
        return

    # --- 5. Configure PEFT with LoRA ---
    print("Configuring LoRA adapters for training...")
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

    print("\nStarting the fine-tuning process...")
    trainer.train()
    print("Fine-tuning complete!")

    # --- 7. Save the Fine-Tuned Model Adapter ---
    print(f"Saving model adapter to '{new_model_name}'...")
    trainer.save_model(new_model_name)
    print("Model adapter saved successfully.")

if __name__ == "__main__":
    main()