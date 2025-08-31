import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from dotenv import load_dotenv
from huggingface_hub import login

def main():
    # --- 1. Load Config ---
    print("Loading environment variables...")
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        print("FATAL: HF_TOKEN not found in .env")
        return

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_path = "dataset.jsonl"
    new_model_name = "llama-3-8b-instruct-presidency-gpt"

    # --- 2. Hugging Face Login ---
    print("Logging into Hugging Face...")
    login(token=HF_TOKEN)
    print("Successfully logged in.")

    # --- 3. Load Base Model (4-bit for memory efficiency) ---
    print("Loading base LLaMA 3 model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # 3090 compatible
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=HF_TOKEN
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Base model loaded.")

    # --- 4. Load Dataset ---
    print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL: {dataset_path} not found.")
        return

    # --- 5. Configure LoRA ---
    print("Setting up LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print("LoRA configuration complete.")

    # --- 6. Training Arguments ---
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # small to fit in 3090 VRAM
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
    )

    # --- 7. Initialize Trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args
    )

    # --- 8. Train ---
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning completed!")

    # --- 9. Save Adapter ---
    print(f"Saving LoRA adapter to {new_model_name}...")
    trainer.save_model(new_model_name)
    print("Adapter saved successfully.")

if __name__ == "__main__":
    main()
