import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from huggingface_hub import login
import os

def main():
    print("Loading configuration...")
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter_path = "./llama-3-8b-instruct-presidency-gpt"

    if not HF_TOKEN:
        print("FATAL: HF_TOKEN not found. Please check your .env file.")
        return

    print("Logging in to Hugging Face...")
    login(token=HF_TOKEN)
    print("Successfully logged in.")

    print("Loading the base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=HF_TOKEN
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_auth_token=HF_TOKEN)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print("Adapter loaded. PresidencyGPT is ready!")

    print("\n--- Start conversation ---")
    while True:
        user_question = input("\nAsk a question (or type 'quit'):\n> ")
        if user_question.lower() == "quit":
            break

        # Encode input and generate
        input_ids = tokenizer(user_question, return_tensors="pt").input_ids.to(model.device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )

        # Skip input tokens in output
        response_ids = output_ids[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        print("\nPresidencyGPT:")
        print(response)

if __name__ == "__main__":
    main()
