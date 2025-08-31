import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from huggingface_hub import login
import os

def main():
    """
    Main function to load the fine-tuned model and start a conversation.
    """
    # --- 1. Load Configuration and Secrets ---
    print("Loading configuration...")
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    # --- ENSURE THIS MATCHES YOUR TRAINING SCRIPT ---
    base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # --- THIS MUST BE THE NAME YOU SAVED YOUR ADAPTER AS ---
    adapter_path = "./llama-3-8b-instruct-presidency-gpt" 

    if not HF_TOKEN:
        print("FATAL: HF_TOKEN not found. Please check your .env file.")
        return

    # --- 2. Log in to Hugging Face ---
    print("Logging in to Hugging Face...")
    login(token=HF_TOKEN)
    print("Successfully logged in.")

    # --- 3. Load the Base Model (using 4-bit for efficiency) ---
    print("Loading the base instruction-tuned model with 4-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=HF_TOKEN)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Base model loaded.")

    # --- 4. Load and Apply the Fine-Tuned Adapter ---
    print(f"Loading fine-tuned adapter from {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
        print("Adapter loaded and applied successfully. Your PresidencyGPT is ready!")
    except Exception as e:
        print(f"FATAL: Could not load the adapter from {adapter_path}. Did the training complete successfully?")
        print(f"Error: {e}")
        return

    # --- 5. Have a Conversation! ---
    print("\n--- Starting conversation with PresidencyGPT ---")
    print("This model is now an expert on your class notes.")

    while True:
        user_question = input("\nAsk a question about your notes (or type 'quit' to exit):\n> ")
        if user_question.lower() == 'quit':
            break

        messages = [
            {"role": "user", "content": user_question},
        ]
        
        # apply_chat_template is the correct way for instruct models
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Generate the response
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1, # Lower temperature for more factual answers
            top_p=0.9,
        )
        
        response = outputs[0][input_ids.shape[-1]:]
        print("\nPresidencyGPT:")
        print(tokenizer.decode(response, skip_special_tokens=True))

if __name__ == "__main__":
    main()

