import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from dotenv import load_dotenv
from huggingface_hub import login
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio

# --- 1. Load Configuration and Secrets ---
print("Loading configuration...")
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
adapter_path = "./llama-3-8b-instruct-presidency-gpt" 

# --- 2. Initialize FastAPI App ---
app = FastAPI(title="PresidencyGPT API")

# --- 3. Load the Fine-Tuned Model (do this once on startup) ---
print("Loading the fine-tuned model...")
if not HF_TOKEN:
    raise ValueError("FATAL: HF_TOKEN not found. Please check your .env file.")

login(token=HF_TOKEN)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(base_model, adapter_path)
print("✅ PresidencyGPT model loaded successfully!")


# --- 4. WebSocket for Real-Time Chat ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

manager = ConnectionManager()

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print("New client connected.")
    try:
        while True:
            # Wait for a question from the user
            user_question = await websocket.receive_text()
            
            print(f"Received question: {user_question}")
            
            # Prepare the prompt for the model
            messages = [{"role": "user", "content": user_question}]
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
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            
            response_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            print(f"Generated response: {response_text}")

            # Send the response back to the user's browser
            await websocket.send_text(response_text)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected.")


# --- 5. Serve the Frontend ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')
