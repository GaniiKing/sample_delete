from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# =========================
# 1. Load base model
# =========================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


print(model_name)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# =========================
# 2. Load your LoRA adapter
# =========================
model = PeftModel.from_pretrained(
    base_model,
    r"C:\Users\mycla\Downloads\ganii\ganii-tiny-lora"   # 👈 your downloaded folder path
)

# =========================
# 3. Chat
# =========================
while True:
    user_input = input("You: ")

    prompt = f"### Instruction:\n{user_input}\n### Response:\n"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7
    )

    print("Ganii:", tokenizer.decode(output[0], skip_special_tokens=True))