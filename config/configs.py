from peft import LoraConfig
from transformers import TrainingArguments

API_KEY = "secret123"
API_URL = "https://barbituric-inhomogeneous-marisol.ngrok-free.dev/generate"
API_HEADERS = {"Authorization": "Bearer secret123"}
NGROK_TOKEN = "35xpeJxb2oIm1GxULp8VMjZnqeN_6t1nU7D67QQQqrSqPwXQY"
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

TRAINING_ARGS = TrainingArguments(
    output_dir="legal-qwen25-7b",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=5,
    save_steps=200,
    fp16=True,
    report_to="none",
)