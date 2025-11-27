from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import os 

#Function for normalizing datasets in order to combine them into one 
def normalize(ds):
    rename_map = {}

    if "claim" in ds.column_names:
        rename_map["claim"] = "question"

    if "policy" in ds.column_names:
        rename_map["policy"] = "text"

    if rename_map:
        ds = ds.rename_columns(rename_map)

    #Ensure required fields exist
    for col in ["question", "text", "answer"]:
        if col not in ds.column_names:
            ds = ds.add_column(col, [""] * len(ds))

    return ds.select_columns(["question", "text", "answer"])

#Function to preprocess dataset
def preprocess(example):
    answer = example["answer"]
    if isinstance(answer, list):
        answer = answer[0] if len(answer) > 0 else ""

    prompt = (
        f"User question: {example['question']}\n"
        f"Legal text: {example['text']}\n"
        f"Answer: "
    )

    full_text = prompt + answer + tokenizer.eos_token

    return {"text": full_text}

#Function to tokenize dataset examples
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding=False
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

#Load datasets: contract_qa, insurance_policy_interpretation and privacy_policy_qa
contract = load_dataset("nguha/legalbench", "contract_qa")
insurance = load_dataset("nguha/legalbench", "insurance_policy_interpretation")
privacy = load_dataset("nguha/legalbench", "privacy_policy_qa")

#Apply normalization function to datasets and concatenate them
train_dataset = concatenate_datasets([
    normalize(contract["train"]),
    normalize(insurance["train"]),
    normalize(privacy["train"])
]).shuffle(seed=42)

test_dataset = concatenate_datasets([
    normalize(contract["test"]),
    normalize(insurance["test"]),
    normalize(privacy["test"])
]).shuffle(seed=42)

#Load tokenizer + fix special tokens
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.bos_token is None:
    tokenizer.bos_token = tokenizer.eos_token
if tokenizer.eos_token is None:
    tokenizer.eos_token = tokenizer.eos_token

#Preprocess dataset
train_dataset = train_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

#Tokenize dataset
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

#Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

#Load Qwen in 4-bit + LoRA
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto"
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

#Training arguments
training_args = TrainingArguments(
    output_dir=None,               
    save_strategy="no",            
    save_steps=0,
    save_total_limit=0,
    logging_dir=None,              
    logging_strategy="no",       

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    report_to="none",
)

#Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

#Train
trainer.train()

# Save fine-tuned model and tokenizer
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained("fine-tuned-model")
tokenizer.save_pretrained("fine-tuned-model")

os.rmdir("trainer_output")