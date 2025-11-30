from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model
import shutil

from config.configs import *
from src.dataset import *

def main():
    
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    #Load tokenizer + fix special tokens
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.eos_token

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

    #Apply Lora configs to model
    model = get_peft_model(model, LORA_CONFIG)

    #Create dataset 
    train_dataset, test_dataset = create_dataset(tokenizer)
    
    #Define trainer
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    #Train
    trainer.train()

    #Remove training configs
    shutil.rmtree(TRAINING_ARGS.output_dir)
    
    #Save fully merged model (base + lora parameters)
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained("full_model")
    tokenizer.save_pretrained("full_model")

if __name__ == "__main__" :
    main()
