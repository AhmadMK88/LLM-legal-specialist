from config.configs import TRAINING_ARGS, LORA_CONFIG, MODEL_NAME
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from pathlib import Path
from peft import get_peft_model, PeftModel
import shutil
from src.dataset import create_dataset

def _load_model() -> PeftModel:
    """
    loads used model with applying lora configs
    
    Returns:
        - model(PeftModel): model with lora configs
    """

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map="auto"
    )

    model = get_peft_model(model, LORA_CONFIG)
    return model

def _load_tokenizer() -> AutoTokenizer:
    """
    loads used model tokenizer
    
    Returns:
        - tokenizer(AutoTokenizer) : ready to use
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def _train(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    test_dataset: Dataset
) -> Trainer:
    """
    Create a trainer object using model, tokenizer and dataset
    Args:
        - model(PeftModel): used model for text generation
        - tokenizer(AutoTokenizer): tokenizer from the used model
        - train_dataset(Dataset): training dataset
        - test_dataset(Dataset): testing dataset
    
    Returns:
        - trainer(Trainer): trainer object 
    """

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    return trainer

def main():
    
    #Load tokenizer + fix special tokens
    tokenizer = _load_tokenizer()

    #Load Qwen in 4-bit + LoRA
    model = _load_model()

    #Create dataset 
    train_dataset, test_dataset = create_dataset(tokenizer)
    
    #Train 
    trainer = _train(model, tokenizer, train_dataset, test_dataset)

    #Remove training configs 
    output_dir = Path(TRAINING_ARGS.output_dir) 
    if output_dir.exists(): 
        shutil.rmtree(output_dir) 
    
    #Save fully merged model (base + lora parameters) 
    merged_model = trainer.model.merge_and_unload() 
    merged_model.save_pretrained("full_model") 
    tokenizer.save_pretrained("full_model")

if __name__ == "__main__" :
    main()
