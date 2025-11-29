from datasets import load_dataset, concatenate_datasets


# Function for normalizing datasets in order to combine them into one
def normalize(ds):
    rename_map = {}

    if "claim" in ds.column_names:
        rename_map["claim"] = "question"

    if "policy" in ds.column_names:
        rename_map["policy"] = "text"

    if rename_map:
        ds = ds.rename_columns(rename_map)

    # Ensure required fields exist
    for col in ["question", "text", "answer"]:
        if col not in ds.column_names:
            ds = ds.add_column(col, [""] * len(ds))

    return ds.select_columns(["question", "text", "answer"])


# Function for preprocessing dataset examples by formating each sample in a prompt
def preprocess(example, tokenizer):
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


# Function for tokenizing dataset examples
def tokenize(examples, tokenizer):
    tokens = tokenizer(
        examples["text"], truncation=True, max_length=1024, padding=False
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def create_dataset(tokenizer):
    # Load datasets: contract_qa, insurance_policy_interpretation and privacy_policy_qa
    contract = load_dataset("nguha/legalbench", "contract_qa")
    insurance = load_dataset("nguha/legalbench", "insurance_policy_interpretation")
    privacy = load_dataset("nguha/legalbench", "privacy_policy_qa")

    # Apply normalization to datasets and concatenate them
    train_dataset = concatenate_datasets(
        [
            normalize(contract["train"]),
            normalize(insurance["train"]),
            normalize(privacy["train"]),
        ]
    ).shuffle(seed=42)

    test_dataset = concatenate_datasets(
        [
            normalize(contract["test"]),
            normalize(insurance["test"]),
            normalize(privacy["test"]),
        ]
    ).shuffle(seed=42)

    # Preprocess dataset
    train_dataset = train_dataset.map(lambda x: preprocess(x, tokenizer))
    test_dataset = test_dataset.map(lambda x: preprocess(x, tokenizer))

    # Tokenize dataset
    train_dataset = train_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

    return train_dataset, test_dataset
