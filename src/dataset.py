from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import PreTrainedTokenizerBase
from typing import Dict, List, Tuple

DATASET_CONFIGS = [
    ("nguha/legalbench", "contract_qa"),
    ("nguha/legalbench", "insurance_policy_interpretation"),
    ("nguha/legalbench", "privacy_policy_qa"),
]

REQUIRED_COLUMNS = {"question", "text", "answer"}


def _normalize_schema(dataset: Dataset) -> Dataset:
    """
    Normalize dataset columns schema to match with other datasets

    Args:
        dataset(Dataset): dataset object to normalize

    Returns:
        normalized_dataset(Dataset): normalized dataset with the required columns only
    """
    rename_map = {}

    if "claim" in dataset.column_names:
        rename_map["claim"] = "question"

    if "policy" in dataset.column_names:
        rename_map["policy"] = "text"

    if rename_map:
        dataset = dataset.rename_columns(rename_map)

    for col in REQUIRED_COLUMNS:
        if col not in dataset.column_names:
            dataset = dataset.add_column(col, [""] * len(dataset))

    normalized_dataset = dataset.select_columns(sorted(REQUIRED_COLUMNS))

    return normalized_dataset


def _load_datasets(split: str) -> Dataset:
    """
    load datasets and contatenate them

    Args:
        - split(str): the name of the dataset split desired to obtain

    Returns:
        - final_dataset(Dataset): final dataset
    """

    datasets = []
    for path, name in DATASET_CONFIGS:
        dataset = load_dataset(path, name)
        normalized_dataset = _normalize_schema(dataset[split])
        datasets.append(normalized_dataset)

    final_dataset = concatenate_datasets(datasets).shuffle(seed=42)
    return final_dataset


def _format_example(example: Dict, eos_token: str) -> Dict:
    """
    Function for formating each example in the dataset

    Args:
        example(dict): a single example in a dataset
        eos_token(str): end of sentence token that used tokenizer uses

    Returns:
        formatted_example(dict): example after formatting into a prompt
    """
    answer = example["answer"]
    if isinstance(answer, list):
        answer = answer[0] if len(answer) > 0 else ""

    prompt = (
        f"User question: {example['question']}\n"
        f"Legal text: {example['text']}\n"
        f"Answer: "
    )

    formatted_example = {"text": prompt + answer + eos_token}

    return formatted_example


def _tokenize(examples: Dict, tokenizer: PreTrainedTokenizerBase) -> Dict:
    """
    Tokenize each example in dataset for training

    Args:
        - examples(dict): a batch of examples from dataset
        - tokenizer(PreTrainedTokenizerBase): used tokenizer

    Returns:
        - tokens(dict): tokenized examples
    """
    tokens = tokenizer(
        examples["text"], truncation=True, max_length=1024, padding=False
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def create_dataset(tokenizer: PreTrainedTokenizerBase) -> Tuple:
    """
    Create full dataset using sub datasets

    Args:
        - tokenizer(PreTrainedTokenizerBase): used tokenizer for tokens creation

    Returns:
        - dataset(tuple): final dataset after normalization, concatenation and tokenization

    """

    # Apply normalization to datasets and concatenate them
    train_dataset = _load_datasets("train")
    test_dataset = _load_datasets("test")

    # Preprocess dataset
    train_dataset = train_dataset.map(lambda x: _format_example(x, tokenizer.eos_token))
    test_dataset = test_dataset.map(lambda x: _format_example(x, tokenizer.eos_token))

    # Tokenize dataset
    train_dataset = train_dataset.map(lambda x: _tokenize(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: _tokenize(x, tokenizer), batched=True)

    dataset = (train_dataset, test_dataset)
    return dataset
