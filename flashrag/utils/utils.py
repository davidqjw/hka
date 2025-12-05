import os
import json
import importlib
def get_generator(config, **params):
    """Automatically select generator class based on config."""

    if config["framework"] == "hf":
            return getattr(importlib.import_module("flashrag.generator"), "HFCausalLMGenerator")(config, **params)
    else:
        raise NotImplementedError

def get_retriever(config):
    return getattr(importlib.import_module("flashrag.retriever"), "DenseRetriever")(config)

def get_dataset(config):
    """Load dataset from config."""
    SUPPORT_FILES = ["jsonl", "json", "parquet"]

    dataset_path = config["dataset_path"]
    all_split = config["split"]

    split_dict = {split: None for split in all_split}

    for split in all_split:
        exist_flag = 0
        for file_postfix in SUPPORT_FILES:
            split_path = os.path.join(dataset_path, f"{split}.{file_postfix}")
            if not os.path.exists(split_path):
                continue
            else:
                exist_flag = 1
                break
        if exist_flag == 0:
            continue
        else:
            print(f"Loading {split} dataset from: {split_path}...")
        if split in ["test", "val", "dev"]:
            split_dict[split] = Dataset(
                config, split_path, sample_num=config["test_sample_num"], random_sample=config["random_sample"]
            )
        else:
            split_dict[split] = Dataset(config, split_path)

    return split_dict