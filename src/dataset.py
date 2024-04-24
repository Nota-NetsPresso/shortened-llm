"""
Code modified from
https://github.com/IST-DASLab/gptq/blob/main/datautils.py
https://github.com/horseee/LLM-Pruner/blob/main/LLMPruner/datasets/example_samples.py
https://github.com/horseee/LLM-Pruner/blob/main/LLMPruner/datasets/ppl_dataset.py
"""

import random

import torch
from datasets import load_dataset
from torch.utils.data.dataset import Dataset


class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


def get_wikitext2():
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return traindata, testdata


def get_ptb():
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")
    return traindata, valdata


def process_data(samples, tokenizer, seq_len, field_name, add_bos_to_every=False):
    test_ids = tokenizer(
        "\n\n".join(samples[field_name]),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    if not add_bos_to_every:  # add bos token to only the first segment
        test_ids = torch.cat(
            (torch.LongTensor([tokenizer.bos_token_id]), test_ids), dim=0
        )

    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len) : ((i + 1) * seq_len)]
        if add_bos_to_every:  # add bos token to every segment (especially for gemma)
            batch = torch.cat(
                (torch.LongTensor([tokenizer.bos_token_id]), batch), dim=0
            )
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)

    return IndexDataset(tensors=test_ids_batch)


def get_loaders(name, tokenizer, seq_len=2048, batch_size=8, add_bos_to_every=False):
    if "wikitext2" in name:
        train_data, test_data = get_wikitext2()
        test_dataset = process_data(
            test_data, tokenizer, seq_len, "text", add_bos_to_every
        )
    if "ptb" in name:
        train_data, test_data = get_ptb()
        test_dataset = process_data(
            test_data, tokenizer, seq_len, "sentence", add_bos_to_every
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_data, test_loader


def get_c4_examples(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]["text"], return_tensors="pt")
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i : i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_bookcorpus_examples(tokenizer, n_samples, seq_len):
    traindata = load_dataset("bookcorpus", split="train")

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]["text"], return_tensors="pt")
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i : i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_examples(dataset, tokenizer, n_samples, seq_len=128):
    if dataset == "c4":
        return get_c4_examples(tokenizer, n_samples, seq_len)
    elif dataset == "bookcorpus":
        return get_bookcorpus_examples(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError
