import numpy as np
import pandas as pd
import torch
from collections import Counter
import random
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')

def clean_tokenize(sentence, start_token="<sos>", end_token="<eos>"):
    sentence = [f"{start_token} {s} {end_token}" for s in sentence]
    sentence = [s.strip() for s in sentence if s.strip()]
    tokenize_sent = [s.split() for s in sentence]
    return tokenize_sent

def build_vocab(tokenized_sent, max_size=1000, min_freq=2):
    counter = Counter([token for sentence in tokenized_sent for token in sentence])
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}

    for token, freq in counter.most_common(max_size):
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    
    return vocab

def sentence_to_indices(sentence, vocab):
    return [vocab.get(token, vocab["<unk>"]) for token in sentence]

def pad_sentence(sentences, pad_token=0):
    max_len = max(len(sentence) for sentence in sentences)
    return [sentence + [pad_token] * (max_len - len(sentence)) for sentence in sentences]

def prepare_data(data_path, batch_size=8, max_vocab_size=1000, min_freq=2):

    language_data = pd.read_csv(data_path)
    HINDI = language_data["hindi"][0:1200]
    ENGLISH = language_data["english"][0:1200]


    HINDI = clean_tokenize(HINDI)
    ENGLISH = clean_tokenize(ENGLISH)


    hindi_vocab = build_vocab(HINDI, max_size=max_vocab_size, min_freq=min_freq)
    english_vocab = build_vocab(ENGLISH, max_size=max_vocab_size, min_freq=min_freq)


    indexed_hindi = [sentence_to_indices(sentence, hindi_vocab) for sentence in HINDI]
    indexed_english = [sentence_to_indices(sentence, english_vocab) for sentence in ENGLISH]

    # Shuffle and split data
    data = list(zip(indexed_hindi, indexed_english))
    random.seed(42)
    random.shuffle(data)
    split_idx = int(0.8 * len(data))

    train_data = data[:split_idx]
    test_data = data[split_idx:]

    train_hindi, train_english = zip(*train_data)
    test_hindi, test_english = zip(*test_data)

    # Pad sentences
    train_pad_english = pad_sentence(train_english)
    train_pad_hindi = pad_sentence(train_hindi)
    test_pad_english = pad_sentence(test_english)
    test_pad_hindi = pad_sentence(test_hindi)

    # Convert to tensors
    train_eng_tensor = torch.tensor(train_pad_english)
    train_hind_tensor = torch.tensor(train_pad_hindi)
    test_hindi_tensor = torch.tensor(test_pad_hindi)
    test_eng_tensor = torch.tensor(test_pad_english)


    train_dataset = TensorDataset(train_hind_tensor, train_eng_tensor)
    test_dataset = TensorDataset(test_hindi_tensor, test_eng_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, hindi_vocab, english_vocab
