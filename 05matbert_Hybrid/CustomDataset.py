import torch
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
import string
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
# 고유한 문자 집합 정의
char_set = string.ascii_letters + string.digits + string.punctuation + " "
char_vocab_size = len(char_set)

# 문자와 인덱스를 매핑하는 딕셔너리 생성
char_to_index = {char: idx for idx, char in enumerate(char_set)}
index_to_char = {idx : char for idx, char in enumerate(char_set)}
# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, texts, is_nanoparticle_labels, main_subject_labels, tokenizer, max_len):
        self.texts = texts
        self.is_nanoparticle_labels = is_nanoparticle_labels
        self.main_subject_labels = main_subject_labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.char_to_index = char_to_index

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        is_nanoparticle_label = self.is_nanoparticle_labels[idx]
        main_subject_label = self.main_subject_labels[idx]

        word_encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        char_encoding = self.char_tokenize(text)

        return {
            'word_input_ids': word_encoding['input_ids'].flatten(),
            'char_input_ids': char_encoding['input_ids'].flatten(),
            'is_nanoparticle_labels': torch.tensor(is_nanoparticle_label, dtype=torch.float),
            'main_subject_labels': torch.tensor(main_subject_label, dtype=torch.float)
        }

    def char_tokenize(self, text):
        char_tokens = list(text)
        char_ids = [self.char_to_index.get(c, self.char_to_index[' ']) for c in char_tokens]

        if len(char_ids) < self.max_len:
            padding_length = self.max_len - len(char_ids)
            char_ids += [self.char_to_index[' ']] * padding_length
        else:
            char_ids = char_ids[:self.max_len]

        return {
            'input_ids': torch.tensor(char_ids)
        }