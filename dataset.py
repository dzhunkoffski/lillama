from typing import Any, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import os
from tqdm.autonotebook import tqdm
import random
import glob
import json

class Collator(object):
    def __init__(self, pad_value: int):
        self.pad_idx = pad_value
    
    def __call__(self, batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        actual_length_batch = []
        tokens_batch = []
        for item in batch:
            tokens_batch.append(item[0])
            actual_length_batch.append(item[1])
        actual_length_batch = torch.LongTensor(actual_length_batch)
        tokens_batch = pad_sequence(tokens_batch, batch_first=True, padding_value=self.pad_idx)
        return tokens_batch, actual_length_batch

class TextDataset(Dataset):
    def __init__(
            self, 
            corpus_path: str, save_tokenizer_to: str, 
            max_len: int, vocab_size: int, normalization_rule: str = 'nmt_nfkc_cf', sample: int = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.corpus_path = None
        self.corpus_path = corpus_path
        self.texts = self._load_texts_txt()
        if sample is not None:
            self.texts = random.sample(self.texts, sample)
        self.tokenizer = self._load_tokenizer(save_tokenizer_to, vocab_size, normalization_rule)
        self.bos_id = self.tokenizer.bos_id()
        self.eos_id = self.tokenizer.eos_id()
        self.unk_id = self.tokenizer.unk_id()
        self.pad_id = self.tokenizer.pad_id()
        self.max_len = max_len
    
    def _load_texts_txt(self):
        with open(self.corpus_path, 'r') as fd:
            data = []
            for text in tqdm(fd):
                data.append(text)
        return data
    
    def _load_texts_json(self):
        json_paths = glob.glob(f'{self.corpus_path}/**.json')
        data = []
        for json_file in json_paths:
            with open(json_file, 'r') as fd:
                obj = json.load(fd)
            for text_ix in range(len(obj)):
                data.append(obj[text_ix]['story'])
        return data
    
    def _load_tokenizer(self, tokenizer_prefix_path: str, vocab_size: int, normalization_rule: str):
        if not os.path.exists(f'{tokenizer_prefix_path}.model'):
            spm.SentencePieceTrainer.train(
                f'--input={self.corpus_path} --vocab_size={vocab_size} --model_prefix={tokenizer_prefix_path} --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --model_type=bpe --normalization_rule_name={normalization_rule} --input_sentence_size=50000 --shuffle_input_sentence=false'
            )

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(f'{tokenizer_prefix_path}.model')
        return tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text =  self.texts[item]
        ids = self.tokenizer.encode_as_ids(text)
        if len(ids) > self.max_len - 2:
            ids = ids[:self.max_len-2]
        ids = [self.bos_id] + ids + [self.eos_id]
        actual_len = len(ids)
        return torch.LongTensor(ids), actual_len