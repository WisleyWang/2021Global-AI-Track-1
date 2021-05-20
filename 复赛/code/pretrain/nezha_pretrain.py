# coding:utf8

import warnings
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Tuple

import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask
from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments,


)
from model.modeling_nezha import NeZhaForMaskedLM
from model.configuration_nezha import NeZhaConfig
import shutil
warnings.filterwarnings('ignore')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_DEVICE_ORDER"] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def read_data(train_file_path, tokenizer: BertTokenizer) -> dict:
    # train_df = pd.read_csv(train_file_path, header=None, sep='\t')
    train_df = pd.read_csv(train_file_path, header=None)

    inputs = defaultdict(list)
    for i, row in tqdm(train_df.iterrows(), desc=f'Preprocessing train data', total=len(train_df)):
        # sentence_a, sentence_b = row[0], row[1]
        sentence = row[0]
        inputs_dict = tokenizer.encode_plus(sentence, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)
        inputs['input_ids'].append(inputs_dict['input_ids'])
        inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
        inputs['attention_mask'].append(inputs_dict['attention_mask'])

    return inputs


class OppoDataset(Dataset):

    def __init__(self, data_dict: dict):
        super(OppoDataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (self.data_dict['input_ids'][index], self.data_dict['token_type_ids'][index],
                self.data_dict['attention_mask'][index])
        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class Collator:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer, mlm_probability=0.15):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}

    def pad_and_truncate(self, input_ids_list, token_type_ids_list,
                         attention_mask_list, max_seq_len):
        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids_list)):
            seq_len = min(len(input_ids_list[i]), max_seq_len)
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len], dtype=torch.long)
            else:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len - 1] +
                                                      [self.tokenizer.sep_token_id], dtype=torch.long)
            token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i][:seq_len], dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i][:seq_len], dtype=torch.long)
        return input_ids, token_type_ids, attention_mask

    def _ngram_mask(self, input_ids, max_seq_len, seed):
        np.random.seed(seed)
        cand_indexes = []
        for (i, id_) in enumerate(input_ids):
            if id_ in self.special_token_ids:
                continue
            cand_indexes.append([i])
        num_to_predict = max(1, int(round(len(input_ids) * self.mlm_probability)))

        ngrams = np.arange(1, 4, dtype=np.int64)
        pvals = 1. / np.arange(1, 4)
        pvals /= pvals.sum(keepdims=True)
        # favor_shorter_ngram:
        # pvals = pvals[::-1]  # 很关键？
        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            ngram_indexes.append(ngram_index)
        np.random.shuffle(ngram_indexes)

        covered_indexes = set()
        for cand_index_set in ngram_indexes:
            if len(covered_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes:
                        continue
            n = np.random.choice(ngrams[:len(cand_index_set)],
                         p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
            while len(covered_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            if len(covered_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_ids))]
        mask_labels += [0] * (max_seq_len - len(mask_labels))
        return torch.tensor(mask_labels[:max_seq_len])

    def ngram_mask(self, input_ids_list: List[list], max_seq_len: int):
        mask_labels = []
        for i, input_ids in enumerate(input_ids_list):
            mask_label = self._ngram_mask(input_ids, max_seq_len, seed=i)
            mask_labels.append(mask_label)
        return torch.stack(mask_labels, dim=0)

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training
        # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = mask_labels
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask = self.pad_and_truncate(input_ids_list,
                                                                          token_type_ids_list,
                                                                          attention_mask_list,
                                                                          max_seq_len)
        batch_mask = self.ngram_mask(input_ids_list, max_seq_len)
        input_ids, mlm_labels = self.mask_tokens(input_ids, batch_mask)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': mlm_labels
        }

        return data_dict


def main(train_epoch, batch_size, seq_length, lr, corpus_path, vocab_path,
         config_path, pretrain_model_path, output_record_path, model_save_path):
    num_train_epochs = train_epoch
    pretrain_batch_size = batch_size
    seq_length = seq_length
    lr = lr
    corpus_path = corpus_path
    vocab_path = vocab_path
    config_path = config_path
    output_record_path = output_record_path
    model_save_path = model_save_path

    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    data = read_data(corpus_path, tokenizer)

    train_dataset = OppoDataset(data)

    config = NeZhaConfig.from_pretrained(pretrained_model_name_or_path=config_path)
    if  os.path.exists(pretrain_model_path):
        model = NeZhaForMaskedLM.from_pretrained( pretrain_model_path,config=config)
    else:
        model = NeZhaForMaskedLM(config=config)

    data_collator = Collator(max_seq_len=seq_length, tokenizer=tokenizer, mlm_probability=0.25)

    training_args = TrainingArguments(
        output_dir=output_record_path, overwrite_output_dir=True, num_train_epochs=num_train_epochs,
        learning_rate=lr, dataloader_num_workers=8, prediction_loss_only=True, fp16=True, fp16_backend='amp',
        per_device_train_batch_size=pretrain_batch_size,save_strategy='no'
#         save_steps=2000, save_total_limit=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(model_save_path)


if __name__ == '__main__':
    """
    num_train_epochs = 60
    pretrain_batch_size = 64
    seq_length = 32
    lr = 6e-5
    corpus_path = 'data/pretrain.tsv'
    vocab_path = '***/vocab.txt'
    config_path = '***/config.json'
    pretrain_model_path = '***/pytorch_model.bin'
    output_record_path = '***/record'
    model_save_path = '***/model'
    """
    if not os.path.exists('./nezha_model/pytorch_model.bin'):
        main(420, 64, 100, 6e-5,
             './data/pretrain.tsv',
             '../../tmp/vocab.txt',
             './nezha_model/config.json',
            './nezha_model/pytorch_model.bin',
             './output/record', '../../model_weight/nezha/')
        shutil.copy('../../tmp/vocab.txt','../../model_weight/nezha/vocab.txt')
    else:
         main(80, 64, 100, 4e-5,
             './data/pretrain.tsv',
             './nezha_model/vocab.txt',
             './nezha_model/config.json',
            './nezha_model/pytorch_model.bin',
             './output/record', '../../model_weight/nezha/')