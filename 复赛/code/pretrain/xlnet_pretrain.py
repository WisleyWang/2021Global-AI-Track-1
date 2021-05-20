#!/usr/bin/env python
# coding: utf-8

# In[7]:





# In[1]:


# coding:utf8
import random
import pandas as pd
import warnings
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Tuple
from transformers import XLNetConfig,XLNetLMHeadModel,XLNetPreTrainedModel,XLNetModel,XLMProphetNetForCausalLM,DataCollatorForWholeWordMask,DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
import shutil
from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForPermutationLanguageModeling,
    
)
from typing import Dict
from transformers import PreTrainedTokenizer
from transformers.models.bert.modeling_bert import (
    BertOutput,
    BertPooler,
    BertSelfOutput,
    BertIntermediate,
    BertOnlyMLMHead,
    BertOnlyNSPHead,
    BertPreTrainingHeads,
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
    
)
import tokenizers

warnings.filterwarnings('ignore')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_DEVICE_ORDER"] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def read_data(train_file_path, tokenizer: BertTokenizer) -> dict:
    train_data = open(train_file_path, 'r', encoding='utf8').readlines()

    inputs = defaultdict(list)
    for row in tqdm(train_data, desc=f'Preprocessing train data', total=len(train_data)):
        sentence = row.strip()
        inputs_dict = tokenizer.encode_plus(sentence, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)
        inputs['input_ids'].append(inputs_dict['input_ids'])
        inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
        inputs['attention_mask'].append(inputs_dict['attention_mask'])

    return inputs


class OppoDataset(Dataset):

    def __init__(self,train_file_path,tokenizer,maxlen):
        super(OppoDataset, self).__init__()
        self.data=pd.read_csv(train_file_path,header=None)
        self.tokenizer = tokenizer
        self.maxlen = maxlen
    def __getitem__(self, index: int) -> tuple:
        self.tokenizer(sent, padding='max_length',  # Pad to max_lengthtruncation=True,       # Truncate to max_length
                                     max_length=self.maxlen, add_special_tokens=True, 
                                return_tensors='pt')  # R
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        return token_ids,attn_masks

    def __len__(self):
        return len(self.data)

class Collator:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer, mlm_probability=0.15):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}

    def pad_and_truncate(self, input_ids_list, attention_mask_list, max_seq_len):
        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids_list)):
            seq_len = min(len(input_ids_list[i]), max_seq_len)
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len], dtype=torch.long)
            else:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i][:seq_len - 1] +
                                                      [self.tokenizer.sep_token_id], dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i][:seq_len], dtype=torch.long)
        return input_ids, attention_mask

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
        pvals = pvals[::-1]
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
        input_ids_list, attention_mask_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len =min(cur_max_seq_len, self.max_seq_len)

        input_ids, attention_mask = self.pad_and_truncate(input_ids_list, attention_mask_list, max_seq_len)
        batch_mask = self.ngram_mask(input_ids_list, max_seq_len)
        input_ids, mlm_labels = self.mask_tokens(input_ids, batch_mask)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': mlm_labels
        }

        return data_dict



# In[2]:


class XLNetForMaskedLM(XLNetPreTrainedModel):
    def __init__(self, config,name):
        super().__init__(config)
        self.bert = XLNetModel(config)
        self.cls = BertOnlyMLMHead(config)
#         self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    def forward(
         self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_mems=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
         
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        ltr_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_labels` is provided):
                Next token prediction loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        Examples::

            from transformers import BertTokenizer, BertForMaskedLM
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')

            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, masked_lm_labels=input_ids)

            loss, prediction_scores = outputs[:2]

        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        masked_lm_labels = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs
        return outputs  # (ltr_lm_loss), (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # if model is does not use a causal mask then add a dummy token
        if self.config.is_decoder is False:
            assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1
            )

            dummy_token = torch.full(
                (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
            )
            input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True,
                                   max_length=block_size, padding='max_length')
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]



class OppoDataset(Dataset):

    def __init__(self,train_file_path,tokenizer,maxlen):
        super(OppoDataset, self).__init__()
        self.data=pd.read_csv(train_file_path,header=None)
        self.data.columns=['sentence']
        self.tokenizer=tokenizer
        self.maxlen = maxlen
    def __getitem__(self, index: int) -> tuple:
        sent = str(self.data.loc[index,'sentence'])
        encoded_pair=self.tokenizer(sent, padding='max_length',  # Pad to max_lengthtruncation=True,       # Truncate to max_length
                                     max_length=self.maxlen, add_special_tokens=True, 
                                return_tensors='pt')  # R
#         print(encoded_pair)
        encoded_pair['input_ids'] = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        encoded_pair['attention_mask'] = encoded_pair['attention_mask'].squeeze(0)
        encoded_pair['token_type_ids']=encoded_pair['token_type_ids'].squeeze(0)
        return encoded_pair

    def __len__(self):
        return len(self.data)


# In[18]:


def main(train_epoch, batch_size, seq_length, lr, corpus_path, vocab_path,
         config_path, pretrain_model_path, output_record_path, model_save_path):
    seed_everything(997)
    num_train_epochs = train_epoch
    pretrain_batch_size = batch_size
    seq_length = seq_length
    lr = lr
    corpus_path = corpus_path
    vocab_path = vocab_path
    config_path = config_path
    pretrain_model_path = pretrain_model_path
    output_record_path = output_record_path
    model_save_path = model_save_path

    tokenizer = BertTokenizer.from_pretrained(vocab_path)
#     train_dataset = LineByLineTextDataset(block_size=128, file_path=corpus_path, tokenizer=tokenizer)
    
#     data = read_data(corpus_path, tokenizer)
    train_dataset = OppoDataset(train_file_path=corpus_path,tokenizer=tokenizer,maxlen=128)
    
    data_collator = DataCollatorForPermutationLanguageModeling(tokenizer=tokenizer)

    config = XLNetConfig.from_pretrained(pretrained_model_name_or_path=config_path)
#     model = XLNetForMaskedLM(config=config,name='./xlnet_model/pytorch_model.bin')
    if os.path.exists(pretrain_model_path):
        model = XLNetLMHeadModel.from_pretrained(pretrain_model_path,config=config)
    else:
        model = XLNetLMHeadModel(config=config)
#     data_collator = Collator(max_seq_len=seq_length, tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=output_record_path, overwrite_output_dir=True, num_train_epochs=num_train_epochs,
        learning_rate=lr, dataloader_num_workers=8, prediction_loss_only=True, fp16=True, fp16_backend='amp',
        per_device_train_batch_size=pretrain_batch_size, save_strategy='no',seed=997
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    trainer.train()
    trainer.save_model(model_save_path)

def write(sent, path):
    with open(path, 'a+', encoding='utf-8') as f:
        f.writelines(sent+'\n')
if __name__ == '__main__':
    """
    num_train_epochs = 60
    pretrain_batch_size = 64
    seq_length = 32
    lr = 6e-5
    corpus_path = 'data/pair_connect_with_tab_data.tsv'
    vocab_path = '***/vocab.txt'
    config_path = '***/config.json'
    pretrain_model_path = '***/pytorch_model.bin'
    output_record_path = '***/record'
    model_save_path = '***/model'
    """
    
    
    # 生成预训练模型训练数据
#     复赛测试集：
    with open('../../tcdata/testA.csv', 'r', encoding="utf-8") as f:
        for id, line in enumerate(f):
            _, sent = line.strip().split('|,|')
            write(sent, './data/pretrain.tsv')
    with open('../../tcdata/testB.csv', 'r', encoding="utf-8") as f:
        for id, line in enumerate(f):
            _, sent = line.strip().split('|,|')
            write(sent, './data/pretrain.tsv')

    if not os.path.exists('./xlnet_model/pytorch_model.bin'): # 
        # 复赛的train
        with open('../../tcdata/train.csv', 'r', encoding="utf-8") as f:
            for id, line in enumerate(f):
                _, sent, __ = line.strip().split('|,|')
                write(sent, './data/pretrain.tsv')
    #     初赛的train
        with open('../../tcdata/track1_round1_train_20210222.csv', 'r', encoding="utf-8") as f:
            for id, line in enumerate(f):
                _, sent, __ = line.strip().split('|,|')
                write(sent, './data/pretrain.tsv')
    #     # 初赛测试集
        with open('../../tcdata/track1_round1_testA_20210222.csv', 'r', encoding="utf-8") as f:
            for id, line in enumerate(f):
                _, sent = line.strip().split('|,|')
                write(sent, './data/pretrain.tsv')
        with open('../../tcdata/track1_round1_testB.csv', 'r', encoding="utf-8") as f:
            for id, line in enumerate(f):
                _, sent = line.strip().split('|,|')
                write(sent, './data/pretrain.tsv')

            
    # 生成词汇表     
    with open('../../tcdata/train.csv', 'r', encoding="utf-8") as f:
        for id, line in enumerate(f):
            _, sent, __ = line.strip().split('|,|')
            write(sent, './data/train_data.tsv')
    with open('../../tcdata/track1_round1_train_20210222.csv', 'r', encoding="utf-8") as f:
        for id, line in enumerate(f):
            _, sent, __ = line.strip().split('|,|')
            write(sent, './data/train_data.tsv')
    bwpt = tokenizers.BertWordPieceTokenizer()
    bwpt.train(
        files=[ './data/train_data.tsv'],
        vocab_size=50000,
        min_frequency=1,
        limit_alphabet=1000
    )
    bwpt.save_model('../../tmp/')

    
    if not os.path.exists('./xlnet_model/pytorch_model.bin'):
        main(300, 64, 100, 6e-5,
             'data/pretrain.tsv',
             '../../tmp/vocab.txt',
             'xlnet_model/config.json',
             'xlnet_model/pytorch_model.bin',
             'output/record', '../../model_weight/xlnet/'
             )
        shutil.copy('../../tmp/vocab.txt','../../model_weight/xlnet/vocab.txt') # 预训练模型的词汇表必须要统一
    else:
        main(80, 64, 100,4e-5,
             './data/pretrain.tsv',
             './xlnet_model/vocab.txt',
             './xlnet_model/config.json',
             './xlnet_model/pytorch_model.bin',
             './output/record', '../../model_weight/xlnet/'
             )
    
#         shutil.move('./output/model/config.json','../../model_weight/xlnet/')
#         shutil.move('./output/model/pytorch_model.bin','../../model_weight/xlnet/')

# In[ ]:




