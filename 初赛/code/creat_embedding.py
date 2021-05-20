import panda as pd
from gensim.models import Word2Vec
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# w2v预训练
data=pd.read_csv('../user_data/tmp_data/all_data_txt.txt',header=None)

w2v=Word2Vec(data[0].apply(lambda x:x.split(' ')).tolist(),size=128, window=8, iter=50, min_count=2,
                     sg=1, sample=0.002, workers=6 , seed=1018)
# 保存模型
w2v.wv.save_word2vec_format('../user_data/pretraining_model/w2v_128.txt')

## transfomer预训练-----------------------
from transformers import  BertTokenizer, WEIGHTS_NAME,TrainingArguments
from model.modeling_nezha import NeZhaForSequenceClassification,NeZhaForMaskedLM
from model.configuration_nezha import NeZhaConfig
import tokenizers
from datasets import load_dataset,Dataset

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    LineByLineTextDataset
)
# 进行分词 得到词汇表，其实通过已经训练好的tokenizer直接分也可由
bwpt = tokenizers.BertWordPieceTokenizer()
filepath = '../user_data/tmp_data/all_data_txt.txt'
bwpt.train(
	files=[filepath],
	vocab_size=50000,
	min_frequency=1,
	limit_alphabet=1000
)
bwpt.save_model('../pretrained_models/')

# 加载nezha模型
# 需要提前下载nezha 模型，https://pan.baidu.com/s/1sPC-FZJ20RtTEw9UX_4sDw 提取码: hckq
model_path='../pretraining_model/nezha-cn-base/'
tokenizer =  BertTokenizer.from_pretrained('./pretrained_models/vocab.txt', do_lower_case=True)
config=NeZhaConfig.from_pretrained(model_path)
model=NeZhaForMaskedLM.from_pretrained(model_path, config=config)
# 加载训练文本
train_dataset=LineByLineTextDataset(tokenizer=tokenizer,file_path='../user_data/tmp_data/all_data_txt.txt',block_size=128)

#### 训练
num_train_epochs = 300

pretrain_batch_size = 32
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
training_args = TrainingArguments(
    output_dir='../user_data/pretraining_model/', overwrite_output_dir=True, num_train_epochs=num_train_epochs, learning_rate=6e-5,
    per_device_train_batch_size=pretrain_batch_size, save_total_limit=10)

trainer = Trainer(
    model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset)

trainer.train()

trainer.save_model('../user_data/preytaoning_model/nezha')

## 生成embedding
model_path='../user_data/preytaoning_model/nezha/'
# model_path='./pretrained_models/nezha-cn-base/'
tokenizer =  BertTokenizer.from_pretrained(model_path, do_lower_case=True)
config=NeZhaConfig.from_pretrained(model_path)
model=NeZhaForMaskedLM.from_pretrained(model_path, config=config)

datasets=load_dataset('text',data_files={'train':filepath })
column_names = datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]
## 保存seq
def tokenize_function(examples):
    # Remove empty lines
    examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=102,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=None,
    remove_columns=[text_column_name],
    load_from_cache_file=False,
)
import torch
import numpy as np
# 获得句子的seq编码
a=np.array(tokenized_datasets['train']['input_ids'])
a=a[:,1:-1]
# 保存
np.save('../user_data/tmp_data/nezha_seqB.npy',a)

vocob=np.unique(a)# 得到所有词汇的seq表示
embedding_matrix=np.zeros((vocob.max() ,768))
for v in vocob:
    embedding_matrix[v]=model.get_input_embeddings()(torch.Tensor([v]).long()).detach().numpy()[0]
np.save('../user_data/tmp_data/nezha_embedding.npy',embedding_matrix) # 保存embedding