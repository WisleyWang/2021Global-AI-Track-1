#!/usr/bin/env python
# coding: utf-8

# In[2]:


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer,AutoModelForPreTraining
from transformers import AutoTokenizer,BertTokenizerFast,AutoModel
from torch import nn
from torch.optim import AdamW
import pandas as pd
import numpy as np
import torch
import gc
from tqdm import tqdm
from sklearn.model_selection import KFold
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from model.modeling_nezha import NeZhaForSequenceClassification,NeZhaPreTrainedModel,NeZhaModel,NeZhaForTokenClassification
from model.configuration_nezha import NeZhaConfig
import  torch.nn.functional as F
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
import os

# In[3]:

print('nezhav3-------------')
class NeZhaForSequenceClassification(NeZhaPreTrainedModel):
    def __init__(self, config,model_name,num_labels1,num_labels2):
        super().__init__(config)
        self.num_labels1 = num_labels1
        self.num_labels2=num_labels2
        self.bert = NeZhaModel.from_pretrained(model_name)
        self.attn1=Attn(config.hidden_size)
        self.attn2=Attn(config.hidden_size)
        self.attn3=Attn(config.hidden_size)
        self.attn4=Attn(config.hidden_size)
        self.attn5=Attn(config.hidden_size)
        self.attn6=Attn(config.hidden_size)
        self.attn7=Attn(config.hidden_size)
        self.attn8=Attn(config.hidden_size)
        self.attn9=Attn(config.hidden_size)
        self.attn10=Attn(config.hidden_size)
        self.attn11=Attn(config.hidden_size)
        self.dropouts=nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1,0.5,3)])
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, self.num_labels1-14)
        self.classifier7 = nn.Linear(config.hidden_size, 2)
        self.classifier3 = nn.Linear(config.hidden_size, 3)
        self.classifier5 = nn.Linear(config.hidden_size, 3)
        self.classifier8 = nn.Linear(config.hidden_size, 3)
        self.classifier9 = nn.Linear(config.hidden_size, 3)
        
        self.classifier2 = nn.Linear(config.hidden_size, self.num_labels2-10)
        self.classifier4 = nn.Linear(config.hidden_size,2)
        self.classifier6 = nn.Linear(config.hidden_size,2)
        self.classifier10 = nn.Linear(config.hidden_size,3)
        self.classifier11 = nn.Linear(config.hidden_size,3)
        self.predict=nn.Sigmoid()
#         self.init_weights()
#         if True:
#             for p in self.bert.parameters(): # 冻结所有bert层
#                 p.requires_grad = False

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels1=None,
        labels2=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
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

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )

#         pooled_output = outputs[1]

        att1=self.attn1(outputs[0])
        att2=self.attn2(outputs[0])
        att3=self.attn3(outputs[0])
        att4=self.attn4(outputs[0])
        att5=self.attn5(outputs[0])
        att6=self.attn6(outputs[0])
        att7=self.attn7(outputs[0])
        att8=self.attn8(outputs[0])
        att9=self.attn9(outputs[0])
        att10=self.attn10(outputs[0])
        att11=self.attn11(outputs[0])
        
        pooled_output1 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output2 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output3 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output4 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output5 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output6 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output7 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output8 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output9 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output10 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output11 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
        
        logits1 = self.classifier1(pooled_output1)
        logits2 = self.classifier2(pooled_output2)
        logits3 = self.classifier3(pooled_output3)
        logits4 = self.classifier4(pooled_output4)
        logits5 = self.classifier5(pooled_output5)
        logits6 = self.classifier6(pooled_output6)
        logits7 = self.classifier7(pooled_output7)
        logits8 = self.classifier8(pooled_output8)
        logits9 = self.classifier9(pooled_output9)
        logits10 = self.classifier10(pooled_output10)
        logits11 = self.classifier11(pooled_output11)
        
        logits1=torch.cat([logits1,logits3,logits5,logits7,logits8,logits9],dim=-1)
        logits2=torch.cat([logits2,logits4,logits6,logits10,logits11],dim=-1)

        predict1=self.predict(logits1)
        predict2=self.predict(logits2)
        outputs = (predict1,predict2) + outputs[2:]  # add hidden states and attention if they are here
#         print('label:',labels)
#         print('input_ids:',input_ids)
#         print('attention_mas:',attention_mask)
        if labels1 is not None:
            loss_fct = nn.BCELoss()
#                 print(logits.view(-1, self.num_labels))
#                 print(labels.view(-1, self.num_labels))
            loss1 = loss_fct(predict1.view(-1, self.num_labels1), labels1.view(-1, self.num_labels1))
            loss2 = loss_fct(predict2.view(-1, self.num_labels2), labels2.view(-1, self.num_labels2))
            loss=loss1+loss2
            outputs = (loss,) + outputs

        return outputs  # (loss), predict1,predict2, (hidden_states), (attentions)
    
class Attn(nn.Module):
    def __init__(self,hidden_size):
        super(Attn, self).__init__()
        self.attn = nn.Linear(hidden_size,1)
    def forward(self, x):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''   
        att=self.attn(x)
        att=F.tanh(att)
        att=F.softmax(att,1)
        att_x=att*x
        return att_x.sum(1)
import torch.utils.data as Data
class CustomDataset(Data.Dataset):
    def __init__(self, data, maxlen,tokenizer,with_labels=True, model_name='bert-base-chinese'):
        self.data = data  # pandas dataframe

        #Initialize the tokenizer
        self.tokenizer = tokenizer#AutoTokenizer.from_pretrained(model_name, use_fast=True)  
        self.maxlen = maxlen
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)
    def get_label(self,x,num):
        label=[0]*num
       
        x=x.strip().split(' ')

        for l in x:              
            if l and l!='nan':
                label[int(l)]=1
        return label
    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent = str(self.data.loc[index, 'sentence'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,       # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
#         print(encoded_pair['input_ids'])
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
#         token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label1 = torch.Tensor(self.get_label(str(self.data.loc[index, 'label1']),17))
            label2 = torch.Tensor(self.get_label(str(self.data.loc[index, 'label2']),12))
            return token_ids, attn_masks,label1,label2
        else:
            return token_ids, attn_masks
from sklearn.utils import shuffle as reset
def train_test_split(data_df, test_size=0.2, shuffle=True, random_state=None):
    if shuffle:
        data_df = reset(data_df, random_state=random_state)

    train = data_df[int(len(data_df)*test_size):].reset_index(drop = True)
    test  = data_df[:int(len(data_df)*test_size)].reset_index(drop = True)

    return train, test

from torch.nn.functional import cross_entropy,binary_cross_entropy


def eval(model,epoch, validation_dataloader,output_model = './train_class/model.pth'):

    model.eval()
    eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            predict1,predict2 = model(batch[0], batch[1])
            predict1,predict2 = predict1.detach().cpu(),predict2.detach().cpu()
            label_ids1,label_ids2 = batch[2].cpu(),batch[3].cpu()
            
            tmp_eval_accuracy = binary_cross_entropy(predict1, label_ids1.float()).item()+binary_cross_entropy(predict2, label_ids2.float()).item()
            
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

    print("Validation mlogloss: {}".format(eval_accuracy / nb_eval_steps))
    val_loss[epoch]=eval_accuracy / nb_eval_steps
    global best_score
    if best_score > eval_accuracy / nb_eval_steps:
        best_score = eval_accuracy / nb_eval_steps
        save(model,output_model)
        return 0
    return 1
def save(model,output_model):
    # save
    torch.save(model, output_model)
    print('The best model has been saved')
def flat_accuracy(preds, labels):
#     print(preds,labels)
    return -np.mean(labels*np.log(preds+1.e-7)+(1-labels)*np.log(preds+1.e-7))*10

# 对抗训练
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
    def attack(self, epsilon=1.0, emb_name='word_emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='word_emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
from collections import defaultdict
from torch.optim import Optimizer
import torch


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)

from torch.optim.lr_scheduler import LambdaLR
class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Multiplies the learning rate defined in the optimizer by a dynamic variable determined by the current step.
        Linearly increases the multiplicative variable from 0. to 1. over `warmup_steps` training steps.
        Linearly decreases the multiplicative variable from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)
        return loss
def build_optimizer(model, train_steps, learning_rate):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=False, eps=1e-8)
    optimizer = Lookahead(optimizer, 5, 1)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * 0.1, t_total=train_steps)
    return optimizer, scheduler
def to_predict(model, dataloader,output_model, with_labels=False):
    
    # load model
    checkpoint = torch.load(output_model, map_location='cuda')
#     print(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print('-----Testing-----')

    pred_label =np.zeros((len(test),29))
    model.eval()
    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            predict1,predict2 = model(batch[0], batch[1])
            predict1 = predict1.detach().cpu().numpy()
            predict2 = predict2.detach().cpu().numpy()
            predict=np.concatenate([predict1,predict2],axis=-1)
            pred_label[i*batch_size:(i+1)*batch_size]=predict
    return pred_label

class PGD():
    def __init__(self, model, emb_name='word_emb', epsilon=1., alpha=1):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}
    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

# In[4]:


train=pd.read_csv('../tcdata/train.csv',header=None)
# test=pd.read_csv('../tcdata/track1_round1_testB.csv',header=None)
test=pd.read_csv('../tcdata/testB.csv',header=None)
model_path='../model_weight/nezha/'
output_model='../tmp/nezha.pth'
batch_size=32
# 合并训练集与测试集 制作特征
for i in range(1,3):
    train[i]=train[i].apply(lambda x:x.replace('|','').strip())
for i in range(1,2):
    test[i]=test[i].apply(lambda x:x.replace('|','').strip())
train.columns=['idx','sentence','label1','label2']
test.columns=['idx','sentence']

tokenizer=BertTokenizerFast.from_pretrained(model_path)

config=NeZhaConfig.from_pretrained(model_path,num_labels=17,hidden_dropout_prob=0.3) # config.output_attentions=True


# In[5]:


def train_model(train_df,val_df,test_oof):
    
        ###--------------------
    early_stop=0
    print("Reading training data...")
    train_set = CustomDataset(train_df, maxlen=128,tokenizer=tokenizer)
    train_loader = Data.DataLoader(train_set, batch_size=batch_size, num_workers=5, shuffle=True)

    print("Reading validation data...")
    val_set = CustomDataset(val_df, maxlen=128, tokenizer=tokenizer)
    val_loader = Data.DataLoader(val_set, batch_size=batch_size, num_workers=5, shuffle=True)

    test_set = CustomDataset(test, maxlen=128, tokenizer=tokenizer,with_labels=False)
    test_loader = Data.DataLoader(test_set, batch_size=batch_size, num_workers=5, shuffle=False)
    # 准备模型
    model=NeZhaForSequenceClassification(config=config,model_name=model_path,num_labels1=17,num_labels2=12)
    ### 训练
    model.to(device)
#     fgm = FGM(model)
    pgd = PGD(model)
    K = 5
    train_num = len(train_set)
    train_steps = int(train_num * epochs / batch_size) + 1

    optimizer, scheduler = build_optimizer(model, train_steps, learning_rate=5e-5)
    print('-----Training-----')
    for epoch in range(epochs):
        model.train()
        model.zero_grad()
        print('Epoch', epoch)
        for i, batch in enumerate(tqdm(train_loader)):
            batch = tuple(t.to(device) for t in batch)
            loss, predict1,predict2 = model(batch[0], batch[1], batch[2],batch[3])
            if i % 510 == 0:
                print(i, loss.item())
            optimizer.zero_grad()
            loss.backward()

            # 对抗训练
#             fgm.attack()
#             loss_adv, _,_  = model(batch[0], batch[1], batch[2],batch[3])
#             loss_adv.backward()
#             fgm.restore()
            pgd.backup_grad()
    # 对抗训练
            for t in range(K):
                pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K-1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                loss_adv, _,_ = model(batch[0], batch[1], batch[2],batch[3])
                loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore() # 恢复embedding参数
            optimizer.step()
            scheduler.step()
   
    #         if i % 20 == 0:
    #             eval(model, optimizer, val_loader, output_model='./runs/nezha1.pth')
        if epoch>-1:
            trn_loss[epoch]=loss.item()
            early_stop+=eval(model, epoch, val_loader, output_model='../tmp/nezhav3_{}.pth'.format(fold_s))
        if early_stop==2:
            break

#     test_oof.append(to_predict(model, test_loader,output_model, with_labels=False))
    torch.cuda.empty_cache()
    gc.collect()
#     return test_oof   


# In[6]:


n_fold=KFold(5,shuffle=True,random_state=998)
test_oof=[]
epochs = 5
if os.path.exists('./log/nezha_v3.txt'):
    log_df=pd.read_csv('./log/nezha_v3.txt')
else:
    log_df=pd.DataFrame()
log_df['epoch']=['epoch0','epoch1','epoch2','epoch3','epoch4']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fold_s=0
for trn_idx,val_idx in n_fold.split(train):
    print('---------nezhav3_train: %d fold -----------'%fold_s)
    train_df=train.iloc[trn_idx].reset_index(drop=True)
    val_df=train.iloc[val_idx].reset_index(drop=True)
    best_score = float('inf')
    val_loss=np.zeros(5)
    trn_loss=np.zeros(5)
    train_model(train_df,val_df,test_oof)
    log_df['trn_nezhav3_{}'.format(fold_s)]=trn_loss
    log_df['val_nezhav3_{}'.format(fold_s)]=val_loss  
    log_df.to_csv('./log/nezha_v3.txt',index=False)
    fold_s+=1

# In[10]:
# for i in range(len(test_oof)):
#     test_oof[i]=test_oof[i]-test_oof[i].min(axis=1,keepdims=True)
# test_oof=np.mean(test_oof,axis=0)    
# test_oof=test_oof-test_oof.min(axis=1,keepdims=True)

# sub=pd.DataFrame()
# test=pd.read_csv('../tcdata/testA.csv',header=None)
# sub['report_ID']=test[0]
# sub['Prediction']=[ '|'+' '.join(['%.12f'%j for j in i]) for i in test_oof ]
# sub.to_csv('../result_nezhav3.csv',index=False,header=False)


# In[ ]:




