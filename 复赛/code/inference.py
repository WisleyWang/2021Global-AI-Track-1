#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BertTokenizerFast,XLNetModel,MPNetModel,MPNetPreTrainedModel
import torch
import pandas as pd
import numpy as np
import torch.utils.data as Data
from model.modeling_nezha import NeZhaPreTrainedModel
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from model.modeling_nezha import NeZhaForSequenceClassification,NeZhaPreTrainedModel,NeZhaModel,NeZhaForTokenClassification
from model.configuration_nezha import NeZhaConfig
import  torch.nn.functional as F
import os
from torch import nn
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
import gc
from tqdm import tqdm


# In[2]:


def to_predict(model, dataloader, with_labels=False):
    
    # load model
#     checkpoint = torch.load(output_model, map_location='cuda')
#     print(checkpoint)
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


# In[3]:


# test=pd.read_csv('../tcdata/track1_round1_testB.csv',header=None)
test=pd.read_csv('../tcdata/testB.csv',header=None)
train=pd.read_csv('../tcdata/train.csv',header=None)

for i in range(1,3):
    train[i]=train[i].apply(lambda x:x.replace('|','').strip())
for i in range(1,2):
    test[i]=test[i].apply(lambda x:x.replace('|','').strip())
train.columns=['idx','sentence','label1','label2']
test.columns=['idx','sentence']
batch_size=32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


class NeZhaForSequenceClassification(NeZhaPreTrainedModel):
    def __init__(self, config,model_name,num_labels1,num_labels2):
        super().__init__(config)
        self.num_labels1 = num_labels1
        self.num_labels2=num_labels2
        self.bert = NeZhaModel.from_pretrained(model_name)
        self.attn1=Attn(config.hidden_size)
        self.attn2=Attn(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropouts=nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1,0.5,5)])
        self.classifier1 = nn.Linear(config.hidden_size, self.num_labels1)
        self.classifier2 = nn.Linear(config.hidden_size, self.num_labels2)

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

#         pooled_output1 = self.dropout(att1)
        pooled_output1=torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
#         pooled_output2 = self.dropout(att2)
        pooled_output2=torch.stack([ dd(att2)for dd in self.dropouts],dim=0).mean(dim=0)

        logits1 = self.classifier1(pooled_output1)
        logits2 = self.classifier2(pooled_output2)

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


# In[5]:


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


# In[6]:

tokenizer=BertTokenizerFast('../model_weight/nezha/vocab.txt')
test_set = CustomDataset(test, maxlen=128, tokenizer=tokenizer,with_labels=False)
test_loader = Data.DataLoader(test_set, batch_size=batch_size, num_workers=5, shuffle=False)

train_set = CustomDataset(train, maxlen=128,tokenizer=tokenizer)
train_loader = Data.DataLoader(train_set, batch_size=batch_size, num_workers=5, shuffle=True)

test_oof1=[]
for f in range(5):
    print('-------load  nezhav2_%d.pth ---------------------'%f)
    model=torch.load('../tmp/nezhav2_%d.pth'%f)
    all_loss=0
    model.eval()
    model.to(device)
    for i, batch in enumerate(tqdm(train_loader)):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            loss, predict1,predict2 = model(batch[0], batch[1], batch[2],batch[3])
            all_loss+=loss.item()/len(train_loader)
    print('nezhav2_%d train loss :%f'%(f,all_loss))
    test_oof1.append(to_predict(model,test_loader,with_labels=False))
    del model
    torch.cuda.empty_cache()
    gc.collect()


# In[7]:


class NeZhaForSequenceClassification(NeZhaPreTrainedModel):
    def __init__(self, config,model_name,num_labels1,num_labels2):
        super().__init__(config)
        self.num_labels1 = num_labels1
        self.num_labels2=num_labels2
        self.bert = XLNetModel(config).from_pretrained(model_name)
        
        self.attn1=Attn(config.hidden_size)
        self.attn2=Attn(config.hidden_size)
        self.attn3=Attn(config.hidden_size)
        self.attn4=Attn(config.hidden_size)
        self.dropouts=nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1,0.3,3)])
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, self.num_labels1-10)
        self.classifier3 = nn.Linear(config.hidden_size, 10)
        self.classifier2 = nn.Linear(config.hidden_size, self.num_labels2-9)
        self.classifier4 = nn.Linear(config.hidden_size,9)
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

        att1=self.attn1(outputs[0])
        att2=self.attn2(outputs[0])
        att3=self.attn3(outputs[0])
        att4=self.attn4(outputs[0])
        pooled_output1 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output2 = torch.stack([ dd(att2)for dd in self.dropouts],dim=0).mean(dim=0)#self.dropout(att2)
        pooled_output3 = torch.stack([ dd(att3)for dd in self.dropouts],dim=0).mean(dim=0)#self.dropout(att3)
        pooled_output4 = torch.stack([ dd(att4)for dd in self.dropouts],dim=0).mean(dim=0)#self.dropout(att4)
        logits1 = self.classifier1(pooled_output1)
        logits2 = self.classifier2(pooled_output2)
        logits3 = self.classifier3(pooled_output3)
        logits4 = self.classifier4(pooled_output4)
        
        logits1=torch.cat([logits1,logits3],dim=-1)
        logits2=torch.cat([logits2,logits4],dim=-1)
        
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
#         print(outputs)
        return outputs  # (loss), predict1,predict2, (hidden_states), (attentions)


# In[8]:
tokenizer=BertTokenizerFast('../model_weight/xlnet/vocab.txt')
test_set = CustomDataset(test, maxlen=128, tokenizer=tokenizer,with_labels=False)
test_loader = Data.DataLoader(test_set, batch_size=batch_size, num_workers=5, shuffle=False)

train_set = CustomDataset(train, maxlen=128,tokenizer=tokenizer)
train_loader = Data.DataLoader(train_set, batch_size=batch_size, num_workers=5, shuffle=True)

test_oof2=[]
for f in range(5):
    print('-------load  xlnetv2_%d.pth ---------------------'%f)
    model=torch.load('../tmp/xlnetv2_%d.pth'%f)    
    all_loss=0
    model.eval()
    model.to(device)
    for i, batch in enumerate(tqdm(train_loader)):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            loss, predict1,predict2 = model(batch[0], batch[1], batch[2],batch[3])
            all_loss+=loss.item()/len(train_loader)
    print('xlnetv2_%d train loss :%f'%(f,all_loss))
    test_oof2.append(to_predict(model,test_loader,with_labels=False))
    del model
    torch.cuda.empty_cache()
    gc.collect()


# In[9]:


class NeZhaForSequenceClassification(NeZhaPreTrainedModel):
    def __init__(self, config,model_name,num_labels1,num_labels2):
        super().__init__(config)
        self.num_labels1 = num_labels1
        self.num_labels2=num_labels2
        self.bert = MPNetModel.from_pretrained(model_name)
        
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
        self.dropouts=nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1,0.4,3)])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.classifier1 = nn.Linear(config.hidden_size, self.num_labels1-10)#3,4,5,6,7,8,9,
        self.classifier2 = nn.Linear(config.hidden_size,6) #10,11,13,14,15,17
        self.classifier3 = nn.Linear(config.hidden_size, 1)#1
        self.classifier4 = nn.Linear(config.hidden_size, 1) #2
        self.classifier5 = nn.Linear(config.hidden_size, 1)#12 
        self.classifier6 = nn.Linear(config.hidden_size, 1) #16


        self.classifier7 = nn.Linear(config.hidden_size, self.num_labels2-7) #7,8,9,10,11
        self.classifier8 = nn.Linear(config.hidden_size,4) #2,3,4,6
        self.classifier9= nn.Linear(config.hidden_size,1) #1
        self.classifier10= nn.Linear(config.hidden_size,1) # 5
        self.classifier11= nn.Linear(config.hidden_size,1) # 12
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
        
        pooled_output1 = self.dropout(att1)
        pooled_output2 = torch.stack([ dd(att2)for dd in self.dropouts],dim=0).mean(dim=0)#self.dropout(att2)
        pooled_output3 =torch.stack([ dd(att3)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output4 = torch.stack([ dd(att4)for dd in self.dropouts],dim=0).mean(dim=0)#self.dropout(att4)
        pooled_output5 = torch.stack([ dd(att5)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output6 = torch.stack([ dd(att6)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output7 = self.dropout(att7)
        pooled_output8 = self.dropout(att8)
        pooled_output9 = torch.stack([ dd(att9)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output10 = torch.stack([ dd(att10)for dd in self.dropouts],dim=0).mean(dim=0)
        pooled_output11 =torch.stack([ dd(att11)for dd in self.dropouts],dim=0).mean(dim=0)
        
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
     
        
        logits1=torch.cat([logits3,logits4,logits1,logits2[:,:2],logits5,logits2[:,2:5],logits6,logits2[:,5:]],dim=-1)

        logits2=torch.cat([logits9,logits8[:,:-1],logits10,logits8[:,-1:],logits7,logits11],dim=-1)
        
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

        return outputs  # (loss), predict1,predict2, (hidden_states),


# In[10]:

tokenizer=BertTokenizerFast('../model_weight/mpnet/vocab.txt')
test_set = CustomDataset(test, maxlen=128, tokenizer=tokenizer,with_labels=False)
test_loader = Data.DataLoader(test_set, batch_size=batch_size, num_workers=5, shuffle=False)

train_set = CustomDataset(train, maxlen=128,tokenizer=tokenizer)
train_loader = Data.DataLoader(train_set, batch_size=batch_size, num_workers=5, shuffle=True)
test_oof3=[]
for f in range(5):
    print('-------load  mpnetv2_%d.pth ---------------------'%f)
    model=torch.load('../tmp/mpnetv2_%d.pth'%f)    
    all_loss=0
    model.eval()
    model.to(device)
    for i, batch in enumerate(tqdm(train_loader)):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            loss, predict1,predict2 = model(batch[0], batch[1], batch[2],batch[3])
            all_loss+=loss.item()/len(train_loader)
    print('mpnetv2_%d train loss :%f'%(f,all_loss))
    test_oof3.append(to_predict(model,test_loader,with_labels=False))
    del model
    torch.cuda.empty_cache()
    gc.collect()


# In[20]:


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


# In[21]:
tokenizer=BertTokenizerFast('../model_weight/nezha/vocab.txt')
test_set = CustomDataset(test, maxlen=128, tokenizer=tokenizer,with_labels=False)
test_loader = Data.DataLoader(test_set, batch_size=batch_size, num_workers=5, shuffle=False)

train_set = CustomDataset(train, maxlen=128,tokenizer=tokenizer)
train_loader = Data.DataLoader(train_set, batch_size=batch_size, num_workers=5, shuffle=True)

test_oof4=[]
for f in range(5):
    print('-------load  nezhav3_%d.pth ---------------------'%f)
    model=torch.load('../tmp/nezhav3_%d.pth'%f)    
    all_loss=0
    model.eval()
    model.to(device)
    for i, batch in enumerate(tqdm(train_loader)):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            loss, predict1,predict2 = model(batch[0], batch[1], batch[2],batch[3])
            all_loss+=loss.item()/len(train_loader)
    print('nezhav3_%d train loss :%f'%(f,all_loss))
    test_oof4.append(to_predict(model,test_loader,with_labels=False))
    del model
    torch.cuda.empty_cache()
    gc.collect()


# In[13]:


# class NeZhaForSequenceClassification(NeZhaPreTrainedModel):
#     def __init__(self, config,model_name,num_labels1,num_labels2):
#         super().__init__(config)
#         self.num_labels1 = num_labels1
#         self.num_labels2=num_labels2
#         self.bert = NeZhaModel.from_pretrained(model_name)
#         self.attn1=Attn(config.hidden_size)
#         self.attn2=Attn(config.hidden_size)
#         self.attn3=Attn(config.hidden_size)
#         self.attn4=Attn(config.hidden_size)
#         self.attn5=Attn(config.hidden_size)
#         self.attn6=Attn(config.hidden_size)
#         self.attn7=Attn(config.hidden_size)
#         self.attn8=Attn(config.hidden_size)
    
#         self.dropouts=nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1,0.5,3)])
        
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier1 = nn.Linear(config.hidden_size, self.num_labels1-12)
#         self.classifier3 = nn.Linear(config.hidden_size, 6)
#         self.classifier5 = nn.Linear(config.hidden_size, 6)
        
#         self.classifier2 = nn.Linear(config.hidden_size, self.num_labels2-10)
#         self.classifier4 = nn.Linear(config.hidden_size,2)
#         self.classifier6 = nn.Linear(config.hidden_size,2)
#         self.classifier7 = nn.Linear(config.hidden_size,3)
#         self.classifier8 = nn.Linear(config.hidden_size,3)
        
#         self.predict=nn.Sigmoid()
# #         self.init_weights()
# #         if True:
# #             for p in self.bert.parameters(): # 冻结所有bert层
# #                 p.requires_grad = False

#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             labels1=None,
#         labels2=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for computing the sequence classification/regression loss.
#             Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
#             If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
#             If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

#     Returns:
#         :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
#         loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
#             Classification (or regression if config.num_labels==1) loss.
#         logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
#             Classification (or regression if config.num_labels==1) scores (before SoftMax).
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.

#     Examples::

#         from transformers import BertTokenizer, BertForSequenceClassification
#         import torch

#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

#         input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
#         labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
#         outputs = model(input_ids, labels=labels)

#         loss, logits = outputs[:2]

#         """
        
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#         )

# #         pooled_output = outputs[1]

#         att1=self.attn1(outputs[0])
#         att2=self.attn2(outputs[0])
#         att3=self.attn3(outputs[0])
#         att4=self.attn4(outputs[0])
#         att5=self.attn5(outputs[0])
#         att6=self.attn6(outputs[0])
#         att7=self.attn7(outputs[0])
#         att8=self.attn8(outputs[0])
       
        
#         pooled_output1 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
#         pooled_output2 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
#         pooled_output3 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
#         pooled_output4 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
#         pooled_output5 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
#         pooled_output6 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
#         pooled_output7 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
#         pooled_output8 = torch.stack([ dd(att1)for dd in self.dropouts],dim=0).mean(dim=0)
      
#         logits1 = self.classifier1(pooled_output1)
#         logits2 = self.classifier2(pooled_output2)
#         logits3 = self.classifier3(pooled_output3)
#         logits4 = self.classifier4(pooled_output4)
#         logits5 = self.classifier5(pooled_output5)
#         logits6 = self.classifier6(pooled_output6)
#         logits7 = self.classifier7(pooled_output7)
#         logits8 = self.classifier8(pooled_output8)
     
        
#         logits1=torch.cat([logits1,logits3,logits5],dim=-1)
#         logits2=torch.cat([logits2,logits4,logits6,logits7,logits8],dim=-1)

#         predict1=self.predict(logits1)
#         predict2=self.predict(logits2)
#         outputs = (predict1,predict2) + outputs[2:]  # add hidden states and attention if they are here
# #         print('label:',labels)
# #         print('input_ids:',input_ids)
# #         print('attention_mas:',attention_mask)
#         if labels1 is not None:
#             loss_fct = nn.BCELoss()
# #                 print(logits.view(-1, self.num_labels))
# #                 print(labels.view(-1, self.num_labels))
#             loss1 = loss_fct(predict1.view(-1, self.num_labels1), labels1.view(-1, self.num_labels1))
#             loss2 = loss_fct(predict2.view(-1, self.num_labels2), labels2.view(-1, self.num_labels2))
#             loss=loss1+loss2
#             outputs = (loss,) + outputs

#         return outputs  # (loss), predict1,predict2, (hidden_states), (attentions)


# In[17]:


# test_oof5=[]
# for f in range(8):
#     print('-------load  nezhav3_1_%d.pth ---------------------'%f)
#     model=torch.load('../tmp/nezhav3_1_%d.pth'%f)    
#     all_loss=0
#     model.eval()
#     model.to(device)
#     for i, batch in enumerate(tqdm(train_loader)):
#         batch = tuple(t.to(device) for t in batch)
#         with torch.no_grad():
#             loss, predict1,predict2 = model(batch[0], batch[1], batch[2],batch[3])
#             all_loss+=loss.item()/len(train_loader)
#     print('nezhav3_1_%d train loss :%f'%(f,all_loss))
#     test_oof5.append(to_predict(model,test_loader,with_labels=False))
#     del model
#     torch.cuda.empty_cache()
#     gc.collect()


# In[19]:


#----  deep 模型
deep1=np.load('../deep_v1.npy')
# deep1=pd.read_csv('../result_deep_v1.csv',header=None)
# tmp=deep1[1].apply(lambda x: np.array(list(map(lambda x:float(x),x.replace('|','').strip().split(' ')))))
# tmp=np.vstack(tmp.to_list()) # 深度


# In[18]:


sub1=np.mean(test_oof1,axis=0) # nezhav2
sub2=np.mean(test_oof2,axis=0) # xlnetv2
sub3=np.mean(test_oof3,axis=0) # mpnetv2
sub4=np.mean(test_oof4,axis=0) # nezhav3
# sub5=np.mean(test_oof5,axis=0) # nezhav3_1
# deep1深度

test_oof=sub1*0.25+0.3*sub2+sub3*0.2+sub4*0.1+deep1*0.15


# In[17]:


sub=pd.DataFrame()
# test=pd.read_csv('../tcdata/track1_round1_testB.csv',header=None)
test=pd.read_csv('../tcdata/testB.csv',header=None)
sub['report_ID']=test[0]
sub['Prediction']=[ '|'+' '.join(['%.12f'%j for j in i]) for i in test_oof ]
sub.to_csv('../result.csv',index=False,header=False)


# In[ ]:




