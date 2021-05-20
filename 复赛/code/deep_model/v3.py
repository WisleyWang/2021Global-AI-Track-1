#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import keras
import keras.backend as K
from keras import Model
from keras import initializers, regularizers, constraints
from keras.layers import Layer
from keras.layers import Embedding, Dense, CuDNNLSTM,CuDNNGRU, Bidirectional,SpatialDropout1D,Input,GlobalAveragePooling1D,GlobalMaxPooling1D,Conv1D,concatenate,Dropout,Activation,BatchNormalization,Concatenate,Add,MaxPooling1D,Flatten,AveragePooling1D
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
import gensim, logging
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import KFold,StratifiedShuffleSplit,StratifiedKFold
from keras.utils import multi_gpu_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[52]:


train=pd.read_csv('../tcdata/train.csv',header=None)
# test=pd.read_csv('../tcdata/track1_round1_testB.csv',header=None)
test=pd.read_csv('../tcdata/testA.csv',header=None)
data=pd.concat([train,test],axis=0)
data[1]=data[1].apply(lambda x:x.strip().replace('|',''))


# In[53]:


## 制作标签
### 创建训练集标签 
train_labels1=np.zeros((len(train),17),dtype='int8')
noncase=0
for cnt,i in enumerate(train[2]):
    if i:
        lab=[int(x.replace('|','').strip()) for x in i.split(' ') if x and x!='|']
        for l in lab:
            train_labels1[cnt,l]=1
    else:
        noncase+=1
print('noncase label1:',noncase)
#----------------------------------
noncase=0
train_labels2=np.zeros((len(train),12),dtype='int8')
for cnt,i in enumerate(train[3]):
    if pd.notna(i):
        lab=[int(x.replace('|','').strip()) for x in i.split(' ') if x and x!='|']
        for l in lab:
            train_labels2[cnt,l]=1
    else:
        noncase+=1
print('noncase label2:',noncase)


# In[54]:


cate_num=train_labels1.sum(1)+train_labels2.sum(1)
train_labels=np.concatenate([train_labels1,train_labels2],axis=1)


# In[4]:


## 生成全部的训练文本
data1=pd.read_csv('../tcdata/track1_round1_train_20210222.csv',header=None)
data2=pd.read_csv('../tcdata/track1_round1_testA_20210222.csv',header=None)
data3=pd.read_csv('../tcdata/track1_round1_testB.csv',header=None)
text_data=pd.concat([train,test,data1,data2,data3],axis=0)
text_data[1]=text_data[1].apply(lambda x:x.strip().replace('|',''))
text_data[1].to_csv('../tmp/all_data.txt',header=False,index=False)


# In[12]:


text_data=pd.read_csv('../tmp/all_data.txt',header=None)


# In[6]:


w2v=Word2Vec(text_data[0].apply(lambda x:x.split(' ')).tolist(),size=128, window=8, iter=30, min_count=2,
                     sg=1, sample=0.002, workers=6 , seed=1017)

w2v.wv.save_word2vec_format('../tmp/w2v_128.txt')


# In[55]:


tokenizer = Tokenizer(lower=False, char_level=False, split=' ')
tokenizer.fit_on_texts(data[1].tolist())
seq = tokenizer.texts_to_sequences(data[1].tolist())
# 分训练和测试集合
seq = pad_sequences(seq, maxlen=128, value=0)
train_seq=np.asarray(seq[:len(train)])
test_seq=seq[len(train):]


# In[56]:


embedding_matrix=np.zeros((len(tokenizer.word_index)+1,128))
w2v=gensim.models.KeyedVectors.load_word2vec_format(
        '../tmp/w2v_128.txt', binary=False)

for word in tokenizer.word_index:
    if word not in w2v.wv.vocab:
        continue
    embedding_matrix[tokenizer.word_index[word]] = w2v[word]
embedding_matrix.shape


# In[57]:


from keras.layers import Lambda,Reshape


# In[58]:


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
#         print('-------------',type(input_shape))
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name),
                                     shape=(input_shape[1],),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
       
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        c=a * x
        c = K.sum(c, axis=1,keepdims=True)
        return c
    def compute_output_shape(self, input_shape):
        output_shape=list(input_shape)
        output_shape[1]=1
        return tuple(output_shape)
def NN_huaweiv1(maxlen,embedding_matrix=None,class_num1=17,class_num2=12):    
    emb_layer = Embedding(
       embedding_matrix.shape[0], embedding_matrix.shape[1],input_length=maxlen,weights=[embedding_matrix],trainable=False,
    )
    seq1 = Input(shape=(maxlen,)) 
    
    x1 = emb_layer(seq1)
    sdrop=SpatialDropout1D(rate=0.2)
    lstm_layer = Bidirectional(CuDNNGRU(128, return_sequences=True))
    gru_layer = Bidirectional(CuDNNGRU(128, return_sequences=True))
    cnn1d_layer=Conv1D(64, kernel_size=3, padding="same", kernel_initializer="he_uniform")
    x1 = sdrop(x1)
    lstm1 = lstm_layer(x1)
    gru1 = gru_layer(lstm1)
    att_list=[]
    for i in range(29):
        att_list.append(Attention(maxlen)(gru1))
    x=concatenate(att_list,axis=1)
    x=Activation(activation="relu")(BatchNormalization()(Dense(64)(x)))
    x=Dense(1,activation='sigmoid')(x)
    pred=Lambda(lambda x:x[:,:,0])(x)
    model = Model(inputs=seq1, outputs=pred)
    return model

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim
def mean_pred(y_true, y_pred):
    return -K.mean(y_true*K.log(y_pred+1.e-7)+(1-y_true)*K.log(1-y_pred+1.e-7))*10


# In[62]:


batch_size=128
epochs=100
weight_name='V3'
oof=np.zeros((len(train),29))
tmp=0
test_oof=np.zeros((len(test),29))
# for ii in range(17):
#     per_labels=train_labels[:,ii]
#     print('当前分类类别：'+str(ii))
#     print('正样本比例:'+str(sum(per_labels)/len(per_labels)))
# per_labels=train_labels[:,1]
folds=StratifiedKFold(n_splits=8,shuffle=True, random_state=2018) #2018
for fold_n, (trn_idx, val_idx) in enumerate(folds.split(train,cate_num)):
    print('Fold', fold_n)
    print('Build model...')
#     print('正样本比例:',train_labels[trn_idx].mean(0))
    model=NN_huaweiv1(128,embedding_matrix=embedding_matrix,class_num1=17,class_num2=12)
#     model=multi_gpu_model(model,gpus=2)
    model.compile('adam', ['binary_crossentropy'], metrics=['accuracy',mean_pred])

    print('Train...')
    early_stopping = EarlyStopping(monitor='val_loss', patience=4,mode='min')
    check_point=ModelCheckpoint('../model_weight/%s_%d.h5'%(weight_name,fold_n),monitor='val_loss',verbose=1, save_best_only=True,save_weights_only=True)

    model.fit(train_seq[trn_idx],train_labels[trn_idx],
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[early_stopping,check_point],
              validation_data=(train_seq[val_idx],train_labels[val_idx]))
    model.load_weights('../model_weight/%s_%d.h5'%(weight_name,fold_n))
    oof[val_idx,:] = model.predict(train_seq[val_idx],batch_size=batch_size)
    test_oof[:,:]+=model.predict(test_seq,batch_size=batch_size)/folds.n_splits


# In[81]:


sub=pd.DataFrame()
sub['report_ID']=test[0]
sub['Prediction']=[ '|'+' '.join(['%.12f'%j for j in i]) for i in test_oof ]
sub.to_csv('../result.csv',index=False,header=False)


# In[ ]:




