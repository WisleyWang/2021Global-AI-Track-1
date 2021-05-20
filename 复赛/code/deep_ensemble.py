#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


def ensemble_fun(ensembles):
    cob=0
    weight=[0.35,0.35,0.1,0.2]
    cnt=0
    for ss in ensembles:
        cob+=ss[1].apply(lambda x: np.array(list(map(lambda x:float(x),x.replace('|','').strip().split(' ')))))*weight[cnt]
        cnt+=1
#     cob=cob.apply(lambda x:x-x.min())
    return cob


# In[4]:



## 融合
sub1=pd.read_csv('../result_nezhav3.csv',header=None)
sub2=pd.read_csv('../result_nezhav2.csv',header=None)
sub3=pd.read_csv('../result_mpnet.csv',header=None)
sub4=pd.read_csv('../result_xlnetv2.csv',header=None)
# sub5=pd.read_csv('./data/textrnn_sub1_0.868.csv',header=None)
c=[sub1,sub2,sub3,sub4]
cob=ensemble_fun(c)
test=pd.read_csv('../tcdata/testA.csv',header=None)
# test=pd.read_csv('../tcdata/track1_round1_testB.csv',header=None)

sub=pd.DataFrame()
sub['report_ID']=test[0]
sub['Prediction']=[ '|'+' '.join(['%.12f'%j for j in i]) for i in cob]
sub.to_csv('../result.csv',index=False,header=False)


# In[ ]:




