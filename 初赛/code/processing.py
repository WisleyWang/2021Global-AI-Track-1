import pandas as pd

data_path='../tcdata/medical_nlp_round1_data/'
train=pd.read_csv(data_path+'track1_round1_train_20210222.csv',header=None)
#把a，b榜测试文件放一起，用以生成预训练文本
test1=pd.read_csv(data_path+'track1_round1_testB.csv',header=None)
test2=pd.read_csv(data_path+'track1_round1_testA_20210222.csv',header=None)

data=pd.concat([train,test1,test2],axis=0).reset_index(drop=True)
# 去除竖线
data[1]=data[1].apply(lambda x:x.replace('|','').strip())

# 保存预训练文本:
data[1].to_csv('../user_data/tmp_data/all_data_txt.txt', sep=' ', index=False,header=False)