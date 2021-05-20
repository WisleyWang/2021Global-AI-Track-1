import pandas as pd
import numpy as np
import collections

train=pd.read_csv('./track1_round1_train_20210222.csv',header=None)
test=pd.read_csv('./track1_round1_testB.csv',header=None)

def ensemble_fun(ensembles):
    cob=0
    # ensembles=[sub1,sub2,sub3,sub4,sub5]
    for ss in ensembles:
        cob+=ss[1].apply(lambda x: np.array(list(map(lambda x:float(x),x.replace('|','').strip().split(' ')))))/len(ensembles)
    return cob

## 融合
sub1=pd.read_csv('../prediction_result/B_vv1.csv',header=None)
sub2=pd.read_csv('../prediction_result/B_v1.csv',header=None)
sub3=pd.read_csv('../prediction_result/B_TextAttBi.csv',header=None)
sub4=pd.read_csv('../prediction_result/B_Huawei_v4.csv',header=None)
sub5=pd.read_csv('../prediction_result/B_nezha_vv1.csv',header=None)
sub6=pd.read_csv('../prediction_result/B_nezha_v1.csv',header=None)
# sub7=pd.read_csv('./data/enhencev2_huawei_5fold2_0.0.897.csv',header=None)
# sub8=pd.read_csv('./data/drop2_lookahead_huawei_5fold_0.892.csv',header=None)
# sub9=pd.read_csv('./data/mult_loss3_dence_0.897.csv',header=None)
# sub10=pd.read_csv('./data/mult_loss2_dence_0.896.csv',header=None)
cob=ensemble_fun([sub1,sub2,sub3,sub4,sub5,sub6])

sub=pd.DataFrame()
sub['report_ID']=test[0]
sub['Prediction']=[ '|'+' '.join(['%.12f'%j for j in i]) for i in cob]
sub.to_csv('../prediction_result/result.csv',index=False,header=False)