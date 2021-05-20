# 全球人工智能技术创新大赛【赛道一】
## 初赛
### 文件夹目录结构
```
|-- code
    |-- model # NeZha 模型文件
        |-- configuration_nezha.py
        |-- modeling_nezha.py
    |-- processing.py # 预处理
    |--creat_embedding.py  # 预训练
    |-- deep_model1.py  # 模型1
    |-- deep_model2.py # 模型2
    |-- Nezha_deep_v1.py # 模型3
    |-- ensemble.py # 融合
    
|-- prediction_result  #预测结果
	|-- result.csv  # 最终提交结果
	|-- B_Huawei_v4.csv  # deep_model2预测结果
	|-- B_TextAttBi.csv  # deep_model2预测结果
	|-- B_nezha_vv1.csv  # Nezha_deep_v1预测结果
	|-- B_v1.csv  # deep_model1预测结果
	|-- B_vv1.csv  # deep_model1预测结果
	
|-- tcdata
    |-- medical_nlp_round1_data # 存放比赛数据
        |-- track1_round1_testA_20210222.csv
	    |-- track1_round1_testB.csv
	    |-- track1_round1_train_20210222.csv
	    
|-- user_data #过程数据
    |-- model_data
        |-- deep_model1 # 保存deep_model1的权重
        |-- deep_model2 # 保存deep_model2的权重
        |-- Nezha_deep # 保存deep_model2的权重
    |-- pretraining_model
        |-- nezha # 保存预训练微调后的模型
        |-- nezha-cn-base # 原始预训练模型
        |-- vocab.txt #词汇表
        |-- w2v_128.txt # w2v的训练模型
    |-- tmp_data
        |-- nezha_embedding.npy # 预训练微调后的embedding
        |-- nezha_seqB.npy  # nezha模型对应数据的input_id

|-- README.md #解决方案及算法介绍文件
|-- requirements.txt    # Python环境依赖
```
### 说明
初赛基本使用的是深度模型，Nezha模型只用到了预训练后的embedding,模型未保存，需重新训练，
平均每个模型训练耗时:1.5小时。
```
bash /code/run.sh # 可直接训练并得到预测结果
```
- 环境需求
torch环境:

```buildoutcfg
pytorch==1.7.1
tensorflow==2.4.1
scikit-learn==0.24.1
tokenizers==0.10.1
datasets==1.5.0
transformers==4.4.2
gensim==3.8.3
```
tensorflow | keras 环境:
```json
tensorflow-gpu==1.14.0
tokenizers==0.9.4
keras==2.2.4
scikit-learn==0.23.2
gensim==3.8.3

```


- 预处理

运行`/code/processing.py`  在 `/user_data/tmp_data` 下生成all_data_txt.txt
- 预训练

运行`/code/creat_embedding.py` 预训练w2v 在 `/user_data/pretraining_model`下生成`w2v_128.txt`

预训练Nezha，在`/user_data/pretraining_model/nezha`下生成模型权重

- Deep模型训练与预测

 运行`/code/deep_model1.py` 训练两个模型，在`/prediction_result/`下生成预测结果：`B_v1.csv`和`B_vv1.csv`

运行`/code/deep_model2.py` 训练两个模型，在`/prediction_result/`下生成预测结果：`B_TextAttBi.csv`和`B_Huawei_v4.csv`

运行`/code/Nezha_deep_v1` 训练两个模型，在`/prediction_result/`下生成预测结果：`B_nezha_vv1.csv`和`B_nezha_v1.csv`

- 融合

运行`/code/ensemble.py` 对六个预测结果进行平均加权融合,在`/prediction_result/`下生成`result.csv`

##最后提交与线下差异
由于时间关系，模型权重在比赛过程中没有保存，所以最后B榜时只预测出了5个模型结果如`/prediction_result/`中存放的5个结果。

>  注意：若要重新预训练Nezha模型，需要将模型下载并解压到`/user_data/pretraining_model/`目录下


最后B榜线上分数：0.909 |40

B榜最高分数：0.927112