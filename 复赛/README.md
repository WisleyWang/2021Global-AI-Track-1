# 全球人工智能技术创新大赛【赛道一】
## 复赛
### 文件夹目录结构
```
|-- code
    |-- deep_model # 沿用初赛的deep模型
        |-- v1.py
        |-- ....
    |-- log # 微调时记录val loss 
        |-- ...
    |--model  # nezha模型的代码
        |-- configuration_nezha.py
        |-- modeling_nezha.py
    |-- pretrain  # 预训练各类bert模型代码
        |-- data #预训练用到的数据
        |-- model # nezha模型代码
            |-- configuration_nezha.py
            |-- modeling_nezha.py
        |-- mpnet_model # mpnet模型config
        |-- nezha_mode # nezha模型config
        |-- xlnet_model # xlnet模型config
        |-- output # 预训练时输出文件
            |-- model # 词典
            |-- record # 预训练时，各step保存.bin文件
        |-- runs #GPU运训缓存记录
        |-- mpnet_pretrain.py # mpnet预训练代码
        |-- nezha_pretrain.py # nezha预训练代码
        |-- xlnet_pretrain.py # xlnet预训练代码
    |-- ensemble.py # 融合
    |-- inference.py #线上直接推断代码
    |-- MPNet_v2.py # mpnet微调代码
    |-- nezha_v2.py # nezha微调代码 v2
    |-- nezha_v3.py # nezha微调代码v3
    |-- xlnet_v2.py # xlnet微调代码

|-- model_weight  #预训练后模型保存
    |-- mpnet
        |--config.json
        |-- vocab.txt
        |-- pytorch_model.bin 
    |-- nezha
        |-- ...
    |-- xlnet 
        |-- ...

|-- tcdata  # 复赛数据
	|-- train.csv  
	|-- track1_round1_testA_20210222.csv
    |-- track1_round1_testB.csv
    |-- track1_round1_train_20210222.csv
    |-- ...
	
|-- tcdata
    |-- medical_nlp_round1_data # 存放比赛数据
        |-- track1_round1_testA_20210222.csv
	    |-- track1_round1_testB.csv
	    |-- track1_round1_train_20210222.csv
|-- tmp # 存储临时文件
    |-- w2v_128.txt # w2v模型参数
    |-- ...

|-- README.md #解决方案及算法介绍文件
```
### 说明
复赛基本全部采用bert模型，deep模型线上0.925后就未继续尝试了
后面通过FDG 等手段和调参后，线上分数应该更高些，但是比赛快结束了，测试次数不够，未知线上分数。

| 模型 | 线上B榜分数 | 预训练loss|
| :---:  | :-----:  | :--:|
| nezha  | 0.936  | 0.31|
| mpnet  |0.932   | 0.3 |
| xlnet  |0.934   | 0.7|
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


- 预训练

 nezha 采用ngram 的mlm预训练

mpnet 采用 mlm预训练

xlnet 采用plm 预训练



- 微调 

微调一般4~5epoch
线下val loss 可以达到0.035  线上评估指标是1-mlogloss

- 融合

由于操作失误，只融了nezhav2和xlnet模型，而且不是最佳线下参数的，线上B榜0.939。
理论上融合 nezhav2+nezhav3+xlnet+mpnet+deepmodel 收益会很高。但是b榜要求线上全流程，最后几次出了bug就没机会了，**也算是一个教训**吧，前期花太多时间在单模测试上，没机器跑，测试一个方案要跑一天多，每天3次机会都没用完。后面比赛完了解到前排很多单模分数不高，但是融合收益高很多的。另外我们在预训练的时间上也花的比较久，据了解，有些预训练就几小时，loss=1.2的，最后通过微调效果也不错。我们没在预训练上做过多的尝试，也是各失误的地方。


>  在预训练阶段，发现不加载权重，loss才能绛下去。另外不同的预训练模式也能带来较大是收益。

最终复赛B榜线上排名17