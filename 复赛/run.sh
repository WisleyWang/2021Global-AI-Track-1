#!/bin/sh
python ./code/pretrain/xlnet_pretrain.py
python ./code/pretrain/nezha_pretrain.py
python ./code/pretrain/mpnet_pretrain.py

python ./code/deep_model/v1.py
# python ./code/v2.py
# python ./code/v4.py
# python ./code/deep_ensemble.py
# python ./code/nezha_v2_smart.py
# python ./code/nezha_v1.py


python ./code/MPNet_v2.py

# python ./code/nezha_v3.py
python ./code/nezha_v2.py
python ./code/nezha_v3.py
python ./code/xlnet_v2.py
# 
# python ./code/deep_ensemble.py
# python ./code/bert_v1.py
python ./code/inference.py