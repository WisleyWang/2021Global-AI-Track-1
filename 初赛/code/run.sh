#!/bin/sh

pip install -r requirements.txt -i https:/pypi.douban.com/simple/

echo "training deep_mode1------."
python3 ./deep_model1.py
echo "training deep_model2-----"
python3 ./deep_model2.py
echo "training Nezha_deep_v1-----"
python3 ./Nezha_deep_v1.py
echo "ensemble...."
python3 ./ensemble.py

echo "finish!"
