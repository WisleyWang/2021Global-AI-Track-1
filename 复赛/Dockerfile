#e Images 
## 从天池基础镜像构建(from的base img 根据自己的需要更换，建议使用天池open list镜像链接：https://tianchi.aliyun.com/forum/postDetail?postId=67720) 
FROM  registry.cn-shanghai.aliyuncs.com/tcc-public/tensorflow:2.4.1-cuda10.1-py3
##安装python依赖包 
RUN pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple \
&& pip install tensorflow-gpu==2.4.1 -i https://pypi.douban.com/simple \
&& pip install gensim==3.8.3 -i https://pypi.douban.com/simple \
&& pip install pandas -i https://pypi.douban.com/simple \
&& pip install sklearn -i https://pypi.douban.com/simple \
&& pip install keras==2.2.4 -i https://pypi.douban.com/simple \
&& pip install --upgrade pip \
&& pip install tqdm  -i https://pypi.douban.com/simple \
&& pip install torch  -i https://pypi.douban.com/simple \
&& pip install sentencepiece -i https://pypi.douban.com/simple \ 
&& pip install transformers==4.4.2
## 把当前文件夹里的文件构建到镜像的根目录下,并设置为默认工作目录 
ADD . / 
WORKDIR / 
## 镜像启动后统一执行 sh run.sh 
# CMD ["sh", "run.sh"]
