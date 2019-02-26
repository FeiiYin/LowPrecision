## fairseq 环境搭建

项目地址：https://github.com/pytorch/fairseq

项目版本要求 ： python = 3.6， pytorch >= 1.0.0, 训练要求: NVIDIA GPU and NCCL

#### Anaconda 环境搭建

参考博客：https://blog.csdn.net/u011669700/article/details/79555095

+ 下载镜像，下载地址为：https://mirrors.tuna.tsinghua.edu.cn/。 

+ `chmod +x Anaconda3-5.1.0-Linux-x86_64.sh`

+ `bash Anaconda3-5.1.0-Linux-x86_64.sh`

+ 在服务器无root权限，~~需要修改conda的快捷路径 `export PATH=~/anaconda3/bin:$PATH`~~ 这个添加了错误的PATH， 删去在 ~/.bashrc 中错误路径

+ 参考博客：https://www.jianshu.com/p/cd0096b24b43

+ `source activate`  `source deactivate` 激活

#### 虚拟环境搭建

+ 创建 `conda create -n fairseq python=3.6`  fairseq为名称

+ 进入虚拟环境 `conda activate fairseq`, 不进就装在整个环境里了 = =

#### 查看CUDA版本

+ `cat /usr/local/cuda/version.txt`

#### pytorch 环境搭建

+ 卸载之前的 `conda uninstall pytorch`

+ 安装 `conda install pytorch=1.0.0 cuda90 -c pytorch`

+ 安装中出现网络问题，跟换下载目标为清华镜像 
  `conda install pytorch=1.0.0 cuda90 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch`

#### `CXXABI_1.3.8` not found 

+ 这是c++的环境，更新需要系统root权限，由于这个原因，更换了服务器，工作速度慢了一倍

+ 下载地址：https://download.csdn.net/download/d_bigwolf/10264318

+ 参考博客：https://blog.csdn.net/u012811841/article/details/77854581/

+ 检查是否缺少：`strings /usr/lib64/libstdc++.so.6 | grep 'CXXABI'`

+ 安装`libstdc++.so.6.0.24`

+ `cp libstdc++.so.6.0.24 /usr/lib64/`   拷贝到/usr/lib64目录下

+ `rm -rf libstdc++.so.6`   删除原来的libstdc++.so.6符号连接

+ `ln -s libstdc++.so.6.0.24 libstdc++.so.6`  新建新符号连接

注：`.so`文件是链接文件，类似于路径，在虚拟环境中像快捷方式一样

#### conda 速度慢问题

+ 将清华镜像添加到默认channel中， 无需每次都输入网址

+ `conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/`

+ `conda config --set show_channel_urls yes`

#### fairseq 其余环境配置

```
  conda install gcc numpy cudnn nccl
  conda install magma-cuda80 -c soumith
  pip install cmake
  pip install cffi
```
+ 安装fairseq-py

```
pip install -r requirements.txt
python setup.py build
python setup.py develop
```

#### 模型运行

参考文章：https://ptorch.com/news/58.html

+ 快速开始

* python preprocess.py：数据预处理：构建词汇和二值化培训数据

* python train.py：在一个或多个GPU上训练新的模型

* python generate.py：用训练有素的模型翻译预处理的数据

* python generate.py -i：用训练有素的模型翻译原始文本

* python score.py：BLEU生成的翻译与参考翻译的得分

+ 预先训练模型载入

参考文章:https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md

```
$ mkdir -p data-bin
$ curl https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf - -C data-bin
$ curl https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.newstest2014.tar.bz2 | tar xvjf - -C data-bin
$ fairseq-generate data-bin/wmt14.en-fr.newstest2014  \
  --path data-bin/wmt14.en-fr.fconv-py/model.pt \
  --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
...
| Translated 3003 sentences (96311 tokens) in 166.0s (580.04 tokens/s)
| Generate test with beam=5: BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)

# Compute BLEU score
$ grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
$ grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
$ fairseq-score --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref
BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)
```

或者

```
$ curl https://s3.amazonaws.com/fairseq-py/models/wmt14.en-fr.fconv-py.tar.bz2 | tar xvjf - -C data-bin
$ curl https://s3.amazonaws.com/fairseq-py/data/wmt14.en-fr.newstest2014.tar.bz2 | tar xvjf - -C data-bin
$ python generate.py data-bin/wmt14.en-fr.newstest2014  \
  --path data-bin/wmt14.en-fr.fconv-py/model.pt \
  --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
...
| Translated 3003 sentences (95451 tokens) in 81.3s (1174.33 tokens/s)
| Generate test with beam=5: BLEU4 = 40.23, 67.5/46.4/33.8/25.0 (BP=0.997, ratio=1.003, syslen=80963, reflen=81194)

# Scoring with score.py:
$ grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
$ grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
$ python score.py --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref
BLEU4 = 40.23, 67.5/46.4/33.8/25.0 (BP=0.997, ratio=1.003, syslen=80963, reflen=81194)
```
