## Fairseq 模型的 低精度分类器测试

在FaceBook的fairseq翻译模型上尝试修改 
 
这里直接采用它提供的预先训练的模型进行测试，直接修改其最后 的decoder输出部分的分类器 

基线模型地址：https://github.com/pytorch/fairseq

修改的全部代码在 fairseq 中

项目部署markdown ： fairseq.md

项目修改主线 markdown： fairseq_modify.md

测试代码主线 markdown： test_code.md

测试结果

### Pre-trained models  测试模型

Description | Dataset | Model | Test set(s)
---|---|---|---
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.newstest2014.tar.bz2) <br> newstest2012/2013: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.ntst1213.tar.bz2)
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-German](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-de.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-de.newstest2014.tar.bz2)
Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2)
Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)


#### 测试结果
Model | Origin BLUES | Now BLUES | Percent | Nom Before Softmax
---|---|---|---|---
CNN - WMT14 English-French | 40.83 | 40.08 | 98.163% | 40.09
CNN - WMT14 English-German | 25.70 | 24.80 | 96.498% | 24.81
Transformer - WMT14 English-French | 43.00 | 42.26 | 98.279% | 42.29
Transformer - WMT16 English-German | 29.23 | 28.68 | 98.118% | 28.69 
