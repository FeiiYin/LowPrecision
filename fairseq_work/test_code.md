## 关于fairseq项目测试的 服务器代码

### Pre-trained models  测试模型

Description | Dataset | Model | Test set(s)
---|---|---|---
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.newstest2014.tar.bz2) <br> newstest2012/2013: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.ntst1213.tar.bz2)
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-German](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-de.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-de.newstest2014.tar.bz2)
Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2)
Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)

#### 测试：这里均采用cpu上测试，因为当前修改项目的代码与部署在cpu环境中

#### 注意修改的代码， cpu 返回： return output

####                 gpu 返回： return output.cuda().data

测试模型 1： CNN - 英-法
```
fairseq-generate --cpu  data-bin/2/wmt14.en-de.newstest2014 \ 
  --path data-bin/2/wmt14.en-de.fconv-py/model.pt \
  --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
```

测试模型 2： CNN - 英-德
```
fairseq-generate --cpu  data-bin/2/wmt14.en-de.newstest2014 \ 
  --path data-bin/2/wmt14.en-de.fconv-py/model.pt \
  --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
```

测试模型 3： transformer - 英-法
```
fairseq-generate --cpu  data-bin/4/wmt14.en-fr.joined-dict.newstest2014 \
  --path data-bin/4/wmt14.en-fr.joined-dict.transformer/model.pt \
  --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
```

测试模型 4： transformer - 英-德
```
fairseq-generate --cpu  data-bin/2/wmt14.en-de.newstest2014 \ 
  --path data-bin/2/wmt14.en-de.fconv-py/model.pt \
  --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
```

#### 测试结果
Model | Origin BLUES | Now BLUES | Percent
---|---|---|---
CNN - WMT14 English-French | 40.83 | 40.08 | 98.163%
CNN - WMT14 English-German | 25.70 | 24.80 | 96.498%
Transformer - WMT14 English-French | 43.00 | 42.26 | 98.279%
Transformer - WMT16 English-German | 29.23 | 28.68 | 98.118%


