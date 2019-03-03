# LowPrecision

该项目为低精度相关的工作，主要是尝试对于NMT系统的decoder部分，尝试用更低精度的单位，如float16，int，int8等
来替换原来神经网络中常用的float32的单位，来提高运行速度，同时节省内存。

---

## 目录

目录 | 内容
---|---
/BLUE |                     测试BLUE值的python脚本
/LinerSoftmax |             基于Neu.Tensor项目，在FNNLM的语言模型上实验低精度分类器的源码
/LowPrecision/src |	        编写CPU环境下半精度（float16）与其他精度单位的转换与计算源码
/NN/src |	                  FNNLM源码
/Note |	                    初期笔记，gpu低精度编程问题记录
/NumberVerification/src |	  MINST识别源码
/RNN&Attention |	          基于一个RNN&Attention的seq2seq翻译模型上测试低精度decoder的源码
/fairseq_work |	            基于facebook的fairseq项目翻译模型上测试低精度decoder的源码
/paper |                    相关论文
presentation.pdf |          汇报

---

该项目下整理了 于2019年1月-2019年3月在 东北大学小牛翻译实验室 实习的主要内容，感谢实验室的各位老师，学长学姐，同学的大力帮助，特别感谢肖桐老师和林野学姐对我实验的指点和帮助

try something

I love NLP.

