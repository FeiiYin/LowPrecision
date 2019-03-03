## 翻译模型 的低精度分类器测试

在一个基于RNN和Attention机制 Sequence to Sequence 的翻译模型上修改 

修改了decoder中输出的分类器 
 
decoder的输出会影响到训练的反向过程 
 
loss图像前后收敛过程相近
 
训练数据量 10599 
 
测试数据  1000 
 
（数据均为简单句子） 
 
测试BLUE值结果：          初始：32.83                           修改后：33.04 

由于数据太少，该数据不足以太让人信服

训练及样例输出

```
sys.path.extend(['D:\\python code\\RNNMT', 'D:/python code/RNNMT'])
Python 3.6.8 |Anaconda, Inc.| (default, Feb 11 2019, 15:03:47) [MSC v.1915 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.2.0 -- An enhanced Interactive Python. Type '?' for help.
PyDev console: using IPython 7.2.0
Python 3.6.8 |Anaconda, Inc.| (default, Feb 11 2019, 15:03:47) [MSC v.1915 64 bit (AMD64)] on win32
runfile('D:/python code/RNNMT/RNN/origin.py', wdir='D:/python code/RNNMT/RNN')
Reading lines...
Read 135842 sentence pairs
Trimmed to 10599 sentence pairs
Counting words...
Counted words:
fra 4345
eng 2803
['je me fais trop vieux pour ce travail .', 'i m getting too old for this job .']
Backend TkAgg is interactive backend. Turning interactive mode on.
13m 10s (- 184m 30s) (5000 6%) 2.8535
32m 15s (- 209m 42s) (10000 13%) 2.2660
52m 42s (- 210m 48s) (15000 20%) 1.9649
70m 54s (- 195m 1s) (20000 26%) 1.6854
88m 9s (- 176m 19s) (25000 33%) 1.4910
102m 11s (- 153m 17s) (30000 40%) 1.3456
119m 5s (- 136m 6s) (35000 46%) 1.2008
137m 57s (- 120m 42s) (40000 53%) 1.0822
154m 9s (- 102m 46s) (45000 60%) 0.9900
169m 50s (- 84m 55s) (50000 66%) 0.8647
183m 33s (- 66m 44s) (55000 73%) 0.7977
197m 35s (- 49m 23s) (60000 80%) 0.7327
210m 49s (- 32m 26s) (65000 86%) 0.6808
223m 31s (- 15m 57s) (70000 93%) 0.5891
236m 5s (- 0m 0s) (75000 100%) 0.5612
> je suis desole si je t ai effraye .
= i m sorry if i frightened you .
< i m sorry if i frightened you . <EOS>
> il est gaucher .
= he s a southpaw .
< he s a southpaw . <EOS>
> j en suis absolument certain .
= i m absolutely certain of it .
< i am quite sure of it . <EOS>
> je suis plutot occupee .
= i m rather busy .
< i m rather busy . <EOS>
> je regrette vraiment d entendre ca .
= i m really sorry to hear that .
< i am sorry if to hear that . <EOS>
> il est dans les affaires .
= he is in business .
< he is in business . <EOS>
> il est deux fois plus vieux qu elle .
= he is twice as old as she is .
< he is twice as old as is . <EOS>
> vous allez bien .
= you re all right .
< you re all right . <EOS>
> je ne suis personne de particulier .
= i m no one special .
< i m no one special . <EOS>
> il m aide .
= he s helping me .
< he s helping me . <EOS>
input = elle a cinq ans de moins que moi .
output = she is five years younger than me . <EOS>

input = elle est trop petit .
output = she s too trusting . <EOS>
input = je ne crains pas de mourir .
output = i m not afraid to die . <EOS>
input = c est un jeune directeur plein de talent .
output = he is a very young writer . <EOS>

```
