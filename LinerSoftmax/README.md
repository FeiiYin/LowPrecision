## 低精度

基于小牛翻译的Neu.Trans的项目文件，尝试自己修改的线性softmax分类器，并在FNNLM的语言模型上进行测试，包括涉及修改与编写的文件

实验结果，收敛速度较基线明显提高

时间结果，在一个epoch内，初始softmax的计算时间为 1.723192 s

修改后，softmax 时间 0.956978 秒

但是加上将tensor缩放（ScaleAndShift）转化（FloatToInt）的时间会比初始长；不过最后理想的结果是全整形，即不包括这两个步骤
