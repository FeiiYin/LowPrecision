## 介绍现在常用的模型压缩(Compression)和量化(quantization)的方法

+ 参数修剪和共享（parameter pruning and sharing）

基于参数修剪和共享的方法针对模型参数的冗余性，试图去除冗余和不重要的项。

+ 低秩因子分解（low-rank factorization）

基于低秩因子分解的技术使用矩阵/张量分解来估计深度学习模型的信息参数。（压缩矩阵）

+ 转移/紧凑卷积滤波器（transferred/compact convolutional filters）

基于传输/紧凑卷积滤波器的方法设计了特殊的结构卷积滤波器来降低存储和计算复杂度。(以为是压缩卷积核，这里并不理解)

+ 知识蒸馏（knowledge distillation）

知识蒸馏方法通过学习一个蒸馏模型，训练一个更紧凑的神经网络来重现一个更大的网络的输出。 

+ 尝试用位运算来进行加速矩阵的乘法与加法

## CLIP-Q : http://yanjoy.win/2018/08/06/CLIP-Q/#more

