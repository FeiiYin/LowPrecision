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

+ 进入虚拟环境 `conda activate fairseq`

#### 查看CUDA版本

+ `cat /usr/local/cuda/version.txt`

#### pytorch 环境搭建

+ 卸载之前的 `conda uninstall pytorch`

+ 安装 `conda install pytorch=1.0.0 cuda90 -c pytorch`

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
