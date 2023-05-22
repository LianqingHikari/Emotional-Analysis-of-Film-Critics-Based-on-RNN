## 关于train.py

运行环境：

- python库：tensorflow-gpu==2.3.0, numpy
- Python版本：3.6.2
- 数据集：imdb_reviews/subwords8k
- 软件开发工具包：NVIDIA Toolkit 9.0, NVIDIA CUDA 10.1
- 硬件：计算机需搭载GPU，内存需>=4G，否则训练过程可能相对缓慢

具有上述运行环境后直接运行即可，若不搭载gpu，tensorflow或python版本不对，可能导致无法运行
或训练速度较慢

若为第一次运行，数据集将自动下载到本地，因此第一次运行请务必联网

训练所得模型默认保存为my_model.h5，压缩文件中已包含训练好的模型

## 关于interface.py

运行时务必将my_model.h5文件与.py文件放在同一文件夹

运行后出现界面窗口，在文本框中输入影评文本在点击确定即可进行预测。

预测结果为“积极”和“消极”两种，得分位于0~10分之间。
