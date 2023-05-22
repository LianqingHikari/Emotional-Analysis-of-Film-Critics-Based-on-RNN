# Emotional-Analysis-of-Film-Critics-Based-on-RNN
本项目为2020年华南理工大学计算机科学与工程学院数据结构课程设计。我想你可能会好奇，为什么数据结构课程设计要做深度学习的项目。这也是课程设计答辩上我的老师问我的问题：“你这个项目好像跟数据结构没什么关系啊”。“这是课程设计任务书上的其中一个项目，我已经完成了该任务的所有要求。”我是这么回答的。显然我在阴阳怪气：“你们自己搞的任务书自己不知道的吗”。然后果不其然我的课设痛失“优秀”评分。
好吧以上只是玩笑，痛失“优秀”当然是因为我的课设做得还不够好，但是这段对话是真实存在过的。

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

