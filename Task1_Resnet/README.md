# CIFAR-10 Classification（ResNet-18）
## 1. 背景与目标

- 任务：**图像分类**。将 32×32 **彩色**图像划分到 **10** 类(*airplane、automobile、bird、cat、deer、dog、frog、horse、ship、truck*)。
	- 共60000张，分为10类，每类6000张图
	- train:50,000, batch_size=10000, batch=5
	- test:10,000, 每一类随机抽取1,000张
	- dimension (3, 32, 32)
    
- 评价指标：**loss, accuracy** 
    
- 目标：基于 ResNet-18（CIFAR-stem）在标准增强下得到稳定的结果。