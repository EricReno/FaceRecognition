# FaceNet 

- facenet的pytorch源码，含训练、推理、评测、部署脚本，论文传送门：

https://arxiv.org/pdf/1503.03832 

## 数据集一: LFW
13000+ 250*250 jpg图像
每张脸都贴上了所画的人的名字，图片中的1680人在数据集中有两个或更多不同的照片。

训练精度ACC达98%

## 数据集二: CASIA-FaceV5-CUT
为CASIA-FaceV5的剪裁版，仅保留头像，去除背景框。

CASIA-FaceV5：2500 640*480 
CASIA-FaceV5-CUT：2500 160*160

总共有500个不同实例，每个实例五张不同角度人脸图像

Val: 250, max ACC: 0.96667, threshold = 0.95000
Train: 2250, max ACC: 0.97585, threshold = 0.81000


https://github.com/user-attachments/assets/e1feebb3-c521-4d31-a23d-801dfab28457