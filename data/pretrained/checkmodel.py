import torch
import torch.nn as nn
from torchsummary import summary

model = torch.load('./capsule.pt')
model.eval()


# 모델의 summary를 출력합니다.
# (10,)은 input tensor의 shape를 나타냅니다. 이 shape는 모델에 따라 적절히 변경해야 합니다.
summary(model, input_size=(10,))