import torch
import torch.nn as nn
import torch.nn.functional as F
from models.efficientformer import efficientformer_l1 as create_model

import torchvision
from thop import profile

print('==> Building model..')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BatchNorm2d = nn.BatchNorm2d
model = create_model().to(device)
print(model)
# Model
# dummy_input = torch.randn(1, 3, 244, 244).to(device)
# flops, params = profile(model, (dummy_input,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))


