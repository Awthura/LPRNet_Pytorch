import torch
import torch.nn as nn
from torch.autograd import Variable
from model.LPRNet import LPRNet

model = LPRNet(lpr_max_len=8, phase=False, class_num=68, dropout_rate=0.5)

dummy_input = Variable(torch.randn(1, 3, 24, 94))
model.load_state_dict(torch.load('weights/Final_LPRNet_model.pth', map_location=torch.device('cpu')))
torch.onnx.export(model, dummy_input, "LPR.onnx")