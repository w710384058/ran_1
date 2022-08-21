#import math, shutil, os, time, argparse
#import numpy as np
#import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from ITrackerModel import ITrackerModel
from ITrackerData import ITrackerData

import torch
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path




model = ITrackerModel()
model = torch.nn.DataParallel(model)
saved = torch.load(r'D:\source\test\test\best_checkpoint.pth.tar')
#print(saved)
state = saved['state_dict']
model.load_state_dict(state)
trans = transforms.Compose([
    
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    ])

image = Image.open(r'D:\Desktop\GazeCapture-master\pytorch\GazeCapture\00003\frames\00001.jpg')

input = trans(image)

#input = input.view(1, 3, 32,32)

output = model(input)


print(output)







#model = ITrackerModel()
#params = model.state_dict()
#for k,v in params.items():
#    print('params=',k)
#model = torch.nn.DataParallel(model)
#checkpoint =torch.load("D:\desktop\GazeCapture-master\pytorch\checkpoint.pth.tar", map_location=torch.device('cpu'))
#for k,v in checkpoint.items():
#    print(k)
#state = checkpoint['state_dict']
##print(state)
#model.load_state_dict(state)
#model.eval()

#def predict_(imFace, imEyeL, imEyeR, faceGrid):
#    output = model(imFace, imEyeL, imEyeR, faceGrid)
#    return output

#if __name__ == "__main__":
#    predict_()
#    print(output)
#    print(done)

