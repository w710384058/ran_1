import math, shutil, os, time, argparse
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import openpyxl
import tensorflow as tf

from pandas import DataFrame
from ITrackerData import ITrackerData
from ITrackerModel import ITrackerModel
from ran_model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel



print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

'''
Train/test code for iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
parser.add_argument('--data_path', help="Path to processed dataset. It should contain metadata.mat. Use prepareDataset.py.")
parser.add_argument('--sink', type=str2bool, nargs='?', const=True, default=False, help="Just sink and terminate.")
parser.add_argument('--reset', type=str2bool, nargs='?', const=True, default=False, help="Start from scratch (do not load).")
args = parser.parse_args()

# Change there flags to control what happens.
#doLoad = not args.reset # Load checkpoint at the beginning
#doTest = args.sink # Only run test, no training

doLoad = True # Load checkpoint at the beginning
doTest = False # Only run test, no training

#workers = 16
workers = 8
epochs = 25
#batch_size = torch.cuda.device_count()*100 # Change if out of cuda memory
batch_size =10
#batch_size=100
#print(torch.cuda.device_count())

base_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 1e20
lr = base_lr

count_test = 0
count = 0



def main():
    global args, best_prec1, weight_decay, momentum

    model = ResidualAttentionModel()
    model = torch.nn.DataParallel(model)
    model.cuda()
    imSize=(224,224)
    cudnn.benchmark = True   

    epoch = 0
   # if doLoad:
        #saved = load_checkpoint()
        #print('saved=',saved)
        #if saved:
        #    print('Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...' % (saved['epoch'], saved['best_prec1']))
        #    state = saved['state_dict']
        #    try:
        #        model.module.load_state_dict(state)
        #    except:
        #        model.load_state_dict(state)
        #    epoch = saved['epoch']
        #    best_prec1 = saved['best_prec1']
        #else:
        #    print('Warning: Could not read checkpoint!')
           

    #print('datapath=')
    #print(args.data_path)

    #dataTrain = ITrackerData(dataPath = args.data_path, split='train', imSize = imSize)
    #dataVal = ITrackerData(dataPath = args.data_path, split='test', imSize = imSize)

    dataTrain = ITrackerData(dataPath = r'D:\desktop\GazeCapture-master\pytorch\newdata', split='train', imSize = imSize)
    dataVal = ITrackerData(dataPath = r'D:\desktop\GazeCapture-master\pytorch\newdata', split='val', imSize = imSize)
    dataTest = ITrackerData(dataPath = r'D:\desktop\GazeCapture-master\pytorch\newdata', split='test', imSize = imSize)


    #print('batchsize=',batch_size)
    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        dataTest,
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)

    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # Quick test
    if doTest:
        test(test_loader, model, criterion, epoch)
        #validate(val_loader, model, criterion, epoch)
        return
    #print("epoch=",epoch)
    for epoch in range(0, epoch):
        
        adjust_learning_rate(optimizer, epoch)
        
    for epoch in range(epoch, epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion,optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    #for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(train_loader):
    for i, (imFace, gaze) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.cuda()
        #imEyeL = imEyeL.cuda()
        #imEyeR = imEyeR.cuda()
        #faceGrid = faceGrid.cuda()
        gaze = gaze.cuda()
        
        imFace = torch.autograd.Variable(imFace, requires_grad = True)
        #imEyeL = torch.autograd.Variable(imEyeL, requires_grad = True)
        #imEyeR = torch.autograd.Variable(imEyeR, requires_grad = True)
        #faceGrid = torch.autograd.Variable(faceGrid, requires_grad = True)
        gaze = torch.autograd.Variable(gaze, requires_grad = False)

        # compute output
        output = model(imFace)

        loss = criterion(output, gaze)
        #loss = criterion(outputs,gaze.view(1, -1))
        losses.update(loss.data.item(), imFace.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count=count+1

        print('Epoch (train): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def validate(val_loader, model, criterion, epoch):
    global count_test
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()


    oIndex = 0
    for i, (imFace,gaze) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.cuda()
        #imEyeL = imEyeL.cuda()
        #imEyeR = imEyeR.cuda()
        #faceGrid = faceGrid.cuda()
        gaze = gaze.cuda()
        
        imFace = torch.autograd.Variable(imFace, requires_grad = False)
        #imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
        #imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
        #faceGrid = torch.autograd.Variable(faceGrid, requires_grad = False)888888888888888
        gaze = torch.autograd.Variable(gaze, requires_grad = False)

        # compute output
        with torch.no_grad():
            output = model(imFace)

        loss = criterion(output, gaze)
        
        lossLin = output - gaze
        lossLin = torch.mul(lossLin,lossLin)
        lossLin = torch.sum(lossLin,1)
        lossLin = torch.mean(torch.sqrt(lossLin))

        losses.update(loss.data.item(), imFace.size(0))
        lossesLin.update(lossLin.item(), imFace.size(0))
     
        # compute gradient and do SGD step
        # measure elapsed time
        # batch_time.update(time.time() - end)
        end = time.time()


        print('Epoch (val): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                   loss=losses,lossLin=lossesLin))

    return lossesLin.avg







def test(test_loader, model, criterion, epoch):

    model = ITrackerModel()
    model = torch.nn.DataParallel(model)
    saved = torch.load(r'D:\source\test\test\best_checkpoint.pth.tar')
    #print(saved)
    state = saved['state_dict']
    model.load_state_dict(state)

    #print(len(test_loader))
    global count_test
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()


    oIndex = 0
    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, device) in enumerate(test_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()
        imEyeR = imEyeR.cuda()
        faceGrid = faceGrid.cuda()
        gaze = gaze.cuda()
        
        imFace = torch.autograd.Variable(imFace, requires_grad = False)
        imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
        faceGrid = torch.autograd.Variable(faceGrid, requires_grad = False)
        gaze = torch.autograd.Variable(gaze, requires_grad = False)

        

        # compute output
        with torch.no_grad():
            output = model(imFace, imEyeL, imEyeR, faceGrid)

        loss = criterion(output, gaze)
        
        lossLin = output - gaze
        lossLin = torch.mul(lossLin,lossLin)
        lossLin = torch.sum(lossLin,1)
        lossLin = torch.mean(torch.sqrt(lossLin))

        losses.update(loss.data.item(), imFace.size(0))
        lossesLin.update(lossLin.item(), imFace.size(0))
     
        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        print('Epoch (test): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                    epoch, i, len(test_loader), batch_time=batch_time,
                   loss=losses,lossLin=lossesLin))
        #print(device)

        path = 'device.txt'
        f = open(path, 'a')
        f.write(str(device))
        f.write('\n')
        
        path = 'gaze_x.txt'
        f = open(path, 'a')
        gaze = gaze.cpu().numpy()
        #print('gaze = ',gaze)
        f.write(str(gaze[0][0]))
        f.write('\n')

        path = 'gaze_y.txt'
        f = open(path, 'a')
        f.write(str(gaze[0][1]))
        f.write('\n')

        path = 'pre_x.txt'
        f = open(path, 'a')
        pre = output.cpu().numpy()
        print('pre = ',pre)
        f.write(str(pre[0][0]))
        f.write('\n')

        path = 'pre_y.txt'
        f = open(path, 'a')
        f.write(str(pre[0][1]))
        f.write('\n')



        #path = 'predict_gaze.txt'
        #f = open(path, 'a')
        #f.write(str(pre))
        #f.write('\n')

    return lossesLin.avg









CHECKPOINTS_PATH = '.'

def load_checkpoint(filename='checkpoint.pth.tar'):
    #filename = os.path.join(CHECKPOINTS_PATH, filename)
    #filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    #if not os.path.isfile(filename):
    if not os.path.isfile("D:\desktop\GazeCapture-master\pytorch\checkpoint.pth.tar"):
        
        return None
    state = torch.load("D:\desktop\GazeCapture-master\pytorch\checkpoint.pth.tar")
    return state

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_' + filename)
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
    print(f'DONE')
