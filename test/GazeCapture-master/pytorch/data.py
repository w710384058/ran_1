import os 
from mat4py import loadmat
import scipy.io
import matplotlib.pyplot as plt
import numpy
import numpy as np
import math
import torch.nn as nn
import torch
#def loadMetadata(filename,silent=False):
#    try:
#        if not silent:
#            print('\tReading metadata from %s...'%filename)
#        metadata = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)
#    except:
#        print('\tFailed to read the meta file "%s"!'%filename)
#        return None
#    return metadata


device=[]
gaze_x=[]
gaze_y=[]
pre_x=[]
pre_y=[]


def readtxt(txtname,appendname):
    f = open(txtname)
    for line in f.readlines():
        appendname.append(line)
    f.close


#def cal(device, gaze_x, gaze_y, pre_x, pre_y):
#    for i in device[i]:
#        if device[i] == "['iPad Air 2']\n":
#            ipadair2

def plotpoint(device_name):
    y=-1
    x=0
    tem_gaze_x=[]
    tem_gaze_y=[]
    tem_pre_x=[]
    tem_pre_y=[]
    for i in device:
        y+=1
        characters="\"'[\n]"
        
        for x in range(len(characters)):
            i = i.replace(characters[x],"")
        #print(i)
        if i == device_name:
            tem_gaze_x.append(float(gaze_x[y]))
            tem_gaze_y.append(float(gaze_y[y]))
            tem_pre_x.append(float(pre_x[y]))
            tem_pre_y.append(float(pre_y[y]))
    print(len(tem_gaze_x))
    print('plot')
    #print(len(device))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(device_name)
    ax.xlabel=('x')
    plt.ylabel=('y')
    plt.scatter(tem_gaze_x, tem_gaze_y, s=6)
    plt.show()

    return [tem_gaze_x, tem_gaze_y, tem_pre_x, tem_pre_y]


def cal_tab(value):
    error1=[]
    error2=[]
    error3=[]
    error4=[]
    error5=[]
    error6=[]
    error7=[]
    error8=[]
    error9=[]
    error10=[]
    error11=[]
    error12=[]
    error13=[]
    error14=[]
    error15=[]
    error16=[]
    error17=[]
    error18=[]
    error19=[]
    error20=[]
    #error_y=[]
    tem_gaze_x_array = numpy.array(value[0])
    tem_gaze_y_array = numpy.array(value[1])
    tem_pre_x_array = numpy.array(value[2])
    tem_pre_y_array = numpy.array(value[3])
    criterion = nn.MSELoss().cuda()
    #print(max(tem_gaze_x_array))
    for i in range(len(tem_gaze_x_array)):
        if((tem_gaze_x_array[i]>-20.3)&(tem_gaze_x_array[i]<=-13.5)&(tem_gaze_y_array[i]>0.1)&(tem_gaze_y_array[i]<=6.9)):
           mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
           #print(mse_error)
           #print('loss.item=',type(loss.item()))
           error1.append(mse_error.item())#float
        if((tem_gaze_x_array[i]>-13.5)&(tem_gaze_x_array[i]<=-6.7)&(tem_gaze_y_array[i]>0.1)&(tem_gaze_y_array[i]<=6.9)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error2.append(mse_error.item())        
        if((tem_gaze_x_array[i]>-20.3)&(tem_gaze_x_array[i]<=-13.5)&(tem_gaze_y_array[i]>-6.7)&(tem_gaze_y_array[i]<=0.1)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error3.append(mse_error.item())           
        if((tem_gaze_x_array[i]>-13.5)&(tem_gaze_x_array[i]<=-6.7)&(tem_gaze_y_array[i]>-6.7)&(tem_gaze_y_array[i]<=0.1)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error4.append(mse_error.item())
        if((tem_gaze_x_array[i]>-6.7)&(tem_gaze_x_array[i]<=0.1)&(tem_gaze_y_array[i]>13.7)&(tem_gaze_y_array[i]<=20.5)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error5.append(mse_error.item())
        if((tem_gaze_x_array[i]>0.1)&(tem_gaze_x_array[i]<=6.9)&(tem_gaze_y_array[i]>13.7)&(tem_gaze_y_array[i]<=20.5)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error6.append(mse_error.item())
        if((tem_gaze_x_array[i]>-6.7)&(tem_gaze_x_array[i]<=0.1)&(tem_gaze_y_array[i]>6.9)&(tem_gaze_y_array[i]<=13.7)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error7.append(mse_error.item())  
        if((tem_gaze_x_array[i]>0.1)&(tem_gaze_x_array[i]<=6.9)&(tem_gaze_y_array[i]>6.9)&(tem_gaze_y_array[i]<=13.7)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error8.append(mse_error.item())
        if((tem_gaze_x_array[i]>-6.7)&(tem_gaze_x_array[i]<=0.1)&(tem_gaze_y_array[i]>0.1)&(tem_gaze_y_array[i]<=6.9)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error9.append(mse_error.item())
        if((tem_gaze_x_array[i]>0.1)&(tem_gaze_x_array[i]<=6.9)&(tem_gaze_y_array[i]>0.1)&(tem_gaze_y_array[i]<=6.9)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error10.append(mse_error.item())
        if((tem_gaze_x_array[i]>-6.7)&(tem_gaze_x_array[i]<=0.1)&(tem_gaze_y_array[i]>-6.7)&(tem_gaze_y_array[i]<=0.1)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error11.append(mse_error.item())
        if((tem_gaze_x_array[i]>0.1)&(tem_gaze_x_array[i]<=6.9)&(tem_gaze_y_array[i]>-6.7)&(tem_gaze_y_array[i]<=0.1)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error12.append(mse_error.item())
        if((tem_gaze_x_array[i]>-6.7)&(tem_gaze_x_array[i]<=0.1)&(tem_gaze_y_array[i]>-13.5)&(tem_gaze_y_array[i]<=-6.7)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error13.append(mse_error.item())
        if((tem_gaze_x_array[i]>0.1)&(tem_gaze_x_array[i]<=6.9)&(tem_gaze_y_array[i]>-13.5)&(tem_gaze_y_array[i]<=-6.7)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error14.append(mse_error.item())
        if((tem_gaze_x_array[i]>-6.7)&(tem_gaze_x_array[i]<=0.1)&(tem_gaze_y_array[i]>-20.3)&(tem_gaze_y_array[i]<=-13.5)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error15.append(mse_error.item())
        if((tem_gaze_x_array[i]>0.1)&(tem_gaze_x_array[i]<=6.9)&(tem_gaze_y_array[i]>-20.3)&(tem_gaze_y_array[i]<=-13.5)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error16.append(mse_error.item())
        if((tem_gaze_x_array[i]>6.9)&(tem_gaze_x_array[i]<=13.7)&(tem_gaze_y_array[i]>0.1)&(tem_gaze_y_array[i]<=6.9)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error17.append(mse_error.item())
        if((tem_gaze_x_array[i]>13.7)&(tem_gaze_x_array[i]<=20.5)&(tem_gaze_y_array[i]>0.1)&(tem_gaze_y_array[i]<=6.9)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error18.append(mse_error.item())
        if((tem_gaze_x_array[i]>6.9)&(tem_gaze_x_array[i]<=13.7)&(tem_gaze_y_array[i]>-6.7)&(tem_gaze_y_array[i]<=0.1)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error19.append(mse_error.item())
        if((tem_gaze_x_array[i]>13.7)&(tem_gaze_x_array[i]<=20.5)&(tem_gaze_y_array[i]>-6.7)&(tem_gaze_y_array[i]<=0.1)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error20.append(mse_error.item())

    
    errorvalue1 = np.sum(error1)/len(error1)
    print('1=',errorvalue1)
    errorvalue2 = np.sum(error2)/len(error2)
    print('2=',errorvalue2)
    errorvalue3 = np.sum(error3)/len(error3)
    print('3=',errorvalue3)
    errorvalue4 = np.sum(error4)/len(error4)
    print('4=',errorvalue4)
    errorvalue5 = np.sum(error5)/len(error5)
    print('5=',errorvalue5)
    errorvalue6 = np.sum(error6)/len(error6)
    print('6=',errorvalue6)
    errorvalue7 = np.sum(error7)/len(error7)
    print('7=',errorvalue7)
    errorvalue8 = np.sum(error8)/len(error8)
    print('8=',errorvalue8)
    errorvalue9 = np.sum(error9)/len(error9)
    print('9=',errorvalue9)
    errorvalue10 = np.sum(error10)/len(error10)
    print('10=',errorvalue10)
    errorvalue11 = np.sum(error11)/len(error11)
    print('11=',errorvalue11)
    errorvalue12 = np.sum(error12)/len(error12)
    print('12=',errorvalue12)
    errorvalue13 = np.sum(error13)/len(error13)
    print('13=',errorvalue13)
    errorvalue14 = np.sum(error14)/len(error14)
    print('14=',errorvalue14)
    errorvalue15 = np.sum(error15)/len(error15)
    print('15=',errorvalue15)
    errorvalue16 = np.sum(error16)/len(error16)
    print('16=',errorvalue16)
    errorvalue17 = np.sum(error17)/len(error17)
    print('17=',errorvalue17)
    errorvalue18 = np.sum(error18)/len(error18)
    print('18=',errorvalue18)
    errorvalue19 = np.sum(error19)/len(error19)
    print('19=',errorvalue19)
    errorvalue20 = np.sum(error20)/len(error20)
    print('20=',errorvalue20)


def cal_phone(value):
    error1=[]
    error2=[]
    error3=[]
    error4=[]
    error5=[]
    error6=[]
    error7=[]
    error8=[]
    error9=[]
    error10=[]
    error11=[]
    error12=[]
    #error_y=[]
    tem_gaze_x_array = numpy.array(value[0])
    tem_gaze_y_array = numpy.array(value[1])
    tem_pre_x_array = numpy.array(value[2])
    tem_pre_y_array = numpy.array(value[3])
    criterion = nn.MSELoss().cuda()
    #print(max(tem_gaze_x_array))
    for i in range(len(tem_gaze_x_array)):
        if((tem_gaze_x_array[i]<=-5)&(tem_gaze_y_array[i]>0)):
           mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
           error1.append(mse_error.item())#float
        if((tem_gaze_x_array[i]>-5)&(tem_gaze_x_array[i]<=0)&(tem_gaze_y_array[i]>0)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error2.append(mse_error.item())        
        if((tem_gaze_x_array[i]>0)&(tem_gaze_x_array[i]<=5)&(tem_gaze_y_array[i]>0)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error3.append(mse_error.item())           
        if((tem_gaze_x_array[i]>5)&(tem_gaze_y_array[i]>0)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error4.append(mse_error.item())
        if((tem_gaze_x_array[i]<=-5)&(tem_gaze_y_array[i]>-4)&(tem_gaze_y_array[i]<=0)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error5.append(mse_error.item())
        if((tem_gaze_x_array[i]>-5)&(tem_gaze_x_array[i]<=0)&(tem_gaze_y_array[i]>-4)&(tem_gaze_y_array[i]<=0)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error6.append(mse_error.item())
        if((tem_gaze_x_array[i]>0)&(tem_gaze_x_array[i]<=5)&(tem_gaze_y_array[i]>-4)&(tem_gaze_y_array[i]<=0)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error7.append(mse_error.item())  
        if((tem_gaze_x_array[i]>5)&(tem_gaze_y_array[i]>-4)&(tem_gaze_y_array[i]<=0)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error8.append(mse_error.item())
        if((tem_gaze_x_array[i]>-5)&(tem_gaze_x_array[i]<=0)&(tem_gaze_y_array[i]>-8)&(tem_gaze_y_array[i]<=-4)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error9.append(mse_error.item())
        if((tem_gaze_x_array[i]>0)&(tem_gaze_x_array[i]<=5)&(tem_gaze_y_array[i]>-8)&(tem_gaze_y_array[i]<=-4)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error10.append(mse_error.item())
        if((tem_gaze_x_array[i]>-5)&(tem_gaze_x_array[i]<=0)&(tem_gaze_y_array[i]<=-8)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error11.append(mse_error.item())
        if((tem_gaze_x_array[i]>0)&(tem_gaze_x_array[i]<=5)&(tem_gaze_y_array[i]<=-8)):
            mse_error = criterion(torch.FloatTensor([tem_pre_x_array[i],tem_pre_y_array[i]]), torch.FloatTensor([tem_gaze_x_array[i],tem_gaze_y_array[i]]))#tensor
            error12.append(mse_error.item())
        
    
    errorvalue1 = np.sum(error1)/len(error1)
    print('1=',errorvalue1)
    errorvalue2 = np.sum(error2)/len(error2)
    print('2=',errorvalue2)
    errorvalue3 = np.sum(error3)/len(error3)
    print('3=',errorvalue3)
    errorvalue4 = np.sum(error4)/len(error4)
    print('4=',errorvalue4)
    errorvalue5 = np.sum(error5)/len(error5)
    print('5=',errorvalue5)
    errorvalue6 = np.sum(error6)/len(error6)
    print('6=',errorvalue6)
    errorvalue7 = np.sum(error7)/len(error7)
    print('7=',errorvalue7)
    errorvalue8 = np.sum(error8)/len(error8)
    print('8=',errorvalue8)
    errorvalue9 = np.sum(error9)/len(error9)
    print('9=',errorvalue9)
    errorvalue10 = np.sum(error10)/len(error10)
    print('10=',errorvalue10)
    errorvalue11 = np.sum(error11)/len(error11)
    print('11=',errorvalue11)
    errorvalue12 = np.sum(error12)/len(error12)
    print('12=',errorvalue12)
    

def cal_distance(gx, px, gy, py):
    dis = math.sqrt(np.power((gx-px),2)+np.power((gy-py),2))
    return dis
    

def main():
    readtxt('device.txt',device)
    readtxt('gaze_x.txt',gaze_x)
    readtxt('gaze_y.txt',gaze_y)
    readtxt('pre_x.txt',pre_x)
    readtxt('pre_y.txt',pre_y)
    #print(device)
    label_pre_value = plotpoint('iPad Air 2')
    cal_tab(label_pre_value)
    #cal_phone(label_pre_value)
    #cal(device, gaze_x, gaze_y, pre_x, pre_y)


if __name__ == "__main__":
    main()
    print('DONE')

