import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
from model import ft_net, ft_net_dense, PCB, PCB_test
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
from apex.contrib.optimizers import FP16_Optimizer
from apex.fp16_utils import network_to_half
# 定义参数
gpu_ids='0'
which_epoch=59
name='ft_ResNet50'
test_dir='./data/Market-1501-v15.09.15/pytorch'
train_all=0
color_jitter=0
batchsize=32
stride=2
erasing_p=default=0
use_dense=0
lr=0
droprate=0.5
PCB=0   #action='store_true'的意思为：如果不写参数，PCB默认为FALSE的，写了才置为TRUE
multi=0
fp16=0  #use float16 instead of float32, which will save about 50% memory' )

###加载参数文件###
# load the training config
config_path = './opts.yaml'
with open(config_path, 'r') as stream:
    config = yaml.load(stream)

fp16 = config['fp16']
PCB = config['PCB']
use_dense = config['use_dense']

str_ids = gpu_ids.split(',')
# which_epoch =  which_epoch
name = name
test_dir = test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True


#加载模型
def load_network(network):
    save_path = './net_last.pth'
    print(save_path)
    network.load_state_dict(torch.load(save_path))
    return network
######################################################################
# Load Collected data Trained model
# gallery_num: Matket:751; DukeMTMC-reID:702 ; cuhk03:767
print('-------test-----------')
if  use_dense:
    model_structure = ft_net_dense(751)
else:
    model_structure = ft_net(751)

if  PCB:
    model_structure = PCB(751)

if  fp16:
    model_structure = network_to_half(model_structure)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
if  PCB:
    if  fp16:
        model = PCB_test(model[1])
    else:
        model = PCB_test(model)
else:
    if  fp16:
        model[1].model.fc = nn.Sequential()
        model[1].classifier = nn.Sequential()
    else:
        model.model.fc = nn.Sequential()
        model.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
model = model.cuda()


# 定义工具函数
def show_tensor_img(tensor_img, text=None, should_save=False):
    # 1 将tensor数据转为numpy数据
    array1 = tensor_img.numpy()
    maxValue = array1.max()
    # 2 normalize，将图像数据扩展到[0,255]
    array1 = array1 * 255 / maxValue
    # 3 float32-->uint8
    mat = np.uint8(array1)
    print('mat_shape:', mat.shape)  # mat_shape: (3, 982, 814)
    # 4 transpose
    mat = mat.transpose(1, 2, 0)  # mat_shape: (982, 814，3)
    # 5 cvimg到PIL
    image = Image.fromarray(mat)

    plt.imshow(image)  # 转换为(H,W,C)
    plt.show()


def img_to_tensor(path):
    t1 = transforms.Resize((256, 128), interpolation=3)
    t2 = transforms.ToTensor()
    t3 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = PIL.Image.open(path)
    img = img.convert("RGB")
    img = t1(img)
    img = t2(img)
    img = t3(img)

    print(img.size())
    return img
def batchlize_tensor_img(img_tensor):
    # 作者的img.size(): torch.Size([32, 3, 256, 128])
    # 我的是img.size(): torch.Size([1, 3, 256, 128])
    print('input shape:',img_tensor.shape)
    img_tensor=img_tensor.unsqueeze(0)
    print('after shape',img_tensor.shape)
    return img_tensor


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, data):
    use_dense = 0
    PCB = 0

    features = torch.FloatTensor()
    count = 0

    img = data
    print('img.size()', img.size())
    n, c, h, w = img.size()
    if use_dense:
        ff = torch.FloatTensor(n, 1024).zero_()
    else:
        ff = torch.FloatTensor(n, 2048).zero_()
    if PCB:
        ff = torch.FloatTensor(n, 2048, 6).zero_()  # we have six parts
    for i in range(2):
        if (i == 1):
            img = fliplr(img)
        input_img = Variable(img.cuda())
        if fp16:
            input_img = input_img.half()
        outputs = model(input_img)
        f = outputs.data.cpu().float()
        ff = ff + f
    # norm feature
    if PCB:
        # feature size (n,2048,6)
        # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
    else:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

    features = torch.cat((features, ff), 0)
    return features


# 最终的函数
def get_one_feature(path):
    tensor_img = img_to_tensor(path)  # 测试没问题，读取图片并transform
    tensor_img_batchlized = batchlize_tensor_img(tensor_img)  # 转成假的batch
    t1 = time.time()
    f = extract_feature(model, tensor_img_batchlized)
    t2 = time.time()
    print('f time:', t2 - t1)
    return f
def cos_dis(x1,x2):
    x1=x1.view(1,-1)
    x2=x2.view(1,-1)
    dis=torch.cosine_similarity(x1, x2, dim=1)
    torch.squeeze(dis)
    print(dis.cpu().numpy())
    return dis.cpu().numpy()

path1='./human/1/2.jpg'
path2='./human/1/3.jpg'

tensor_img1=img_to_tensor(path1)  #测试没问题，读取图片并transform
tensor_img_batchlized1=batchlize_tensor_img(tensor_img1) #转成假的batch

tensor_img2=img_to_tensor(path2)  #测试没问题，读取图片并transform
tensor_img_batchlized2=batchlize_tensor_img(tensor_img2) #转成假的batch
c=torch.cat((tensor_img_batchlized1,tensor_img_batchlized2),0)
c=torch.cat((c,c),0)  #4
c=torch.cat((c,c),0)  #8
c=torch.cat((c,c),0)  #16
c=torch.cat((c,c),0)  #32
# c=torch.cat((c,c),0)  #64
print('****************************')
print('tensor_img1.shape',tensor_img_batchlized2.shape)
print('tensor_img2.shape',tensor_img_batchlized2.shape)
print('c.shape',c.shape)
# t1=time.time()
# f=extract_feature(model,tensor_img_batchlized)
# t2=time.time()
# print('f time:',t2-t1)


t11=time.time()
f=extract_feature(model,c)
t22=time.time()
print(f)
print('time:',t22-t11)