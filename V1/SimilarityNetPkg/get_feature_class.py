import argparse
import os
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
import yaml
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
import cv2
# from apex.contrib.optimizers import FP16_Optimizer
# from apex.fp16_utils import network_to_half
from net_model import ft_net, ft_net_dense, PCB

path_now='.'
abs_path_now=os.path.abspath(path_now)
model_path_now=os.path.join(abs_path_now,'net_last.pth')
config_path_now=os.path.join(abs_path_now,'opts.yaml')

class SimilarityNet:
    def __init__(self,model_path=model_path_now,config_path=config_path_now):
        #变量
        self.model_path=model_path
        self.config_path=config_path
        self.config=0
        self.fp16=0
        self.PCB=0
        self.use_dense=0
        self.model = 0
        #初始化函数
        self.paser_config()
        self.init_net()
    def paser_config(self):
        with open(self.config_path, 'r') as stream:
            self.config = yaml.load(stream)
            self.fp16 = self.config['fp16']
            self.PCB = self.config['PCB']
            self.use_dense = self.config['use_dense']
        print('paser_config finished')
    def init_net(self):
        # gallery_num: Matket:751; DukeMTMC-reID:702 ; cuhk03:767
        model_structure=0
        if self.use_dense:
            model_structure = ft_net_dense(751)
        else:
            model_structure = ft_net(751)
        if self.PCB:
            model_structure = PCB(751)
        if self.fp16:
            model_structure = network_to_half(model_structure)

        #print('model is ',model_structure)
        self.model = model_structure
        self.model.load_state_dict(torch.load(self.model_path))
        print('model loaded: ', self.model)
        if self.PCB:
            if self.fp16:
                self.model = PCB_test(self.model[1])
            else:
                self.model = PCB_test(self.model)
        else:
            if self.fp16:
                self.model[1].model.fc = nn.Sequential()
                self.model[1].classifier = nn.Sequential()
            else:
                self.model.model.fc = nn.Sequential()
                self.model.classifier = nn.Sequential()
        print('model modified: ', self.model)
                # Change to test mode
        self.model = self.model.eval()
        self.model = self.model.cuda()

    def show_tensor_img(self,tensor_img, text=None, should_save=False):
        # 1 将tensor数据转为numpy数据
        array1 = tensor_img.numpy()
        maxValue = array1.max()
        # 2 normalize，将图像数据扩展到[0,255]
        array1 = array1 * 255 / maxValue
        # 3 float32-->uint8
        mat = np.uint8(array1)
        #print('mat_shape:', mat.shape)  # mat_shape: (3, 982, 814)
        # 4 transpose
        mat = mat.transpose(1, 2, 0)  # mat_shape: (982, 814，3)
        # 5 cvimg到PIL
        image = Image.fromarray(mat)

        plt.imshow(image)  # 转换为(H,W,C)
        plt.show()

    def img_to_tensor(self,path):
        t1 = transforms.Resize((256, 128), interpolation=3)
        t2 = transforms.ToTensor()
        t3 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = PIL.Image.open(path)
        img = img.convert("RGB")
        img = t1(img)
        img = t2(img)
        img = t3(img)

        #print(img.size())
        return img

    def batchlize_tensor_img(self,img_tensor):
        # 作者的img.size(): torch.Size([32, 3, 256, 128])
        # 我的是img.size(): torch.Size([1, 3, 256, 128])
        #print('input shape:', img_tensor.shape)
        img_tensor = img_tensor.unsqueeze(0)
        #print('after shape', img_tensor.shape)
        return img_tensor

    def fliplr(self,img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    # data样子  nx3x256x128
    def extract_feature(self, data):
        features = torch.FloatTensor()
        count = 0
        img = data
        #print('img.size()', img.size())
        n, c, h, w = img.size()
        if self.use_dense:
            ff = torch.FloatTensor(n, 1024).zero_()
        else:
            ff = torch.FloatTensor(n, 2048).zero_()
        if self.PCB:
            ff = torch.FloatTensor(n, 2048, 6).zero_()  # we have six parts
        for i in range(2):
            if (i == 1):
                img = self.fliplr(img)
            input_img = Variable(img.cuda())
            if self.fp16:
                input_img = input_img.half()
            outputs = self.model(input_img)
            f = outputs.data.cpu().float()
            ff = ff + f
        # norm feature
        if self.PCB:
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

    def cos_dis(self,x1, x2):
        x1 = x1.view(1, -1)
        x2 = x2.view(1, -1)
        dis = torch.cosine_similarity(x1, x2, dim=1)
        torch.squeeze(dis)
        #print(dis.cpu().numpy())
        return dis.cpu().numpy()

    def get_one_img_feature(self,path):
        t1 = time.time()
        tensor_img=self.img_to_tensor(path)
        tensor_img_batched=self.batchlize_tensor_img(tensor_img)
        f=self.extract_feature(tensor_img_batched)
        print('time used one img:',time.time()-t1)
        return f
    def get_one_img_feature_from_cvimg(self,img):
        t1 = transforms.Resize((256, 128), interpolation=3)
        t2 = transforms.ToTensor()
        t3 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        img = t1(img)
        img = t2(img)
        img = t3(img)
        tensor_img_batched = self.batchlize_tensor_img(img)
        f = self.extract_feature(tensor_img_batched)


    def get_multi_img_feature(self,paths):
        t1=time.time()
        tensor_img_batched_list=[]
        c=0
        flag=0
        for path in paths:
            tensor_img=self.img_to_tensor(path)
            tensor_img_batched=self.batchlize_tensor_img(tensor_img)
            tensor_img_batched_list.append(tensor_img_batched)
        for img in tensor_img_batched_list:
            if(flag==0):
                c=img
            if(flag!=0):
                c=torch.cat((c,img),0)
            flag=flag+1
        f=self.extract_feature(c)
        print('time used multi img:', time.time() - t1)
        return f




def test_one_img():
    #实例化类
    s=SimilarityNet('./net_last.pth','./opts.yaml')

    #读取并作操作
    path1='./human/1/1.jpg'
    f=s.get_one_img_feature(path1)
    print(f)
def test_multi_img():
    # s=SimilarityNet('./net_last.pth','./opts.yaml')
    s = SimilarityNet()

    path1='../human/1/2.jpg'
    path2='../human/1/3.jpg'
    paths=[]
    paths.append(path1)
    paths.append(path2)
    paths.append(path1)
    paths.append(path2)
    paths.append(path1)
    paths.append(path2)
    paths.append(path1)
    paths.append(path2)
    paths.append(path1)
    paths.append(path2)
    fs=s.get_multi_img_feature(paths)
if __name__ == '__main__':
    # test_one_img()
    test_multi_img()