# -*- coding:utf-8 -*-
# 用于模型的单张图像分类操作
import os
os.environ['GLOG_minloglevel'] = '2' # 将caffe的输出log信息不显示，必须放到import caffe前
import caffe # caffe 模块
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import sys
import threading

# 分类单张图像img
def detection(im, net, transformer, i):
    #im = cv2.imread(img)
    #im = caffe.io.load_image(img)
    if lock.acquire():
        net.blobs['data'].data[...] = transformer.preprocess('data', im)
        start = time.clock()
        # 执行测试
        net.forward()
        end = time.clock()
        print('detection time: %f s' % (end - start))

        # 查看目标检测结果
        print(net.blobs['bbox-list'].data)
        loc = net.blobs['bbox-list'].data[0]
        tmp_confidence = 0
        print('orig pic width = %d height = %d' % (im.shape[1], im.shape[0]))
	
        #查看了结构文件发现在CAFFE一开始图像输入的时候就已经将图片缩小了，宽度640高度640
        #然后我们在net.blobs['bbox-list'].data得到的是侦测到的目标座标，但是是相对于640*640的
        #所以我们要把座标转换回相对原大小的位置，下面im.shape是保存在原尺寸的宽高，
	    #此处值要根据detectnet众的input dim来设置
        for l in range(len(loc)):
		    xmin = int(loc[l][0] * im.shape[1] / 640)
		    ymin = int(loc[l][1] * im.shape[0] / 640)
		    xmax = int(loc[l][2] * im.shape[1] /640)
		    ymax = int(loc[l][3] * im.shape[0] / 640)
		    #print('xmin = %f ymin = %f xmax = %f ymax = %f'%(xmin, ymin, xmax, ymax))
		    #保存最可信的被检测到的区域
		    if loc[l][4] > 0:
			    #if loc[l][4] > tmp_confidence:
                cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (55 / 255.0, 255 / 255.0, 155 / 255.0), 2)
                #cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (55 / 255.0, 255 / 255.0, 155 / 255.0), 2)
                # 显示结果
                print('confidence =%f ' % (loc[l][4]))
                #print('max_xleft = %f max_ytop = %f max_xright = %f max_ybottom = %f'%(max_xleft, max_ytop, max_xright, max_ybottom))
                crop_img = im[ymin:ymax,xmin:xmax]
                cv2.imwrite('result/crop_test'+str(i)+str(l)+'.jpg', crop_img)
                cv2.imwrite('result/test_boxed'+str(i)+str(l)+'.jpg', im)
        lock.release()

#CPU或GPU模型转换
#caffe.set_mode_cpu()
#caffe.set_device(0)
caffe.set_mode_gpu()

caffe_root = '/home/raiden/caffe/'
# 网络参数（权重）文件
caffemodel = './coco-bottle.caffemodel'
# 网络实施结构配置文件
deploy = './deploy.prototxt'


img_root = caffe_root + 'data/'

# 网络实施分类
net = caffe.Net(deploy,  # 定义模型结构
                caffemodel,  # 包含了模型的训练权值
                caffe.TEST)  # 使用测试模式(不执行dropout)

# 加载ImageNet图像均值 (随着Caffe一起发布的)
#print(os.environ['PYTHONPATH'])
mu = np.load('./mean.npy')
mu = mu.mean(1).mean(1)  # 对所有像素值取平均以此获取BGR的均值像素值

# 图像预处理
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
#[0, 1]缩放到[0, 255] 使用cv2接口read的不需要
#transformer.set_raw_scale('data', 255)
# RGB -> BGR 使用cv2接口read的不需要
#transformer.set_channel_swap('data', (2,1,0))

# 处理图像
#img = './test2.jpg'
cap = cv2.VideoCapture("videotest.mp4")
success,frame = cap.read() 
i = 0
lock = threading.Lock()#创建一个锁对象
while success:
    success,im = cap.read()
    detection(im,net,transformer, i)
    i+= 1
cap.release()


