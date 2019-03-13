import argparse
from Models import VGGUnet, VGGSegnet, FCN8, FCN32, FCN, ResNet
from cnn_finetune import resnet_101
import LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
from keras.optimizers import Adam, RMSprop, SGD
import random
import os
from LoadBatches import getimg
import ScanNet2_deploy
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--n_classes", type=int, default=2)
parser.add_argument("--multi_gpu", type=int, default=1)
parser.add_argument("--thresh", type=float, default=.7)
parser.add_argument("--load_weights", type=str, default="")
parser.add_argument("--model_name", type=str, default="scan")
args = parser.parse_args()
n_classes = args.n_classes
model_name = args.model_name

modelFns = {'fcn': FCN.FCN, 'vgg_segnet': VGGSegnet.VGGSegnet, 'vgg_unet': VGGUnet.VGGUnet,
            'vgg_unet2': VGGUnet.VGGUnet2, 'fcn8': FCN8.FCN8, 'fcn32': FCN32.FCN32}
if model_name == 'scan':
    m = ScanNet2_deploy.GetModel()
elif model_name == 'res':
    m = ResNet.GetModel()
elif model_name == 'res101':
    m = resnet_101.resnet101_model(244, 244, 3, 2)
else:
    modelFN = modelFns[model_name]
    m = modelFN(n_classes)
from keras.utils import multi_gpu_model
if args.multi_gpu>1:
    m = multi_gpu_model(m, gpus=args.multi_gpu)

m.load_weights(args.load_weights)
m.compile(loss='categorical_crossentropy',
          optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
          metrics=['accuracy'])

with open('/home/data/ACDC/Patch4/train_0.txt', 'r') as f:
    list0 = f.readlines()
lists = list0[2439000:]
data = []
img = []
name = []
batch_size=100
for i in tqdm(range(len(lists))):
    l = lists[i]
    try:
        [number, x, y]=map(int, l.split(',')[:3])
        img_data = getimg(number, x, y)/255.
        if img_data is None:
            print(l)
            continue
        
        img.append(img_data) 
        name.append(l)
        if len(img)==batch_size:
            img = np.array(img)
            result = m.predict(img)[:,0,1]
            with open("re_train_1.txt", 'a+') as re:
                for k in range(batch_size):
                    re.write(f'{result[k]:.2f}:'+name[k])
            img = []
            name = []
    except:
        pass