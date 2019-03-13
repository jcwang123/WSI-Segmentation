import argparse
from Models import VGGUnet, VGGSegnet, FCN8, FCN32, FCN, ResNet
from cnn_finetune import resnet_101
import Models
import LoadBatches
import openslide
import os
from adabound import AdaBound

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import keras.callbacks as cb
import keras.backend as K
from keras.optimizers import Adam, RMSprop, SGD
import ScanNet2_deploy
import numpy as np
import random

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument("--logs", type=str)
parser.add_argument("--n_classes", type=int, default=2)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=48)
parser.add_argument("--multi_gpu", type=int, default=2)
parser.add_argument("--load_weights", type=str, default="")
parser.add_argument("--model_name", type=str, default="scan")
parser.add_argument("--model_size", type=int, default=244)
parser.add_argument("--patch_number", type=int, default=5)
parser.add_argument("--initial_epoch", type=int, default=0)
args = parser.parse_args()


steps = int(100000/args.batch_size)
train_batch_size = args.batch_size
n_classes = args.n_classes
validate = True
epochs = args.epochs
load_weights = args.load_weights
model_name = args.model_name

if validate:
    val_batch_size = args.batch_size
print(args.model_name)
modelFns = {'fcn': FCN.FCN, 'vgg_segnet': VGGSegnet.VGGSegnet, 'vgg_unet': VGGUnet.VGGUnet,
            'vgg_unet2': VGGUnet.VGGUnet2, 'fcn8': FCN8.FCN8, 'fcn32': FCN32.FCN32}
if model_name == 'scan':
    m = ScanNet2_deploy.GetModel()
elif model_name == 'res':
    m = ResNet.GetModel(args.model_size)
elif model_name == 'res101':
    m = resnet_101.resnet101_model(244, 244, 3, 2)
else:
    modelFN = modelFns[model_name]
    m = modelFN(n_classes)
from keras.utils import multi_gpu_model

if args.multi_gpu > 1:
    m = multi_gpu_model(m, gpus=args.multi_gpu)
    print("multi training...")
if args.load_weights:
    m.load_weights(args.load_weights, by_name=True)
# m.compile(loss='categorical_crossentropy',
#           optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
#           metrics=['accuracy'])
m.compile(loss='categorical_crossentropy',
          optimizer=Adam(lr=1e-04, ),
          metrics=['accuracy'])
m.summary()
filepath = "/model_{epoch:02d}-{val_acc:.2f}.hdf5"
modelCheckpoint = cb.ModelCheckpoint('logs/' + args.logs + filepath, monitor='val_loss', verbose=1,
                                     save_best_only=False)
tensorboard = cb.TensorBoard(log_dir='logs/tensor/' + args.logs)
# reduce_lr = cb.LearningRateScheduler(scheduler)
callback = [modelCheckpoint, tensorboard]

# val_ids = [30, 118, 27, 94, 103, 109, 97, 69, 63, 73, 47, 102, 54, 3, 82, 57, 49, 135, 59, 14]
# train_ids = []
# for i in range(1, 141):
#     if i not in val_ids:
#         train_ids.append(i)
# print("train_ids:\t", train_ids)
# print("val_ids:\t", val_ids)
random.seed()
G = LoadBatches.DataGenerator('train', args.model_size, train_batch_size, n_classes, arg=True, patch_number=args.patch_number)
if validate:
    G2 = LoadBatches.DataGenerator('val', args.model_size, val_batch_size, n_classes, arg=False, patch_number=args.patch_number)

# for i in range(epochs):
history = m.fit_generator(G, steps, validation_data=G2, validation_steps=int(steps / 5),
                          epochs=epochs, use_multiprocessing=True, workers=10,
                          initial_epoch=args.initial_epoch,
                          max_queue_size=10,
                          callbacks=callback)
print(history.history)
