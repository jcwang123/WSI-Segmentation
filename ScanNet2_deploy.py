'''
Created on 17 Jun 2017

@author: hjlin
'''
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Reshape, Add, BatchNormalization, Activation, AveragePooling2D, Concatenate, Cropping2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, \
    Conv2D, UpSampling2D, Conv2DTranspose
from keras.optimizers import SGD
from keras import activations
from keras.engine.topology import InputLayer
from keras.layers import merge, concatenate
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda
from keras import backend as K
import tensorflow as tf


def GetModel(flag='train', size=244):
    '''
       Get The model of FCN of VGG16 - 1024
    '''
    input = Input((size, size, 3))
    x = Conv2D(64, (3, 3), activation="relu", padding='valid', name="conv1_1")(input)
    x = Conv2D(64, (3, 3), activation="relu", padding='valid', name="conv1_2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    x = Conv2D(128, (3, 3), activation="relu", padding='valid', name="conv2_1")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding='valid', name="conv2_2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    x = Conv2D(256, (3, 3), activation="relu", padding='valid', name="conv3_1")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding='valid', name="conv3_2")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding='valid', name="conv3_3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    x = Conv2D(512, (3, 3), activation="relu", padding='valid', name="conv4_1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding='valid', name="conv4_2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding='valid', name="conv4_3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
    x = Conv2D(512, (3, 3), activation="relu", padding='valid', name="conv5_1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding='valid', name="conv5_2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding='valid', name="conv5_3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
    x = Conv2D(1024, (2, 2), activation="relu", padding='valid', name="conv6")(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1024, (1, 1), activation="relu", padding='valid', name="conv7")(x)
    x = Dropout(0.5)(x)
    x = Conv2D(2, (1, 1), activation="softmax", padding='valid', name="conv8_2")(x)
    if flag == 'train':
        x = Reshape((-1, 2))(x)
    model = Model(input=input, output=x)
    return model


def GetModel2(load_weights = "", total_weights="", flag='train', size=244):
    m = GetModel()
    if load_weights:
        m.load_weights(load_weights)
    i = m.input
    o1 = m.output
    f2 = m.get_layer('conv2_2').output#116 128
    f3 = m.get_layer('conv3_3').output#52 256
    f4 = m.get_layer('conv4_3').output#20 512 
    f5 = m.get_layer('conv5_3').output#4 512

    f4 = UpSampling2D((3,3))(f4) 
    f4 = Conv2D(128, (1,1))(f4)

    f3 = ZeroPadding2D((4,4))(f3)
    f3_1 = Conv2D(128, (1,1))(f3)
    f3_2 = Conv2D(128, (3,3), dilation_rate=(6,6), padding='same')(f3)
    f3_3 = Conv2D(128, (3,3), dilation_rate=(12,12), padding='same')(f3)

    f3 = concatenate([f3_1, f3_2, f3_3], axis=-1)
    
    f3 = UpSampling2D((2,2))(f3)
    f3 = Conv2D(128, (1,1))(f3)

    f2 = ZeroPadding2D((2,2))(f2)
    f2 = concatenate([f2,f3], axis=-1)
    f2 = Conv2D(64, (3,3), padding='same')(f2)
    o2 = Conv2D(2, (1,1), activation='softmax')(f2)
    o2 = Reshape((-1, 2))(o2)
    m2 = Model(i, [o1, o2])

    if total_weights:
        m2.load_weights(total_weights)
    return m2    
def gcn(input, padding,k, c):
    left = Conv2D(c, (k, 1), padding=padding)(input)
    left = Conv2D(c, (1, k), padding=padding)(left)
    right = Conv2D(c, (1, k), padding=padding)(input)
    right = Conv2D(c, (k, 1), padding=padding)(right) 
    return Add()([left, right])
def conv_bn_relu(x, filters, padding,name):
    x = Conv2D(filters, (3,3), padding=padding, name=name+'_conv')(x)
    x = BatchNormalization(axis=-1, name=name+'_bn')(x)
    x = Activation(activation='relu', name=name+'_relu')(x)
    return x

def GetModel3(load_weights = "", total_weights="", flag='train', size=244, k=13):
    m = GetModel()
    if load_weights:
        m.load_weights(load_weights)
    i = m.input
    o1 = m.output
    f2 = m.get_layer('conv2_2').output#116 128
    f3 = m.get_layer('conv3_3').output#52 256
    f4 = m.get_layer('conv4_3').output#20 512 
    f5 = m.get_layer('conv5_3').output#4 512

    f5 = Conv2DTranspose(512, (3,3), strides=(3,3), padding='same', activation='relu')(f5)
    f4 = Concatenate(axis=-1)([Cropping2D(4)(f4),f5])
    f4 = Conv2D(512, (3,3), padding='same', activation='relu')(f4)
    f4 = Conv2D(512, (3,3), padding='same', activation='relu')(f4)
    f4 = Conv2DTranspose(256, (3,3), strides=(3,3), padding='same', activation='relu')(f4)

    f3 = Concatenate(axis=-1)([Cropping2D(8)(f3),f4])
    f3 = Conv2D(256, (3,3), padding='same', activation='relu')(f3)
    f3 = Conv2D(256, (3,3), padding='same', activation='relu')(f3)
    f3 = Conv2DTranspose(128, (3,3), strides=(3,3), padding='same', activation='relu')(f3)

    f2 = Concatenate(axis=-1)([Cropping2D(4)(f2),f3])
    f2 = Conv2D(128, (3,3), padding='same', activation='relu')(f2)
    f2 = Conv2D(128, (3,3), padding='same', activation='relu')(f2)
    
    o2 = Conv2D(2, (1,1), activation='softmax')(f2)
    o2 = Reshape((-1, 2))(o2)
    m2 = Model(i, [o1, o2])

    if total_weights:
        m2.load_weights(total_weights)
    return m2    
if __name__ == "__main__":
    model = GetModel("Net2A_FP_Single_VL.tfmodel")
    model.summary()
