# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:36:59 2019

@author: Administrator
"""

from keras.layers import Input,Conv2D,MaxPooling2D,BatchNormalization,Dense,LSTM,Dropout,Reshape
from keras import backend as K
from keras.layers import Lambda

from keras.layers.merge import concatenate


my_concat = Lambda(lambda x: K.concatenate([x[0],x[1]],axis=-1))
squeeze = Lambda(lambda x: K.squeeze(x,axis=1))


def crnn(input,nclass):
    conv_1=Conv2D(64,(3,3),activation='relu',padding='same',name='conv_1')(input)
    batchnorm_1 = BatchNormalization()(conv_1)
    pool_1=MaxPooling2D(pool_size=(2,2))(batchnorm_1)
    pool_1_shape = pool_1.get_shape()
    print('pool_1: '+str(pool_1_shape))
    
    conv_2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
    conv_2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_2_1)
    pool_2=MaxPooling2D(pool_size=(2,2))(conv_2_2)
    pool_2_shape = pool_2.get_shape()
    print('pool_2: '+str(pool_2_shape))
    
    conv_3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)
    conv_3_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3_1)
    batchnorm_3 = BatchNormalization()(conv_3_2)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(batchnorm_3)
    pool_3_shape = pool_3.get_shape()
    print('pool_3: '+str(pool_3_shape))

    conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_3)
    conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv_4)
    batchnorm_5 = BatchNormalization()(conv_5)
    pool_4 = MaxPooling2D(pool_size=(2, 1))(batchnorm_5)
    pool_4_shape = pool_4.get_shape()
    print('pool_4: '+str(pool_4_shape))

    conv_6 = Conv2D(512, (2, 2), activation='relu', padding='valid')(pool_4)
    #conv_7 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv_6)
    batchnorm_7 = BatchNormalization()(conv_6)
    bn_shape = batchnorm_7.get_shape()
    
    print('bn_shape: '+ str(bn_shape))  # (?, 1, 24, 512)

    x_reshape=squeeze(batchnorm_7) #squeeze函数将y_pred的dimension中的1去掉

    #x_reshape = Reshape((int(bn_shape[1]*bn_shape[2]),int(bn_shape[3])))(batchnorm_7)
    print('x_reshape: '+str(x_reshape.get_shape()))

    rnn_1 = LSTM(128, kernel_initializer="he_normal", return_sequences=True)(x_reshape)
    rnn_1b = LSTM(128, kernel_initializer="he_normal", go_backwards=True, return_sequences=True)(x_reshape)
    rnn1_merged = my_concat([rnn_1, rnn_1b])

    rnn_2 = LSTM(128, kernel_initializer="he_normal", return_sequences=True)(rnn1_merged)
    rnn_2b = LSTM(128, kernel_initializer="he_normal", go_backwards=True, return_sequences=True)(rnn1_merged)
    rnn2_merged = my_concat([rnn_2, rnn_2b])
    
    print('rnn2_merged: '+str(rnn2_merged.get_shape()))
    
    drop_1 = Dropout(0.25)(rnn2_merged)

    y_pred = Dense(nclass, name='out', activation='softmax')(drop_1)

    #fc_2 = Dense(nclass, kernel_initializer='he_normal', activation='softmax')(drop_1)
    
    print('y_pred: '+str(y_pred.get_shape()))
    return y_pred
    


#input = Input(shape=(32, 280, 1), name='the_input')
#crnn(input, 5931)


