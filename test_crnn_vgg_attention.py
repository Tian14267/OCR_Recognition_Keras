# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import os
#os.environ['CUDA_VISIBLE_DEVICES']='-1'
import time
import cv2
import numpy as np
from PIL import Image

from keras.models import Model
from keras.layers import Input
import VGG
import keys_560W

characters = keys_560W.CHAR_ALL_560W[0:]
nclass = len(characters)
#print("Nclass:",nclass)
#print("6506_char:",characters[6505])
input = Input(shape=(32, None, 1), name='the_input')
y_pred = VGG.VGG_cnn(input, nclass,use_LSTM=True)  ### 使用LSTM或者Attention时打开

basemodel = Model(inputs=input, outputs=y_pred)

modelPath = os.path.join(os.getcwd(), 'models/weights_vggnet_attention_560w-64batch-04-2.25.h5')

if os.path.exists(modelPath):
    print('in the loading....')
    basemodel.load_weights(modelPath)
    print('load model has done')
else:
    print('######################## modelPath is not exist! ##########################')

def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    print("len:",len(pred_text))
    for i in range(len(pred_text)):
        #print("Pred_text_num_ALL:", i, pred_text[i])
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            #print("Pred_text_num:",i,pred_text[i])
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)

def predict(img):
    print(img.size)
    width, height = img.size[0], img.size[1]
    #width, height = img.shape[1], img.shape[0]
    #(height, width) = img.shape[:2]
    #print(width,height)
    scale = height * 1.0 / 32
    width = int(width / scale)

    #print(width)
    img = img.resize([width, 32], Image.ANTIALIAS)

    img = np.array(img).astype(np.float32) / 255.0
    
    X = img.reshape([1, 32, width, 1])

    y_pred = basemodel.predict(X)
    y_pred = y_pred[:, :, :]
    print("y_pred:",y_pred.shape)

    out = decode(y_pred)
    return out


if __name__=='__main__':
    ALL_START = time.time()
    '''
    ########### one image #################
    img_path = './data_img/0#104452673#1_9.jpg'
    image=Image.open(img_path).convert('L')
    result=predict(image)
    print(result)
    '''
    ############### image  files  ##########################
    Path = '/home/hj/smbshare/fffan/Data/Test_Data_My/Business/name/'
    im_names=os.listdir(Path)
    total=0
    all_result = []
    for im_name in im_names:
        t=time.time()
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        img_path = os.path.join(Path,im_name)
        image=Image.open(img_path).convert('L')

        result=predict(image)
        print(result)
        
        split_signal = '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        all_result.append(split_signal)
        all_result.append(im_name)
        all_result.append(result)
        
        #if result==im_name.split('.')[0].decode('utf-8'):  # python 2
        if result == im_name.split('.')[0]: # python 3
            all_result.append('True')
            print(True)
            total+=1
        else:
            all_result.append('False')
            print(False)

    print(total)
    all_result.append(total)
    with open('all_result.txt','w') as f:
        for line in all_result:
            f.write(str(line)+'\n')
    f.close()
    ################################################################
    
    ALL_END = time.time()
    print("ALL_TIME: ",ALL_END-ALL_START)
