#-*- coding:utf-8 -*-
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.layers.core import Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from imp import reload
import VGG
import keys_560W

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_h = 32
img_w = 280
batch_size = 128
maxlabellength = 10
model_restore = False
total_images=500000  ### 设置每100W数据量保存一次模型
#total_images= 3279606  ## 360W训练数据集
#total_images= 4660934  ## 560W训练数据集

def get_session(gpu_fraction=1):

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def readfile(filename):
    res = []
    with open(filename, 'r',encoding='UTF-8') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic

class random_uniform_num():
    """
    鍧囧寑闅忔満锛岀‘淇濇瘡杞瘡涓彧鍑虹幇涓€娆?
    """
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0
    def get(self, batchsize):
        r_n=[]
        if(self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index : self.index + batchsize]
            self.index = self.index + batchsize

        return r_n

def gen(data_file, image_path, batchsize= batch_size, maxlabellength=10, imagesize=(32, 280)):
    image_label = readfile(data_file)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength])
    #print(labels.shape)
    #labels=[]
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            #print(i,j)
            #img1 = Image.open(os.path.join(image_path, j)).convert('L')
            img1 = Image.open(os.path.join(image_path,j)).convert('L')
            img1 = img1.resize([280, 32], Image.ANTIALIAS)
            #img = np.array(img1, 'f') / 255.0 - 0.5
            
            #img1=image_compose(j)
            #img = np.array(img1, 'f') / 255.0 - 0.5
            img = np.array(img1, 'f') / 255.0

            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape', img.shape)
            str = image_label[j]

            for v in str:
                lab_num = int(v)
                if (lab_num < 0):
                    print("Error with negative:",j, str)

            #print("Str:",str)
            label_length[i] = len(str)

            if(len(str) <= 0):
                print("len < 0", j)
            input_length[i] = imagesize[1] // 8
            labels[i, :len(str)] = [int(k) for k in str]
            #labels.append([int(k) - 1 for k in str])

        inputs = {'the_input': x,
                'the_labels': labels,
                'input_length': input_length,
                'label_length': label_length,
                }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model(img_h, nclass):
    input = Input(shape=(img_h, 280, 1), name='the_input')
    #y_pred = densenet.dense_cnn(input, nclass)
    y_pred = VGG.VGG_cnn(input, nclass,use_LSTM=False)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])

    return basemodel, model


if __name__ == '__main__':
    #
    char_set = keys_560W.CHAR_ALL_560W[0:]
    nclass = len(char_set)
    print(len(char_set))

    K.set_session(get_session())
    #reload(densenet)
    reload(VGG)
    basemodel, model = get_model(img_h, nclass)

    if model_restore == True:
        modelPath = './models/weights_vggnet_selfattention_560w-64batch_2-04-4.87.h5'
        if os.path.exists(modelPath):
            print("Loading model weights...")
            basemodel.load_weights(modelPath)
            print('done!')



    train_loader = gen(
        '/home/hj/smbshare/fffan/Data/OCR_Recognize/Synthetic_Chinese_String_Dataset/Labels/label_keras_360w/360_train_keras.txt',
        '/home/hj/smbshare/fffan/Data/OCR_Recognize/Synthetic_Chinese_String_Dataset/images/', batchsize=batch_size,
        maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    test_loader = gen(
        '/home/hj/smbshare/fffan/Data/OCR_Recognize/Synthetic_Chinese_String_Dataset/Labels/label_keras_360w/360_test_keras.txt',
        '/home/hj/smbshare/fffan/Data/OCR_Recognize/Synthetic_Chinese_String_Dataset/images/',
        batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    

    checkpoint = ModelCheckpoint(filepath='./models/weights_vggnet_only_360w-64batch_3-{epoch:02d}-{val_loss:.2f}.h5', 
                                                monitor='val_loss', save_best_only=False, save_weights_only=True)
    lr_schedule = lambda epoch: 0.00005 * 0.5**epoch
    learning_rate = np.array([lr_schedule(i) for i in range(10)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    tensorboard = TensorBoard(log_dir='./logs', write_graph=True)

    print('-----------Start training-----------')
    model.fit_generator(train_loader,
    	steps_per_epoch = total_images // batch_size,
    	epochs = 10,
    	initial_epoch = 0,
    	validation_data = test_loader,
    	validation_steps = 10000 // batch_size,
    	callbacks = [checkpoint, earlystop, changelr, tensorboard])

