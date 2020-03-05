# -*- coding:utf-8 -*-
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.regularizers import l2
from keras import backend as K
from keras import initializers,regularizers,constraints
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input, Flatten,Lambda,LSTM,Bidirectional,RepeatVector,Multiply,CuDNNGRU
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute
from keras.engine.topology import Layer

my_concat = Lambda(lambda x: K.concatenate([x[0],x[1]],axis=-1))
squeeze = Lambda(lambda x: K.squeeze(x,axis=1))


class SelfAttention(Layer):

	def __init__(self,
				 W_regularizer=None, b_regularizer=None,
				 W_constraint=None, b_constraint=None,
				 bias=True, **kwargs):
		self.supports_masking = True
		self.init = initializers.get('glorot_uniform')

		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		super(SelfAttention, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 3

		self.W = self.add_weight((input_shape[-1],input_shape[-1]),
								 initializer=self.init,
								 name='{}_W'.format(self.name),
								 regularizer=self.W_regularizer,
								 constraint=self.W_constraint)


		self.b = self.add_weight((input_shape[-1],),
								 initializer='zero',
								 name='{}_b'.format(self.name),
								 regularizer=self.b_regularizer,
								 constraint=self.b_constraint)


		self.u = self.add_weight((input_shape[-1],1),
								 initializer='zero',
								 name='u',
								 regularizer=self.b_regularizer,
								 constraint=self.b_constraint)

		self.built = True

	def compute_mask(self, input, input_mask=None):
		# 后面的层不需要mask了，所以这里可以直接返回none
		return None

	def call(self, x, mask=None):

		# 这里应该是 step_dim是我们指定的参数，它等于input_shape[1],也就是rnn的timesteps

		k_dot =K.dot(x, self.W)
		eij = K.tanh(k_dot + self.b)
		k_dot_e = K.dot(eij,self.u)
		new_k_dot = K.squeeze(k_dot_e, axis=-1)
		#print("Text:", new_k_dot.shape) # (?, 35, 1)

		alphas = Activation('softmax')(new_k_dot)
		output = x * K.expand_dims(alphas, -1)

		return output

def GRU(units):
		return CuDNNGRU(units,return_sequences=True,return_state=False,  ## 输出状态层 hidden
							recurrent_initializer='glorot_uniform')



def vgg_block_2Lay(input,channel,_weight_decay = 1e-4):
	input_BN = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
	x_1 = Conv2D(channel, (3, 3), strides=(1, 1), kernel_initializer='he_normal', padding='same',
				 use_bias=False, kernel_regularizer=l2(_weight_decay))(input_BN)
	r_1 = Activation('relu')(x_1)
	x_2 = Conv2D(channel, (3, 3), strides=(1, 1), kernel_initializer='he_normal', padding='same',
				 use_bias=False, kernel_regularizer=l2(_weight_decay))(r_1)
	r_2 = Activation('relu')(x_2)
	#p_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid', dim_ordering='th')(r_2) # 长和宽同时减半
	p_1 = MaxPooling2D(pool_size=(2, 2))(r_2)
	r_3 = Activation('relu')(p_1)
	return r_3


def vgg_block_3Lay(input,channel,_weight_decay = 1e-4):
	input_BN = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
	v1 = Conv2D(channel, (3, 3), strides=(1, 1), kernel_initializer='he_normal', padding='same',
				 use_bias=False, kernel_regularizer=l2(_weight_decay))(input_BN)
	r_1 = Activation('relu')(v1)
	v2 = Conv2D(channel, (3, 3), strides=(1, 1), kernel_initializer='he_normal', padding='same',
				use_bias=False, kernel_regularizer=l2(_weight_decay))(r_1)
	r_2 = Activation('relu')(v2)
	v3 = Conv2D(channel, (3, 3), strides=(1, 1), kernel_initializer='he_normal', padding='same',
				use_bias=False, kernel_regularizer=l2(_weight_decay))(r_2)
	r_3 = Activation('relu')(v3)
	#pool = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), border_mode='valid', dim_ordering='th')(r_3)
	pool = MaxPooling2D(pool_size=(2, 1))(r_3)
	r_4 = Activation('relu')(pool)
	return r_4

def VGG_cnn(input,nclass,use_LSTM=False):
	_weight_decay = 1e-4
	## input  (32,w,3)
	block_1 = vgg_block_2Lay(input,64,_weight_decay = 1e-4)  # block_1 (16,w/2,64)

	block_2 = vgg_block_2Lay(block_1, 128, _weight_decay=1e-4) # block_2 (8,w/4,128)

	block_3 = vgg_block_3Lay(block_2,256,_weight_decay = 1e-4) # block_3 (4,w/4,256)

	block_4 = vgg_block_3Lay(block_3, 512, _weight_decay=1e-4) #  block_4 (2,w/4,512)
	BN_1 = BatchNormalization(axis=-1, epsilon=1.1e-5)(block_4)
	v1 = Conv2D(512, (1, 1), strides=(1, 1), kernel_initializer='he_normal', padding='same',
				use_bias=False, kernel_regularizer=l2(_weight_decay))(BN_1)   #  v1 (2,w/4,512)
	a_1 = Activation('relu')(v1)
	BN_2 = BatchNormalization(axis=-1, epsilon=1.1e-5)(a_1)

	v2 = Conv2D(512, (2, 2), strides=(2, 2), kernel_initializer='he_normal', padding='same',
				use_bias=False, kernel_regularizer=l2(_weight_decay))(BN_2) #  v1 (1,w/8,512)
	a_2 = Activation('relu')(v2)
	print("a_2_pred_shape:", a_2.shape)  ### (?, 1, 35, 512)

	if not use_LSTM:
		print("Only CNN Net! ")
		x = Permute((2, 1, 3), name='permute')(a_2)  # 调整各维顺序 (?, 35, 1, 512)
		to_drop = TimeDistributed(Flatten(), name='flatten')(x)
		out = Dropout(0.5)(to_drop)
	else:
		print("CNN + LSTM(GRU) + Attention !")
		x_reshape = squeeze(a_2)  # squeeze函数将y_pred的dimension中的1去掉
		output = VGG_LSTM(x_reshape)
		#out = attention_3d_block(x_reshape)
		#output = GRU(nclass)(x_reshape)  ### GRU接口
		out = SelfAttention()(output)
		#out_gru, _ = GRU(nclass)(out_attention)
		#out = Dropout(0.5)(out_attention)


	y_pred = Dense(nclass, name='out', activation='softmax')(out) ## 全连接层

	print("Y_pred_shape:", y_pred.shape)
	return y_pred

def VGG_LSTM(input):
	#
	#rnn_1 = LSTM(128, kernel_initializer="he_normal", return_sequences=True)(input)
	#rnn_1b = LSTM(128, kernel_initializer="he_normal", go_backwards=True, return_sequences=True)(rnn_1)
	#rnn1_merged = my_concat([rnn_1, rnn_1b])
	# rnn_2 = LSTM(128, kernel_initializer="he_normal", return_sequences=True)(rnn1_merged)
	# rnn_2b = LSTM(128, kernel_initializer="he_normal", go_backwards=True,return_sequences=True)(rnn_2)
	# rnn2_merged = my_concat([rnn_2, rnn_2b])
	bilstm = Bidirectional(LSTM(128, dropout_W=0.1, dropout_U=0.1, return_sequences=True))(input)
	drop_1 = Dropout(0.25)(bilstm)
	return drop_1

def attention_3d_block(inputs, single_attention_vector=False):
	# 如果上一层是 LSTM，需要 return_sequences=True
	# inputs.shape = (batch_size, time_steps, input_dim)
	time_steps = K.int_shape(inputs)[1]
	input_dim = K.int_shape(inputs)[2]
	a = Permute((2, 1))(inputs)
	a = Dense(time_steps, activation='softmax')(a)   ## 标准的一维全连接层  所实现的运算是output = activation(dot(input, kernel)+bias)
	if single_attention_vector:
		a = Lambda(lambda x: K.mean(x, axis=1))(a)
		a = RepeatVector(input_dim)(a)

	a_probs = Permute((2, 1))(a)
	# 乘上了attention权重，但是并没有求和，好像影响不大
	# 如果分类任务，进行Flatten展开就可以了
	# element-wise
	output_attention_mul = Multiply()([inputs, a_probs])  ##返回它们的逐元素积的张量
	return output_attention_mul


if __name__ == '__main__':
	input = Input(shape=(32, None, 1), name='the_input')
	VGG_cnn(input, 5000,True)
