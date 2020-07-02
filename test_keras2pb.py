# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import keys_560W
characters = keys_560W.CHAR_ALL_560W[0:]
nclass = len(characters)


class CRNN_TEST(object):
	def __init__(self):
		self.height = 32
		self.input = tf.placeholder(tf.float32, [1, self.height, None, 1])
		pb_path = "./models/weights_vggnet_560w-03-32-0.47_138.pb"
		print("Input: ",self.input.get_shape())

		sess_config = tf.ConfigProto(allow_soft_placement=True)
		sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
		sess_config.gpu_options.allow_growth = True

		self.sess = tf.Session(config = sess_config)
		'''
		with tf.Graph().as_default():
			output_graph_def = tf.GraphDef()
			with open(pb_path, "rb") as f:
				output_graph_def.ParseFromString(f.read())
				tf.import_graph_def(output_graph_def, name="")

				self.input_image_tensor = self.sess.graph.get_tensor_by_name("the_input:0")
				self.output_tensor_name = self.sess.graph.get_tensor_by_name("out/truediv:0")
		'''
		f = gfile.FastGFile(pb_path, 'rb')
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		# Add the graph to the session
		tf.import_graph_def(graph_def, name='')

		graph = tf.get_default_graph()
		self.input_image_tensor = graph.get_tensor_by_name("the_input:0")
		self.output_tensor_name = graph.get_tensor_by_name("out/truediv:0")


		print("################ load CRNN model down! ##########################")

	def _close(self):
		self.sess.close()

	def decode(self, pred):
		char_list = []
		pred_text = pred.argmax(axis=2)[0]
		# print("len:",len(pred_text))
		for i in range(len(pred_text)):
			# print("Pred_text_num_ALL:", i, pred_text[i])
			if pred_text[i] != nclass - 1 and (
					(not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
				# print("Pred_text_num:",i,pred_text[i])
				char_list.append(characters[pred_text[i]])
		return u''.join(char_list)

	def predice_cv(self,img):
		(height, width) = img.shape
		scale = height * 1.0 / 32
		width = int(width / scale)
		img_test = cv2.resize(img, (width, 32))
		img = np.array(img_test).astype(np.float32) / 255.0
		X = img.reshape([1, 32, width, 1])

		y_pred = self.sess.run(self.output_tensor_name, feed_dict={self.input_image_tensor: X})

		y_pred = y_pred[:, :, :]
		out = self.decode(y_pred)

		return out

def Textcnn_pb(pb_path):
	image_path = "./Images/0.jpg"
	image_origin = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
	img_gray = cv2.cvtColor(image_origin, cv2.COLOR_RGB2GRAY)

	(height, width) = img_gray.shape
	scale = height * 1.0 / 32
	width = int(width / scale)
	img_test = cv2.resize(img_gray, (width, 32))
	img = np.array(img_test).astype(np.float32) / 255.0
	X = img.reshape([1, 32, width, 1])

	def decode(pred):
		char_list = []
		pred_text = pred.argmax(axis=2)[0]
		# print("len:",len(pred_text))
		for i in range(len(pred_text)):
			# print("Pred_text_num_ALL:", i, pred_text[i])
			if pred_text[i] != nclass - 1 and (
					(not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
				# print("Pred_text_num:",i,pred_text[i])
				char_list.append(characters[pred_text[i]])
		return u''.join(char_list)

	with tf.Graph().as_default():
		output_graph_def = tf.GraphDef()
		with open(pb_path, "rb") as f:
			output_graph_def.ParseFromString(f.read())
			tf.import_graph_def(output_graph_def, name="")
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			# 定义输入的张量名称,对应网络结构的输入张量
			# input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
			input_image_tensor = sess.graph.get_tensor_by_name("the_input:0")
			# 定义输出的张量名称
			output_tensor_name = sess.graph.get_tensor_by_name("out/truediv:0")

			y_pred=sess.run(output_tensor_name, feed_dict={input_image_tensor: X})
			y_pred = y_pred[:, :, :]
			out = decode(y_pred)
			print(out)

if __name__ == '__main__':
	######################################

	crnn = CRNN_TEST()
	image_path = "./Images/0.jpg"
	image_origin = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
	img_gray = cv2.cvtColor(image_origin, cv2.COLOR_RGB2GRAY)
	result = crnn.predice_cv(img_gray)
	print("####  ",result)
	'''
	result = Textcnn_pb(pb_path="./models/weights_vggnet_560w-03-32-0.47_138.pb")
	'''
