
# This is project of Avanced multimedia image processing class
# Convolutional-Auto-Encoder-base dimensionality reduction
# Author: Truong Quang Noi

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# define some const
batch_size = 100
n_epoch = 1000
n_test = 10
output_folder = "./output"

x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="input_placeholder")

def conv_encoder_decoder_model():
	# encoder
	encoder = tf.layers.conv2d(
      inputs=x,
      filters=1,
      kernel_size=[5, 5],
      strides=[2,2],
      padding="valid",
      activation=tf.nn.relu,
      name="conv_encoder")

	#decoder
	#filter = tf.ones([5,5,1,1], tf.float32)
	filter = tf.get_variable("filter", shape=(5,5,1,1), initializer=tf.zeros_initializer())
	decoder = tf.nn.conv2d_transpose(
		encoder, 
		filter, 
		[batch_size, 28, 28, 1], 
		[1, 2, 2, 1], 
		padding='VALID',
		name="conv_decoder")

	conv_encoder = tf.identity(decoder, name="conv_encoder_decode")

	return conv_encoder

def train():
	model = conv_encoder_decoder_model()

	# calculate Mean Square Error (MSE)
	dif = x - model
	mse = tf.reduce_mean(tf.square(dif), name='mse')
	optimizer = tf.train.AdamOptimizer(1e-4)

	# using MSE as the LOSS function
	loss = mse
	training_optimizer = optimizer.minimize(loss)

	init_op = tf.initialize_all_variables()

	with tf.Session() as sess:
		# init all variable
		init = tf.global_variables_initializer()
		sess.run(init)
		sess.run(init_op)
		for i in range (n_epoch):
			for j in range(600):
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
				batch_x = np.reshape(batch_xs, [-1, 28, 28, 1])
				_, mse_val = sess.run([training_optimizer, loss], feed_dict={x: batch_x})

				print ("Epoch: %4d Step: %4d MSE: %f" % (i, j, mse_val));

		saver = tf.train.Saver()
		saver.save(sess, 'models/conv_encoder_decoder_model')

		#
		print("Train finished!")

def test():

	tf.reset_default_graph()  
	new_saver = tf.train.import_meta_graph('models/conv_encoder_decoder_model.meta')

	with tf.Session() as sess:
		
		new_saver.restore(sess, tf.train.latest_checkpoint('./models/'))

		#test model
		batch_xs, batch_ys = mnist.test.next_batch(batch_size)
		batch_x = np.reshape(batch_xs, [-1, 28, 28, 1])


		graph = tf.get_default_graph()

		#[print(n.name) for n in tf.get_default_graph().as_graph_def().node]

		encoder = graph.get_tensor_by_name("conv_encoder/Relu:0")
		decoder = graph.get_tensor_by_name("conv_decoder:0")
		model = graph.get_tensor_by_name("conv_encoder_decode:0")
		encoder_kernel = graph.get_tensor_by_name("conv_encoder/kernel:0")
		mse = graph.get_tensor_by_name("mse:0")

		x = graph.get_tensor_by_name("input_placeholder:0")

		filter = graph.get_tensor_by_name("filter:0")

		result, encode_result, kernel_result, mse_result = sess.run([model, encoder, encoder_kernel, mse], feed_dict={x: batch_x})
		
		if not os.path.isdir(output_folder):
			os.mkdir(output_folder)
		else:
			shutil.rmtree(output_folder)
			os.mkdir(output_folder)

		for i in range(batch_size):
			show_result(np.reshape(batch_x[i], [28, 28]), np.reshape(result[i], [28,28]), np.reshape(encode_result[i], [12,12]), np.reshape(kernel_result[:,:,:,0], [5,5]), "figure_%d"%i)


def show_result(origin, result, code, kernel, name):
	plt.clf()
	fig=plt.figure(figsize=(8, 8))

	fig.add_subplot(2, 2, 1)
	plt.imshow(origin)
	plt.title("Origin Image (28x28)")

	fig.add_subplot(2, 2, 2)
	plt.imshow(result)
	plt.title("Restored Image (28x28)")

	fig.add_subplot(2, 2, 3)
	plt.imshow(code)
	plt.title("Reduced Dimensionality Image (12x12)")

	fig.add_subplot(2, 2, 4)
	plt.imshow(kernel)	
	plt.title("Learnt Weight (5 x 5)")

	plt.savefig(output_folder + "/"+name+".jpg")

def main(unused_argv):
	if (len(unused_argv)>1 and unused_argv[1] == 'train'):
		print ("Start training model...")
		train()
	elif(len(unused_argv)>1 and unused_argv[1] == 'test'):
		print ("Test model...")
		test()
	else:
		print("Invalid parameters! use command: \"python3 main.py test\" or \"python3 main.py train\"")


if __name__ == "__main__":
	tf.app.run()
