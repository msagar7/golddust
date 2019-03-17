import pywt
import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
#import sys
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.contrib.layers import fully_connected

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def main():
	X_fill = load_data("train_filled.csv")
	#X_eval = load_data("test_filled.csv")
	X_wv = denoise(X_fill)
	#X_evalwv = denoise(X_eval)
	X_train, Y_train, X_test, Y_test = split(X_wv)
	#parameters = fc(X_train, Y_train)
	#parameters_sae = stackedAutoencoders(X_train, X_test)
	Y_sae, Y_sae_test = stackedAutoencoders(X_train, X_test)
	#Y_sae = forward_prop_sae(X_train, parameters_sae)
	#Y_sae_test = forward_prop_sae(X_test, parameters_sae)
	# print "***** Y_SAE *******"
	# print Y_sae
	# print type(Y_sae)
	# print Y_sae.shape[0]
	# print Y_sae.shape[1]
	# print Y_sae.shape
	# print "***** Y_SAE_TEST *******"
	# print Y_sae_test
	# print type(Y_sae_test)
	# print Y_sae_test.shape
	x0_train, x1_train = Y_sae.shape
	x0_test, x0_train = Y_sae_test.shape


	Y_hat, Y_hat_train = LSTM(Y_sae, Y_train, Y_sae_test, x0_train)

	# #Y_hat = forward_prop_test_set(parameters, X_test)
	accuracy_test = metric(Y_hat, Y_test)
	accuravy_train = metric(Y_hat_train, Y_train)
	print accuravy_train
	print accuracy_test

#loads data
def load_data(filename):
	return np.loadtxt(filename, delimiter = ',')

#applies wavelet transform
def denoise(X):
	m, n = X.shape

	first_part = np.zeros((m, 28))
	third_part = np.zeros((m, 64))
	for row in range(m):
		for col1 in range(28):
			first_part[row][col1] = X[row][col1]
		for col2 in range(64):
			third_part[row][col2] = X[row][col2]

	wav = pywt.Wavelet('haar')

	D = np.zeros((m, 120))
	for i in range(len(X)):
		coeffs = pywt.wavedec(X[i][28:147], wav, mode='symmetric', level=1)
		cA, cD = coeffs
		cA = np.array(cA)
		cD = np.array(cD)
		D[i][:] = np.concatenate((cA, cD))
	
	return np.concatenate((first_part, D, third_part), axis = 1)

#splits data into X train, Y train, X test, Y test
def split(X_raw):
	m, n = X_raw.shape
	np.random.shuffle(X_raw)
	X_train = np.zeros((30000, 147))
	Y_train = np.zeros((30000, 62))
	X_test = np.zeros((10000, 147))
	Y_test = np.zeros((10000, 62))
	for row in range(m):
		if row < 30000:
			for col1 in range(1, 148):
				X_train[row][col1-1] = X_raw[row][col1]
			for col2 in range(148, 210):
				Y_train[row][col2 - 148] = X_raw[row][col2]
		else:
			for col1 in range(1, 148):
				X_test[row-30000][col1-1] = X_raw[row][col1]
			for col2 in range(148, 210):
				Y_test[row-30000][col2 - 148] = X_raw[row][col2]
	return X_train.T, Y_train.T, X_test.T, Y_test.T

def stackedAutoencoders(X_input_train, X_input_test):
	num_examples = 30000
	num_inputs=147
	num_hid1=74
	num_hid2=50
	num_hid3=num_hid1
	num_output=num_inputs
	lr=0.01
	actf=tf.nn.relu
	num_epoch=1
	batch_size=200

	#X = tf.placeholder(tf.float32,shape=[num_inputs,batch_size])
	X = tf.placeholder(tf.float32,shape=[num_inputs,30000])
	X_test = tf.placeholder(tf.float32,shape=[num_inputs,10000])
	#initializer=tf.variance_scaling_initializer()
	#initializer=tf.random_normal()

	W1 = tf.get_variable("W1", [74,147], initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable("b1", [74, 1], initializer = tf.zeros_initializer())
	W2 = tf.get_variable("W2", [50,74], initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable("b2", [50,1], initializer = tf.zeros_initializer())
	W3 = tf.get_variable("W3", [74,50], initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable("b3", [74,1], initializer = tf.zeros_initializer())
	W4 = tf.get_variable("W4", [147,74], initializer = tf.contrib.layers.xavier_initializer())
	b4 = tf.get_variable("b4", [147,1], initializer = tf.zeros_initializer())

	# W1=tf.Variable(initializer((num_hid1,num_inputs)),dtype=tf.float32)
	# W2=tf.Variable(tf.random_normal((num_hid2,num_hid1)),dtype=tf.float32)
	# W3=tf.Variable(tf.random_normal((num_hid3,num_hid2)),dtype=tf.float32)
	# W4=tf.Variable(tf.random_normal((num_output,num_hid3)),dtype=tf.float32)

	# b1=tf.Variable(tf.zeros((num_hid1,1)))
	# b2=tf.Variable(tf.zeros((num_hid2,1)))
	# b3=tf.Variable(tf.zeros((num_hid3,1)))
	# b4=tf.Variable(tf.zeros((num_output,1)))

	parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}

	hid_layer1_train = actf(tf.matmul(W1,X)+b1)
	hid_layer2_train = actf(tf.matmul(W2,hid_layer1_train)+b2)
	hid_layer3_train = actf(tf.matmul(W3,hid_layer2_train)+b3)
	output_layer = actf(tf.matmul(W4,hid_layer3_train)+b4)

####
	hid_layer1_test = actf(tf.matmul(W1,X_test)+b1)
	hid_layer2_test = actf(tf.matmul(W2, hid_layer1_test)+b2)
####
	loss = tf.reduce_mean(tf.square(output_layer-X))

	optimizer = tf.train.AdamOptimizer(lr)
	train = optimizer.minimize(loss)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		#for epoch in range(num_epoch):
			#num_batches = X_input_train.shape[0] / batch_size

			#for iteration in range(num_batches):
				#X_batch,y_batch=mnist.train.next_batch(batch_size)
			#	start_val = iteration * batch_size
			#	X_batch = X_input_train[start_val:start_val+batch_size][:]
		sess.run(train,feed_dict={X:X_input_train})
		#print b1
			#train_loss=loss.eval(feed_dict={X:X_input})
			#print("epoch {} loss {}".format(epoch,train_loss))
		y_sae_train = sess.run(hid_layer2_train, feed_dict={X:X_input_train})
		y_sae_test = sess.run(hid_layer2_test, feed_dict={X_test:X_input_test})

		# for param in parameters:
		# 	parameters[param] = sess.run(parameters[param])
		# 	print param
		# 	print parameters[param]
	#return parameters
		# print "***** Y_SAE_TRAIN *******"
		# print y_sae_train
		# print y_sae_train.shape
		# print "***** Y_SAE_TEST *******"
		# print y_sae_test
		# print y_sae_test.shape
		return y_sae_train, y_sae_test


# def forward_prop_sae(X, parameters):
# 	nx, mx = X.shape
# 	#print nx, mx
# 	X_inp = tf.placeholder(tf.float32, [nx, mx], name="X_inp")

# 	W1 = tf.placeholder(tf.float32, [74,147], name="W1")
# 	b1 = tf.placeholder(tf.float32, [74,1], name="W1")
# 	W2 = tf.placeholder(tf.float32, [50,74], name="W1")
# 	b2 = tf.placeholder(tf.float32, [50,1], name="W1")



# 	W1 = parameters["W1"]
# 	b1 = parameters["b1"]
# 	W2 = parameters["W2"]
# 	b2 = parameters["b2"]

# 	Z1 = tf.add(tf.matmul(W1, X_inp), b1)
# 	A1 = tf.nn.relu(Z1)
# 	Z2 = tf.add(tf.matmul(W2, A1), b2)
# 	A2 = tf.nn.tanh(Z2)

# 	with tf.Session() as sess:
# 		y_sae = sess.run(A2, feed_dict={X_inp: X, W1: parameters['W1'], b1: parameters['b1'], W2: parameters['W2'], b2: parameters['b2']})
# 		return y_sae

def LSTM(X,Y, X_test, x0):
	# n_hidden = 50
	# n_input = 111
	# rnn_cell = rnn.BasicLSTMCell(n_hidden)
	# W = tf.Variable(tf.random_normal([n_hidden, vocab_size]))
	# b = tf.Variable(tf.random_normal([vocab_size]))

	# initer = tf.contrib.layers.xavier_initializer()
	# W = tf.get_variable('W',
 #                           dtype=tf.float32,
 #                           shape=[40,50],
 #                           initializer=initer)
	# initial = tf.constant(0., shape=[1], dtype=tf.float32)
	# b = tf.get_variable('b',
 #                           dtype=tf.float32,
 #                           initializer=initial)

	# #X = tf.unstack(X, 111, 1)
	# rnn_cell = rnn.BasicRNNCell(40)
	# states_series, current_state = rnn.static_rnn(rnn_cell, X, dtype=tf.float32)

	# num_layers = 3
	# state_size = 50
	# input_dropout = .8
	# output_dropout = .8
	# cell = tf.nn.rnn_cell.LSTMCell(state_size)#, state_is_tuple=True)
	# cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout)
	# cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
	# x0, x1 = X.shape
	# print "**** x0 x1 *****"
	# print x0
	# print x1

	# print "***** X ******* (in LSTM)"
	# print X
	# print X.shape
	# print "** END 1 **"
	# print "***** X_TEST ******* (in LSTM)"
	# print X_test
	# print X_test.shape
	# print "** END 2 **"
	# X=X.reshape(X.shape[0],X.shape[1],1))
	# X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1))

	# Initialising the RNN
	regressor = Sequential()
	# Adding the first LSTM layer and some Dropout regularisation
	#regressor.add(LSTM(50, 'tanh', True, (50,1)))
	regressor.add(Dropout(0.1))

	# Adding a second LSTM layer and some Dropout regularisation
	#regressor.add(LSTM(50, True))
	#regressor.add(Dropout(0.2))

	# Adding a third LSTM layer and some Dropout regularisation
	#regressor.add(LSTM(50, True))
	#regressor.add(LSTM(50, return_sequences = True))
	#regressor.add(Dropout(0.2))

	# Adding a fourth LSTM layer and some Dropout regularisation
	#regressor.add(LSTM(50))
	#regressor.add(Dropout(0.2))

	# Adding the output layer
	regressor.add(Dense(62))
	# Compiling the RNN
	regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

	# Fitting the RNN to the Training set
	regressor.fit(X.T, Y.T, epochs = 25, batch_size = 200)

	Y_hat = regressor.predict(X_test.T)
	Y_hat_train = regressor.predict(X.T)

	return Y_hat, Y_hat_train

#implements fully connected layers with sizes: input(size = 147)->40->30->40->62
# def fc(X_processed, Y_processed):
# 	nx, mx = X_processed.shape
# 	ny, my = Y_processed.shape
# 	X = tf.placeholder(tf.float32, [nx, mx], name="X")
# 	Y = tf.placeholder(tf.float32, [ny, my], name="Y")

# 	W1 = tf.get_variable("W1", [40, 147], initializer = tf.contrib.layers.xavier_initializer())
# 	b1 = tf.get_variable("b1", [40, 1], initializer = tf.zeros_initializer())
# 	W2 = tf.get_variable("W2", [30, 40], initializer = tf.contrib.layers.xavier_initializer())
# 	b2 = tf.get_variable("b2", [30, 1], initializer = tf.zeros_initializer())
# 	W3 = tf.get_variable("W3", [40, 30], initializer = tf.contrib.layers.xavier_initializer())
# 	b3 = tf.get_variable("b3", [40, 1], initializer = tf.zeros_initializer())
# 	W4 = tf.get_variable("W4", [62, 40], initializer = tf.contrib.layers.xavier_initializer())
# 	b4 = tf.get_variable("b4", [62, 1], initializer = tf.zeros_initializer())
	
# 	parameters = {"W1": W1,
#                   "b1": b1,
#                   "W2": W2,
#                   "b2": b2,
#                   "W3": W3,
#                   "b3": b3,
#                   "W4": W4,
#                   "b4": b4}

# 	Z1 = tf.add(tf.matmul(W1, X), b1)
# 	A1 = tf.nn.relu(Z1)
# 	Z2 = tf.add(tf.matmul(W2, A1), b2)
# 	A2 = tf.nn.tanh(Z2)
# 	Z3 = tf.add(tf.matmul(W3, A2), b3)
# 	A3 = tf.nn.relu(Z3)
# 	Z4 = tf.add(tf.matmul(W4, A3), b4)
# 	A4 = tf.nn.tanh(Z4)

# 	# logits = tf.transpose(A4)
# 	# labels = tf.transpose(Y)

# 	# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
# 	cost = tf.reduce_mean(tf.square(tf.subtract(A4, Y)))


# 	optimizer = tf.train.AdamOptimizer(learning_rate=.0001).minimize(cost)
# 	#Initialize all the variables
# 	init = tf.global_variables_initializer()
# 	# Start the session to compute the tensorflow graph
# 	with tf.Session() as sess:
# 		# Run the initialization
# 		sess.run(init)
# 		_ , cost = sess.run([optimizer, cost], feed_dict={X: X_processed, Y: Y_processed})
# 		parameters = sess.run(parameters)
# 		print("Parameters have been trained!")
# 	return parameters

#forward propagate with test set
# def forward_prop_test_set(parameters, X_test):
# 	nx, mx = X_test.shape
# 	X = tf.placeholder(tf.float32, [nx, mx], name="X")

# 	W1 = parameters["W1"]
# 	b1 = parameters["b1"]
# 	W2 = parameters["W2"]
# 	b2 = parameters["b2"]
# 	W3 = parameters["W3"]
# 	b3 = parameters["b3"]
# 	W4 = parameters["W4"]
# 	b4 = parameters["b4"]

# 	Z1 = tf.add(tf.matmul(W1, X), b1)
# 	A1 = tf.nn.relu(Z1)
# 	Z2 = tf.add(tf.matmul(W2, A1), b2)
# 	A2 = tf.nn.tanh(Z2)
# 	Z3 = tf.add(tf.matmul(W3, A2), b3)
# 	A3 = tf.nn.relu(Z3)
# 	Z4 = tf.add(tf.matmul(W4, A3), b4)
# 	A4 = tf.nn.tanh(Z4)

# 	with tf.Session() as sess:
# 		y_hat = sess.run(A4, feed_dict={X: X_test})
# 		return y_hat

#calculates accuracy of our model
def metric(Y_hat, Y):
	Y_hat_sign = np.sign(Y_hat.T)
	Y_sign = np.sign(Y)
	results = np.equal(Y_hat_sign, Y_sign)
	num_correct = np.sum(results)
	total = results.shape[0] * results.shape[1]
	return float(num_correct) / total

if __name__ == "__main__":
	main()

