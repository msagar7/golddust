import pywt
import numpy as np
import tensorflow as tf

def main():
	X_fill = load_data("train_filled.csv")
	X_wv = denoise(X_fill)
	X_train, Y_train, X_test, Y_test = split(X_wv)
	parameters = fc(X_train, Y_train)
	Y_hat = forward_prop_test_set(parameters, X_test)
	accuracy = metric(Y_hat, Y_test)
	print accuracy

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

#implements fully connected layers with sizes: input(size = 147)->40->30->40->62
def fc(X_processed, Y_processed):
	nx, mx = X_processed.shape
	ny, my = Y_processed.shape
	X = tf.placeholder(tf.float32, [nx, mx], name="X")
	Y = tf.placeholder(tf.float32, [ny, my], name="Y")

	W1 = tf.get_variable("W1", [40, 147], initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable("b1", [40, 1], initializer = tf.zeros_initializer())
	W2 = tf.get_variable("W2", [30, 40], initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable("b2", [30, 1], initializer = tf.zeros_initializer())
	W3 = tf.get_variable("W3", [40, 30], initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable("b3", [40, 1], initializer = tf.zeros_initializer())
	W4 = tf.get_variable("W4", [62, 40], initializer = tf.contrib.layers.xavier_initializer())
	b4 = tf.get_variable("b4", [62, 1], initializer = tf.zeros_initializer())
	
	parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4}

	Z1 = tf.add(tf.matmul(W1, X), b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2, A1), b2)
	A2 = tf.nn.tanh(Z2)
	Z3 = tf.add(tf.matmul(W3, A2), b3)
	A3 = tf.nn.relu(Z3)
	Z4 = tf.add(tf.matmul(W4, A3), b4)
	A4 = tf.nn.tanh(Z4)

	logits = tf.transpose(A4)
	labels = tf.transpose(Y)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

	optimizer = tf.train.AdamOptimizer(learning_rate=.0001).minimize(cost)
	#Initialize all the variables
	init = tf.global_variables_initializer()
	# Start the session to compute the tensorflow graph
	with tf.Session() as sess:
		# Run the initialization
		sess.run(init)
		_ , cost = sess.run([optimizer, cost], feed_dict={X: X_processed, Y: Y_processed})
		parameters = sess.run(parameters)
		print("Parameters have been trained!")
	return parameters

#forward propagate with test set
def forward_prop_test_set(parameters, X_test):
	nx, mx = X_test.shape
	X = tf.placeholder(tf.float32, [nx, mx], name="X")

	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	W3 = parameters["W3"]
	b3 = parameters["b3"]
	W4 = parameters["W4"]
	b4 = parameters["b4"]

	Z1 = tf.add(tf.matmul(W1, X), b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2, A1), b2)
	A2 = tf.nn.tanh(Z2)
	Z3 = tf.add(tf.matmul(W3, A2), b3)
	A3 = tf.nn.relu(Z3)
	Z4 = tf.add(tf.matmul(W4, A3), b4)
	A4 = tf.nn.tanh(Z4)

	with tf.Session() as sess:
		y_hat = sess.run(A4, feed_dict={X: X_test})
		return y_hat

#calculates accuracy of our model
def metric(Y_hat, Y):
	Y_hat_sign = np.sign(Y_hat)
	Y_sign = np.sign(Y)
	results = np.equal(Y_hat_sign, Y_sign)
	num_correct = np.sum(results)
	total = results.shape[0] * results.shape[1]
	return float(num_correct) / total

if __name__ == "__main__":
	main()

