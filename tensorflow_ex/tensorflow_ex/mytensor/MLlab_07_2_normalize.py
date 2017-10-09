import numpy as np
import tensorflow as tf

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])
xy = MinMaxScaler(xy)
print(xy)

x_data = xy[:,0:-1]
y_data = xy[:,[-1]]


# Evaluation our model using this test dataset

X = tf.placeholder("float",[None,4])
Y = tf.placeholder("float",[None,1])
W = tf.Variable(tf.random_normal([4,1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(X,W)+b
cost = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(1001):
		cost_val,hy_val, _ = sess.run([cost,hypothesis,optimizer],feed_dict={X:x_data,Y:y_data})
		if step % 100 == 0:
			print(step,"cost:",cost_val,"prediction:",hy_val)
		
	'''
	
[[ 0.99999999  0.99999999  0.          1.          1.        ]
 [ 0.70548491  0.70439552  1.          0.71881782  0.83755791]
 [ 0.54412549  0.50274824  0.57608696  0.606468    0.6606331 ]
 [ 0.33890353  0.31368023  0.10869565  0.45989134  0.43800918]
 [ 0.51436     0.42582389  0.30434783  0.58504805  0.42624401]
 [ 0.49556179  0.42582389  0.31521739  0.48131134  0.49276137]
 [ 0.11436064  0.          0.20652174  0.22007776  0.18597238]
 [ 0.          0.07747099  0.5326087   0.          0.        ]]
	'''
	