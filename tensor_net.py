import tensorflow as tf
import numpy as np

#input data (100 phony data points)
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.1, 0.2], x_data) + 0.3

#constructing a linear model
b = tf.Variable(tf.zeros(1))
w = tf.Variable(tf.random_uniform([1, 2], -1, 1))
y = tf.matmul(w, x_data) + b

#gradient descent time
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#init
init = tf.global_variables_initializer()

#launch the graph
sess = tf.Session()
sess.run(init)

#train -- fit the plane
for step in range(0, 200):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(w), sess.run(b))
