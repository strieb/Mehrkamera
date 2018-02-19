import tensorflow as tf
import numpy as np

rng = np.random

src_pts = np.load('source_points.npy')
dst_pts = np.load('destination_points.npy')

M = np.load('matrix.npy')
print(M)

#src_pts = np.asarray([[0,0],[0,10],[10,10],[10,0],[0,5],[5,5]]);
#dst_pts = np.asarray([[1,1],[1,21],[11,21],[11,1],[1,11],[6,11]]);

n_samples = src_pts.shape[0]
print(n_samples)

learning_rate = 0.00002
training_epochs = 10000
display_step = 10

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

a = tf.Variable(1., name='a')
b = tf.Variable(0., name='b')
c = tf.Variable(0., name='c')
d = tf.Variable(0., name='d')
e = tf.Variable(1., name='e')
f = tf.Variable(0., name='f')
g = tf.Variable(0., name='g')
h = tf.Variable(0., name='h')
i = tf.Variable(1000., name='i')

x2 = (x * a) + (y * b) + c
y2 = (x * d) + (y * e) + f
z2 = ((x * g) + (y * h) + i) * 0.001

x3 = x2 / z2
y3 = y2 / z2


xt = tf.placeholder(tf.float32)
yt = tf.placeholder(tf.float32)

cost = tf.reduce_mean(tf.square(x3-xt) + tf.square(y3-yt))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    print("x: ", sess.run(x3,feed_dict={x:10,y:10}))
    print("y: ", sess.run(y3,feed_dict={x:10,y:10}))
    
    # Fit all training data
    for epoch in range(training_epochs):
    #    for (p1, p2) in zip(src_pts, dst_pts):
    #        sess.run(optimizer, feed_dict={x:p1[0],y:p1[1],xt:p2[0],yt:p2[1]})
        sess.run(optimizer, feed_dict={x:[row[0] for row in src_pts],y:[row[1] for row in src_pts],xt:[row[0] for row in dst_pts],yt:[row[1] for row in dst_pts]})

    print("x: ", sess.run(x3,feed_dict={x:10,y:10}))
    print("y: ", sess.run(y3,feed_dict={x:10,y:10}))

    print("Optimization Finished!")
    print("a: ", sess.run(a))
    print("b: ", sess.run(b))
    print("c: ", sess.run(c))
    print("d: ", sess.run(d))
    print("e: ", sess.run(e))
    print("f: ", sess.run(f))
    print("g: ", sess.run(g))
    print("h: ", sess.run(h))
    print("i: ", sess.run(i))
    





writer = tf.summary.FileWriter('test')
writer.add_graph(tf.get_default_graph())
writer.flush()