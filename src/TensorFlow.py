import tensorflow as tf
import numpy as np
import homography
import cv2

np.set_printoptions(suppress=True)
rng = np.random

src_pts = np.load('source_points.npy')
dst_pts = np.load('destination_points.npy')
src_pts[:,0] = src_pts[:,0]/320 - 1
src_pts[:,1] = src_pts[:,1]/320 - 0.75
dst_pts[:,0] = dst_pts[:,0]/320 - 1
dst_pts[:,1] = dst_pts[:,1]/320 - 0.75

M_tru, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 0.01)
mask = np.where(mask.transpose()[0] == 0)[0]
src_pts = np.delete(src_pts,mask,0)
dst_pts = np.delete(dst_pts,mask,0)

#M_tru = np.load('matrix.npy')

M = homography.findSVD(src_pts, dst_pts)

M_fix = np.matmul(M_tru,np.linalg.inv(M))
print("tru")
print(M_tru)
print("M")
print(M)
print("fix")
print(M_fix)
print("fix*M")
print(np.matmul(M_fix,M))

src_M_pts = np.zeros(src_pts.shape)
for i in range(0, src_pts.shape[0]):
    src_M_pts[i,:] = homography.project(M,src_pts[i])

    
dst_M_pts = np.zeros(src_pts.shape)
for i in range(0, src_pts.shape[0]):
    dst_M_pts[i,:] = homography.project(M_tru,src_pts[i])

M3 = homography.findSVD(src_M_pts, dst_M_pts)
print(M3)

err = 0
for i in range(0,src_pts.shape[0]):
    b = src_pts[i]
    a1 = dst_pts[i]
    a2 = dst_M_pts[i]
    err += (a1-a2).dot(a1-a2)
print("err: "+ str(err/ src_pts.shape[0]))
#src_pts = np.asarray([[0,0],[0,10],[10,10],[10,0],[0,5],[5,5]]);
#dst_pts = np.asarray([[1,1],[1,21],[11,21],[11,1],[1,11],[6,11]]);


n_samples = src_pts.shape[0]
print(n_samples)

training_epochs = 80
display_step = 10

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


# a = tf.Variable(float(M[0,0]), name='a')
# b = tf.Variable(float(M[0,1]), name='b')
# c = tf.Variable(float(M[0,2]), name='c')
# d = tf.Variable(float(M[1,0]), name='d')
# e = tf.Variable(float(M[1,1]), name='e')
# f = tf.Variable(float(M[1,2]), name='f')
# g = tf.Variable(float(M[2,0]), name='g')
# h = tf.Variable(float(M[2,1]), name='h')

a = tf.Variable(1., name='a')
b = tf.Variable(0., name='b')
c = tf.Variable(0., name='c')
d = tf.Variable(0., name='d')
e = tf.Variable(1., name='e')
f = tf.Variable(0., name='f')
g = tf.Variable(0., name='g')
h = tf.Variable(0., name='h')

i = tf.constant(1., name='i')

x2 = (x * a) + (y * b) + c
y2 = (x * d) + (y * e) + f
z2 = ((x * g)  + (y * h) + i)

x3 = x2 / z2
y3 = y2 / z2


xt = tf.placeholder(tf.float32)
yt = tf.placeholder(tf.float32)

cost = tf.reduce_mean(tf.square(xt-x3) + tf.square(yt-y3))

learning_rate = tf.Variable(0.01, name='learning_rate')
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    
    # Run the initializer
    sess.run(init)

    c_sum = sess.run(cost, feed_dict={x:[row[0] for row in src_M_pts],y:[row[1] for row in src_M_pts],xt:[row[0] for row in dst_pts],yt:[row[1] for row in dst_pts]})
    print("sum: "+str(c_sum))

    # Fit all training data
    for epoch in range(training_epochs):
        if(epoch > 20):
            learning_rate.assign(0.005)
        if(epoch > 40):
            learning_rate.assign(0.001)
        #sess.run(optimizer, feed_dict={x:[row[0] for row in src_M_pts],y:[row[1] for row in src_M_pts],xt:[row[0] for row in dst_pts],yt:[row[1] for row in dst_pts]})
        for (p1, p2) in zip(src_pts, dst_pts):
            #if rng.rand((1)) > 0.05:
            sess.run(optimizer, feed_dict={x:p1[0],y:p1[1],xt:p2[0],yt:p2[1]})
        c_sum = sess.run(cost, feed_dict={x:[row[0] for row in src_pts],y:[row[1] for row in src_pts],xt:[row[0] for row in dst_pts],yt:[row[1] for row in dst_pts]})
        print("sum: "+str(c_sum))
 

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