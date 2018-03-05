import numpy as np
import tensorflow as tf

class Graph:
    def __init__(self):  
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

        i = tf.constant(1., name='i')

        x2 = (x * a) + (y * b) + c
        y2 = (x * d) + (y * e) + f
        z2 = ((x * g)  + (y * h) + i)

        x3 = x2 / z2
        y3 = y2 / z2

        xt = tf.placeholder(tf.float32)
        yt = tf.placeholder(tf.float32)

        self.cost = tf.reduce_mean(tf.square(xt-x3) + tf.square(yt-y3))

        self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.cost)

        self.init = tf.global_variables_initializer()

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.h = h
        self.i = i
        self.x = x
        self.y = y
        self.xt = xt
        self.yt = yt


    def __enter__(self) :
        self.sess = tf.Session()
        self.sess.run(self.init)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def assign(self, mat):
        mat = mat / mat[2,2]
        self.a.assign(mat[0,0])
        self.b.assign(mat[0,1])
        self.c.assign(mat[0,2])
        self.d.assign(mat[1,0])
        self.e.assign(mat[1,1])
        self.f.assign(mat[1,2])
        self.g.assign(mat[2,0])
        self.h.assign(mat[2,1])

    def currentMatrix(self):
        newMat = np.asarray([[self.sess.run(self.a),self.sess.run(self.b),self.sess.run(self.c)],[self.sess.run(self.d),self.sess.run(self.e),self.sess.run(self.f)],[self.sess.run(self.g),self.sess.run(self.h),self.sess.run(self.i)]])
        return newMat

    def train(self, src,dst, mask=None,  training_epochs=20): 

        if mask is not None:
            src = src[mask]
            dst = dst[mask]

        for epoch in range(training_epochs):
            for (p1, p2) in zip(src, dst):
                self.sess.run(self.optimizer, feed_dict={self.x:p1[0],self.y:p1[1],self.xt:p2[0],self.yt:p2[1]})
            # self.sess.run(self.optimizer, feed_dict={self.x:[row[0] for row in src],self.y:[row[1] for row in src],self.xt:[row[0] for row in dst],self.yt:[row[1] for row in dst]})
            c_sum = self.sess.run(self.cost, feed_dict={self.x:[row[0] for row in src],self.y:[row[1] for row in src],self.xt:[row[0] for row in dst],self.yt:[row[1] for row in dst]})
            print("sum: "+str(c_sum))
    