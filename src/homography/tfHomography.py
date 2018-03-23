import numpy as np
import tensorflow as tf
import homography as ho
import cv2

def findHomographyCV(src_pts, dst_pts, threshold, epochs = 5, learning_rate=0.6):
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
    return H, mask.ravel() > 0.5

def findHomography(src_pts, dst_pts, threshold, epochs = 5, learning_rate=0.6, graph = None):
    H, mask = ho.ransac2(src_pts,dst_pts,threshold)

    src = src_pts[mask]
    dst = dst_pts[mask]

    N_1 = ho.findNormalizationMatrix(src)
    N_2 = ho.findNormalizationMatrix(dst)
    N_1inv = np.linalg.inv(N_1)
    N_2inv = np.linalg.inv(N_2)
    
    H = np.matmul(np.matmul(N_2,H),N_1inv)

    src = ho.project(N_1,src)
    dst = ho.project(N_2,dst)

    if graph is None:
        with ho.Graph() as graph:
            graph.assign(H)
            graph.train(src, dst, training_epochs=epochs, learning_rate = learning_rate)
            H = graph.currentMatrix()
    else:
        graph.assign(H)
        graph.train(src, dst, training_epochs=epochs, learning_rate = learning_rate)
        H = graph.currentMatrix()


    H = np.matmul(N_2inv,np.matmul(H,N_1))
    return H, mask




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
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

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
        tf.reset_default_graph()

    def assign(self, mat):
        mat = mat / mat[2,2]
        self.sess.run(self.a.assign(mat[0,0]))
        self.sess.run(self.b.assign(mat[0,1]))
        self.sess.run(self.c.assign(mat[0,2]))
        self.sess.run(self.d.assign(mat[1,0]))
        self.sess.run(self.e.assign(mat[1,1]))
        self.sess.run(self.f.assign(mat[1,2]))
        self.sess.run(self.g.assign(mat[2,0]))
        self.sess.run(self.h.assign(mat[2,1]))
        #self.sess.run(self.i.assign(mat[2,2]))

    def currentMatrix(self):
        newMat = np.asarray([[self.sess.run(self.a),self.sess.run(self.b),self.sess.run(self.c)],[self.sess.run(self.d),self.sess.run(self.e),self.sess.run(self.f)],[self.sess.run(self.g),self.sess.run(self.h),self.sess.run(self.i)]])
        return newMat/newMat[2,2]

    def train(self, src,dst, mask=None,  training_epochs=20, learning_rate=0.01): 
      

        if mask is not None:
            if mask.dtype != np.bool:
                raise Exception("Mask is not a boolean.")
            src = src[mask]
            dst = dst[mask]

        cost = self.sess.run(self.cost, feed_dict={self.x:[row[0] for row in src],self.y:[row[1] for row in src],self.xt:[row[0] for row in dst],self.yt:[row[1] for row in dst]})
        
        # try:
        #     print("start: "+str(cost))
        # except OSError as e:
        #     pass

        dict ={self.x:[row[0] for row in src],self.y:[row[1] for row in src],self.xt:[row[0] for row in dst],self.yt:[row[1] for row in dst],self.learning_rate:learning_rate}
        for epoch in range(training_epochs):
            # for (p1, p2) in zip(src, dst):
            #     self.sess.run(self.optimizer, feed_dict={self.x:p1[0],self.y:p1[1],self.xt:p2[0],self.yt:p2[1],self.learning_rate:learning_rate})
            self.sess.run(self.optimizer, feed_dict=dict)
            cost = self.sess.run(self.cost, feed_dict=dict)
           
            # try:
            #     print("sum: "+str(cost))
            # except OSError as e:
            #     pass

    