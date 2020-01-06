import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data



def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('Folder have been made!')

class Classifier():
    def __init__(self, length = 196, inND = 2, hiND = 10, outND = 10, name = 'TN'):
        self.innode = inND
        self.hiddennode = hiND
        self.outnode = outND
        self.length = length
        self.name = name
        self.var = np.sqrt(1/self.hiddennode)

    def generate_weight(self, name, shape):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            weight = tf.random_normal(shape, stddev=self.var)
            return tf.Variable(weight)

    def tensor_weights(self,):
        self.weights = []
        w_head = self.generate_weight('w_head', [self.innode, self.hiddennode])
        self.weights.append(w_head)
        for i in range(self.length-2):
            w_middle = self.generate_weight('w_middle%d' % i, [self.hiddennode, self.innode, self.hiddennode])
            self.weights.append(w_middle)
        w_end = self.generate_weight('w_end', [self.hiddennode, self.innode, self.outnode])
        self.weights.append(w_end)

    def build(self, image):
        self.image = image
        self.tensor_weights()
        self.outputs = []
        with tf.variable_scope('y_head', reuse=tf.AUTO_REUSE):
            y = tf.einsum('nj,jh->nh', self.image[0], self.weights[0])
            self.outputs.append(y)
        for i in range(length - 2):
            with tf.variable_scope('c_middle%d' % i, reuse=tf.AUTO_REUSE):
                y = tf.einsum('nh,nj,hjk->nk', y, self.image[i + 1], self.weights[i+1])
                self.outputs.append(y)
        y = tf.einsum('nh,nj,hjk->nk', y, self.image[length - 1], self.weights[length-1])
        self.outputs.append(y)

train_size = 55000
batch_size= 1024
test_size= 10000
epoch = 100
innode = 2
hiddenode = 120
outnode = 10
length = 196
# train_data = 0.2*np.reshape(np.load('original_data/train-images.npy').swapaxes(0, 1), [196, 60000, 1])
# train_label = np.load("original_data/train-labels.npy")
# test_data = 0.2*np.reshape(np.load('original_data/test-images.npy').swapaxes(0, 1), [196, 10000, 1])
# test_label = np.load('original_data/test-labels.npy')

#Preprossing dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import cv2
train_data = []
for i in range(len(mnist.train.images)):
    train_data.append(cv2.resize(np.reshape(mnist.train.images[i],[28,28]), (14, 14), interpolation=cv2.INTER_CUBIC))
a = np.zeros([55000,196])
train_data = np.array(train_data)
for i in range(14):
    for j in range(14):
        if i % 2 == 0:
            a[:,14*i+j] = train_data[:,i, j]
        else:
            a[:, 14 * i + j] = train_data[:, i, 13-j]
train_data = 0.2*np.reshape(np.array(a).swapaxes(0, 1), [196, 55000, 1])
train_label = mnist.train.labels
test_data = []
for i in range(len(mnist.test.images)):
    test_data.append(cv2.resize(np.reshape(mnist.test.images[i],[28,28]), (14, 14), interpolation=cv2.INTER_CUBIC))
b = np.zeros([10000,196])
test_data = np.array(test_data)
for i in range(14):
    for j in range(14):
        if i % 2 == 0:
            b[:,14*i+j] = test_data[:,i, j]
        else:
            b[:, 14 * i + j] = test_data[:, i, 13-j]
test_data = 0.2*np.reshape(np.array(b).swapaxes(0, 1), [196, 10000, 1])
test_label = mnist.test.labels
train_data = np.concatenate([np.cos(np.pi / 2 * train_data), np.sin(np.pi / 2 * train_data)], 2)
test_data = np.concatenate([np.cos(np.pi / 2 * test_data), np.sin(np.pi / 2 * test_data)], 2)


#Train
def train():
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        image = tf.placeholder("float", [length, None, innode], name='x_true')
        label = tf.placeholder("float", [None, outnode], name='y_true')
    Tensor_Net = Classifier(length, innode, hiddenode, outnode, name = 'Meanfield_TN')
    Tensor_Net.build(image)
    T_vars = tf.trainable_variables()
    # c_vars = [var for var in t_vars if 'c_' in var.name]
    y_predict = Tensor_Net.outputs[-1]
    loss = tf.reduce_mean(tf.square(label - y_predict))
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        train_step = tf.train.AdamOptimizer(0.00001).minimize(loss, var_list=T_vars)
    correct_predict = tf.equal(tf.argmax(y_predict, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=5)
    for ep in range(epoch):
        for iter in range(0, int(train_size/batch_size)):
            x = train_data[:, iter * batch_size: (iter + 1) * batch_size, :]
            y = train_label[iter * batch_size: (iter + 1) * batch_size, :]
            _, loss_train, acc_train = sess.run([train_step, loss, accuracy], feed_dict={image: x, label: y})
            print("Epoch: %d, Iter: %d, Loss: %f, Accuracy: %f" % (ep,iter,loss_train,acc_train))
        loss_test, acc_test = sess.run([loss, accuracy], feed_dict={image: test_data, label: test_label})
        print("Epoch: %d, Loss: %f, Accuracy: %f" % (ep, loss_test, acc_test))
        mkdir('Model')
        checkpoint_path = os.path.join(os.getcwd(), 'Model/epoch.ckpt')
        saver.save(sess, checkpoint_path, global_step=ep)
        print('*********    model saved    *********')
    sess.close()

if __name__ == '__main__':
    train()

