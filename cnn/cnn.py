import tensorflow as tf
import otto_data as od
import numpy as np
import pandas as pd

def weight_variable(shape):
  return tf.Variable(shape,tf.contrib.layers.xavier_initializer())  #tf.contrib.layers.xavier_initializer(shape)
  #initial = tf.truncated_normal(shape, stddev=0.1)
  #return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  #print x.get_shape()
  #print W.get_shape()
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                        strides=[1, 1, 2, 1], padding='SAME')

data=od.process()
data.read_data_sets()

x = tf.placeholder(tf.float32, shape=[None, 93])
y_ = tf.placeholder(tf.float32, shape=[None, 9])

W_conv1 = tf.get_variable('W_conv1',shape=[1, 3, 1, 32],initializer=tf.contrib.layers.xavier_initializer())#weight_variable([1, 3, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,1,93,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = tf.get_variable('W_conv2',shape=[1, 3, 32, 64],initializer=tf.contrib.layers.xavier_initializer())#weight_variable([1, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = tf.get_variable('W_fc1',shape=[1 * 24 * 64, 1024],initializer=tf.contrib.layers.xavier_initializer())#weight_variable([1 * 24 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 1*24*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.get_variable('W_fc2',shape=[1024, 9],initializer=tf.contrib.layers.xavier_initializer())#weight_variable([1024, 9])
b_fc2 = bias_variable([9])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

reg_constant=0
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
#+reg_constant*tf.nn.l2_loss(W_conv1)
#+reg_constant*tf.nn.l2_loss(W_conv2)
#+reg_constant*tf.nn.l2_loss(W_fc1)
#+reg_constant*tf.nn.l2_loss(W_fc2)
#)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
for i in range(1000):
  batch = data.next_batch(4096)
  if i%10 == 0:
    train_accuracy = sess.run(accuracy,feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

save_path = saver.save(sess, "./model.ckpt")
print("Model saved in file: %s" % save_path)

test=data.testset()
print("test accuracy %g"%(sess.run(accuracy,feed_dict={
    x: test[0], y_: test[1], keep_prob: 1.0})))

result=sess.run(tf.nn.softmax(y_conv),feed_dict={
    x: test[0], y_: test[1], keep_prob: 1.0});

rre=np.insert(result,0,test[2],axis=1)
rre=pd.DataFrame(rre,columns=['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
rre.to_csv('graph.csv',index=False)

result=np.maximum(np.minimum(result,1-10**(-15)),10**-15)
print -1.0/result.shape[0]*np.sum(test[1]*np.log(result))



#just for Kaggle evaluate
quit()
rx = pd.read_csv('../test.csv')
rx=rx.values
rid=rx[:,0]
rx=rx[:,1:]
ry=np.zeros((rx.shape[0],9))

rpy=sess.run(tf.nn.softmax(y_conv),feed_dict={
    x: rx, y_: ry, keep_prob: 1.0});
    
rpy=np.insert(rpy,0,rid,axis=1)
rpy=pd.DataFrame(rpy,columns=['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
rpy.to_csv('result.csv',index=False)
