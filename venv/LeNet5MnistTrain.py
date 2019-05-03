import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
def conv_2d(x, weight, padding):
     # x:input image. weight: [filter_height, filter_width, in_channels, out_channels]
     # strides: 1-D tensor of length.The stride of the sliding window for each dimension of input
     # Must have strides[0] = strides[1] = 1, [1, stride, stride, 1]
     # padding:"SAME" or "VALID".
     # Whether to fill the boundary when the image does not match the convolution kernel
     return tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding=padding)

def max_pool_2x2(x, padding):
     # ksize:the size of pooling window [batch, height, width, channels]
     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)

def gen_weights(shape):
     # stddev:standard deviation
     # Reference: https://blog.csdn.net/abiggg/article/details/79054840
     return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

def gen_bias(shape):
     return tf.Variable(tf.constant(0.1, shape=shape))

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
# one_hot:A one-hot vector except the number of one digit is 1 and
# the other dimension numbers are 0.

input_x = tf.placeholder(tf.float32, [None, 28*28], name="input_x")
input_vec = tf.reshape(input_x, [-1, 28, 28, 1]) # input data structure

# Convolution kernel:5x5 in_channels:1 out_channels:6
convl_w = gen_weights([5, 5, 1, 6])
convl_b = gen_bias([6])

# Convolution calculation and max pooling operation
h_conv1 = tf.nn.sigmoid(conv_2d(input_vec, convl_w, "SAME") + convl_b)
h_pool1 = max_pool_2x2(h_conv1, "SAME")

conv2_w = gen_weights([5, 5, 6, 16])
conv2_b = gen_bias([16])

h_conv2 = tf.nn.sigmoid(conv_2d(h_pool1, conv2_w, "VALID") + conv2_b)
h_pool2 = max_pool_2x2(h_conv2, "SAME")

conv3_w = gen_weights([5, 5, 16, 120])
conv3_b = gen_bias([120])

h_conv2 = tf.nn.sigmoid(conv_2d(h_pool2, conv3_w, "VALID") + conv3_b)

fc1_w = gen_weights([1 * 1 * 120, 84])
fc1_b = gen_bias([84])

h_conv2 = tf.reshape(h_conv2, [-1, 1 * 1 * 120])
h_fcl = tf.sigmoid(tf.matmul(h_conv2, fc1_w) + fc1_b)

fc2_w = gen_weights([84, 10])
fc2_b = gen_bias([10])

y_conv = tf.nn.softmax(tf.matmul(h_fcl, fc2_w) + fc2_b, name = "predict")
y_ = tf.placeholder(tf.float32, [None, 10])

# model training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# accuracy figure
figure_x = []
figure_y = []
plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

epoch_count = 80000
# feed_dict: The role of feed_dict is to assign values to tensors created using placeholders.
for i in range(epoch_count):
     batch = mnist.train.next_batch(50)
     # print(batch[0])
     # print(batch[1])
     if i % 100 == 0:
          train_accuracy = accuracy.eval(feed_dict = {input_x: batch[0], y_: batch[1]})
          print("step %d, training accuracy %g" %(i, train_accuracy))
          figure_x.append(i)
          figure_y.append(train_accuracy)
          if i % 1000 == 0:
               plt.plot(figure_x, figure_y, "b--", linewidth=1)
               plt.axis([0, epoch_count, 0, 1])
               plt.show()
     train_step.run(feed_dict = {input_x: batch[0], y_: batch[1]})
plt.plot(figure_x, figure_y, "b--", linewidth=1)
plt.axis([0, epoch_count, 0, 1])
plt.show()

# saving model
print("saving model")
saver.save(sess, "./checkpoint_dir/mnist_model{0}".format(epoch_count))
print("saving finished")

print("test accuracy %g"%(accuracy.eval(feed_dict = {input_x: mnist.test.images, y_ : mnist.test.labels})))
print("training finished")




