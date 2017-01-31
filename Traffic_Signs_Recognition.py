# https://arxiv.org/pdf/1606.02228v2.pdf
# http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf

# Load pickled data
import pickle
import numpy as np
#import cv2

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.bincount(y_test).nonzero()[0].size

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.

#plt.imshow(X_train[30000])
#plt.show()

def preprocesing(dataset):
    '''
    #normalize Y in YUV model
    for i in range(len(dataset)):
        img = dataset[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img[:,:,0] = cv2.equalizeHist(img[:,:,0])
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        dataset[i] = img
'''
    # Convert to grayscale
    dataset = np.mean(dataset, axis=3,dtype=int)
    dataset.resize([dataset.shape[0],dataset.shape[1],dataset.shape[2], 1])

    # Normalization
    dataset = dataset/255.
    return dataset

X_train = preprocesing(X_train)
X_test = preprocesing(X_test)

#plt.imshow(X_train[1000])
#plt.show()


# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 100
rate = 0.01
std = 0.1

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten

import math
def split2batches(batch_size, features, labels):
    assert len(features) == len(labels)
    outout_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size

        batch = [features[start_i:end_i], labels[start_i:end_i]]
        outout_batches.append(batch)

    return outout_batches

def array2classifier(M, array):
    N = len(array)
    resultArray = np.zeros((N, M), float)
    resultArray[np.arange(N), array] = 1.
    return resultArray

def shuffle(features, labels):
    perm = np.random.permutation(labels.shape[0])
    shuffled_features = features[perm,:,:,:]
    shuffled_labels = labels[perm]
    return shuffled_features, shuffled_labels

def LeNet(x):
    global std

    # 28x28x6
    x = tf.reshape(x, (-1, 32, 32, 1))
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), stddev=std))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    conv1 = tf.nn.relu(conv1)

    # 14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 10x10x16
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), stddev=std))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    conv2 = tf.nn.relu(conv2)

    # 5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten
    fc1 = flatten(conv2)
    # (5 * 5 * 16, 120)
    fc1_shape = (fc1.get_shape().as_list()[-1], 120)

    fc1_W = tf.Variable(tf.truncated_normal(shape=(fc1_shape), stddev=std))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, n_classes), stddev=std))
    fc2_b = tf.Variable(tf.zeros(n_classes))
    return tf.matmul(fc1, fc2_W) + fc2_b

x = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], 1))
y = tf.placeholder(tf.float32, (None, n_classes))
learning_rate = tf.placeholder(tf.float32, shape=[])
fc2 = LeNet(x)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(loss_op)

def eval_data(X_test, y_test):
    steps_per_epoch = len(X_test) // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    batches = split2batches(BATCH_SIZE, X_test, y_test)

    for step in range(steps_per_epoch):
        batch = batches[step]
        batch_x = batch[0]
        batch_y = array2classifier(n_classes, batch[1])
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
    return total_loss/num_examples, total_acc/num_examples

train_limit = int(n_train*.8)
if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        steps_per_epoch = train_limit // BATCH_SIZE
        num_examples = steps_per_epoch * BATCH_SIZE

        # Train model
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            batches = split2batches(BATCH_SIZE, X_train[:train_limit], y_train[:train_limit])

            for step in range(steps_per_epoch):
                batch = batches[step]
                batch_x = batch[0]
                batch_y = array2classifier(n_classes, batch[1])

                loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y, learning_rate: rate})

            val_loss, val_acc = eval_data(X_train[train_limit:], y_train[train_limit:])
            print("EPOCH {} ...".format(i+1))
            print("Validation loss = {:.3f}".format(val_loss))
            print("Validation accuracy = {:.3f}".format(val_acc))

            print("rate", rate)
            rate = max(rate*0.5, 0.00001)

            print()

        # Evaluate on the test data
        test_loss, test_acc = eval_data(X_test, y_test)
        print("Test loss = {:.3f}".format(test_loss))
        print("Test accuracy = {:.3f}".format(test_acc))