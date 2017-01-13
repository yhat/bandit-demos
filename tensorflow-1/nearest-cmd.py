
'''
A nearest neighbor learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
'''
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

from bandit import Bandit
bandit = Bandit()

def main(args):
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # In this example, we limit mnist data
    Xtr, Ytr = mnist.train.next_batch(args.b) #5000 for training (nn candidates)
    Xte, Yte = mnist.test.next_batch(200) #200 for testing

    # save our batch size as part of our metadata
    bandit.metadata.batch = args.b

    # tf Graph Input
    xtr = tf.placeholder("float", [None, 784])
    xte = tf.placeholder("float", [784])

    # Nearest Neighbor calculation using L1 Distance
    # Calculate L1 Distance
    distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.neg(xte))), reduction_indices=1)
    # Prediction: Get min distance index (Nearest neighbor)
    pred = tf.arg_min(distance, 0)

    accuracy = 0.

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # loop over test data
        for i in range(len(Xte)):
            # Get nearest neighbor
            nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
            # Get nearest neighbor class label and compare it to its true label
            print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
                "True Class:", np.argmax(Yte[i]))
            # Calculate accuracy
            if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
                accuracy += 1./len(Xte)
                print('accuracy:', accuracy)
                bandit.report('Accuracy', accuracy)

        print("Done!")
        print("Accuracy:", accuracy)

        # save our accuracy as metadata
        bandit.metadata.accuracy = float(accuracy)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-b', type=int, help='batch size to train over (max is 55000)', required=True)
  args = parser.parse_args()
  print('Batch size: ', args.b )
  main(args)
