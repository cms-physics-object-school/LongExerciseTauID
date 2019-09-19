from keras.models import load_model
import tensorflow as tf
import numpy as np
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_solution_part1 import variables

# Load keras model
model = load_model("model.h5", compile=False)

# Get weights as numpy arrays
weights = {}
for layer in model.layers:
    for weight, array in zip(layer.weights, layer.get_weights()):
        weights[weight.name] = np.array(array)

# Load weights in tensorflow variables
w1 = tf.get_variable('w1', initializer=weights['dense_1/kernel:0'])
b1 = tf.get_variable('b1', initializer=weights['dense_1/bias:0'])
w2 = tf.get_variable('w2', initializer=weights['dense_2/kernel:0'])
b2 = tf.get_variable('b2', initializer=weights['dense_2/bias:0'])

# Build tensorflow graph with weights from keras model
x = tf.placeholder(tf.float32)
l1 = tf.tanh(tf.add(b1, tf.matmul(x, w1)))
f = tf.sigmoid(tf.add(b2, tf.matmul(l1, w2)))

# Add gradient computation to tensorflow graph
df = tf.gradients(f, x)[0]

# Load data
x_in = pickle.load(open("x.pickle", "rb"))

# Compute gradients
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    df_ = sess.run(df, feed_dict={x: x_in})

# Compute scores
scores = np.mean(np.abs(df_), axis=0)

# Make plot
plt.figure(figsize=(len(variables), 8))
r = range(len(variables))
plt.plot(r, scores, "+", ms=8, mew=3, color="r")
plt.yscale("log")
plt.ylim(0.005,100)
plt.grid()
plt.gca().set_xticks(r)
plt.gca().set_xticklabels(variables, rotation=90)
plt.xlim((r[0] - 0.5, r[-1] + 0.5))
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("ranking.png")
