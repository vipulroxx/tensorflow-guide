from __future__ import absolute_import, division, unicode_literals, print_function


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import logging

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.enable_eager_execution()

print(tf.executing_eagerly())

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

a = tf.constant([[1, 2],
                 [3, 4]])
print(a)

b = tf.add(a, 1)
print(b)

print(a * b)
c = np.multiply(a, b)
print(c)

print(a.numpy())

tfe = tf.contrib.eager


def fizzbuzz(max_num):
    counter = tf.constant(0)
    max_num = tf.convert_to_tensor(max_num)
    for num in range(1, max_num.numpy()+1):
        num = tf.constant(num)
        if int(num % 3) == 0 and int(num % 5) == 0:
            print('FizzBuzz')
        elif int(num % 3) == 0:
            print('Fizz')
        elif int(num % 5) == 0:
            print('Buzz')
        else:
            print(num.numpy())
        counter += 1
    print('Counter {}'.format(counter))


fizzbuzz(15)

print('Build a model')


class MySimpleLayer(tf.keras.layers.Layer):

    def __init__(self, output_units):
        super(MySimpleLayer, self).__init__()
        self.output_units = output_units

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel", [input_shape[-1], self.output_units])

    def call(self, input):
        return tf.matmul(input, self.kernel)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(784, )),
    tf.keras.layers.Dense(10)
])

model = MySimpleLayer(model)
print(model)


class MNISTModel(tf.keras.Model):

    def __init__(self):
        super(MNISTModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=10)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, input):
        result = self.dense1(input)
        result = self.dense2(result)
        result = self.dense2(result)
        return result


model = MNISTModel()
print(model)

print('Eager training Computing gradients')
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
    loss = w * w
grad = tape.gradient(loss, w)
print(grad)


print('Eager training Train a model')
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()


dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis]/255, tf.float32),
     tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.shuffle(1000).batch(32)
print(dataset)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, [3,3], activation='relu'),
    tf.keras.layers.Conv2D(16, [3,3], activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
])

for images,labels in dataset.take(1):
    print("Logits: ", mnist_model(images[0:1]).numpy())

optimizer = tf.train.AdamOptimizer()

loss_history = []

for (batch, (images, labels)) in enumerate(dataset.take(400)):
    if batch % 10 == 0:
        print('.', end='')
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)
        loss_value =  tf.losses.sparse_softmax_cross_entropy(labels, logits)

    loss_history.append(loss_value.numpy())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables), global_step=tf.train.get_or_create_global_step())

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.show()

print('\nEager training Variables and optimizers')

