from __future__ import unicode_literals, print_function, division, absolute_import

import json
import pprint
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


import logging
import os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print(tf.VERSION)
print(tf.keras.__version__)

model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

layers.Dense(64, activation='sigmoid')
layers.Dense(64, activation=tf.sigmoid)
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
layers.Dense(64, kernel_initializer='orthogonal')
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')])


model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


def random_one_hot_labels(shape):
    n, n_class = shape
    classes = np.random.randint(0, n_class, n)
    labels = np.zeros((n, n_class))
    labels[np.arange(n), classes] = 1
    return labels


data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))


val_data = np.random.random((100, 32))
val_labels = random_one_hot_labels((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))

print('Using the Datasets API')
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30)


print('Using datasets for validation')
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)

print('Evaluate and Predict')
model.evaluate(data, labels, batch_size=32)
model.evaluate(dataset, steps=30)

result = model.predict(data, batch_size=32)
print(result.shape)

inputs = tf.keras.Input(shape=(32, ))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)


model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, batch_size=32, epochs=5)

print('Model subclassing')


class MyModel(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        self.dense_1 = layers.Dense(32, activation='relu')
        self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)


model = MyModel(num_classes=10)

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, batch_size=32, epochs=5)

print('Custom layers')


class MyLayer(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


model = tf.keras.Sequential([
    MyLayer(10),
    layers.Activation('softmax')])

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, batch_size=32, epochs=5)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks, validation_data=(val_data, val_labels))

model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32, )),
    layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.save_weights('./weights/my_model')
model.load_weights('./weights/my_model')

model.save_weights('my_model.h5', save_format='h5')
model.load_weights('my_model.h5')

json_string = model.to_json()

pprint.pprint(json.loads(json_string))

fresh_model = tf.keras.models.model_from_json(json_string)

yaml_string = model.to_yaml()
print(yaml_string)

model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32, )),
    layers.Dense(10, activation='relu')
])

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model = tf.keras.estimator.model_to_estimator(model)

model = tf.keras.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10,)))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.train.GradientDescentOptimizer(0.2),
              loss='binary_crossentropy')

model.summary()


def input_fn():
    x = np.random.random((1024, 10))
    y = np.random.randint(2, size=(1024, 1))
    x = tf.cast(x, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat(10)
    dataset = dataset.batch(32)
    return dataset


strategy = tf.contrib.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(train_distribute=strategy)


keras_model = tf.keras.estimator.model_to_estimator(
    keras_model=model,
    config=config,
    model_dir='./tmp/model_dir'
)

keras_model.train(input_fn=input_fn, steps=10)