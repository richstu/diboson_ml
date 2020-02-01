#! /usr/bin/env python3
import tensorflow.keras as keras

def mlp(nfeats, dense_layers, nodes, loss, optimizer, activation):
  model_ = keras.models.Sequential()
  model_.add(keras.layers.Input(shape=(nfeats)))
  for i in range(dense_layers):
    model_.add(keras.layers.Dense(nodes, activation=activation, kernel_initializer="lecun_normal"))
  model_.add(keras.layers.Dense(1))
  model_.compile(loss=loss, optimizer=optimizer)
  model_.summary()
  return model_
