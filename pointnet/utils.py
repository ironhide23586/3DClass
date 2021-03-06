# import numpy as np
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Layer, Conv2D, BatchNormalization, Dense, Dropout


# from models.pointnet_cls_F import custom_dense,custom_conv
# from custommodel import CustomModel

def custom_conv(x, filters=32, activation=tf.nn.relu, bn_momentum=0.99):
    x = tf.keras.layers.Conv1D(filters, kernel_size=1, padding="valid", use_bias=True)(x)
    # x = keras.layers.BatchNormalization(momentum=bn_momentum)(x)
    return activation(x)


def custom_dense(x, filters=32):
    x = tf.keras.layers.Dense(filters, use_bias=True)(x)
    # x = keras.layers.BatchNormalization(momentum=0.0)(x)
    return tf.nn.relu(x)
