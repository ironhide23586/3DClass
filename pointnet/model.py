"""  _
    |_|_
   _  | |
 _|_|_|_|_
|_|_|_|_|_|_
  |_|_|_|_|_|
    | | |_|
    |_|_
      |_|
Author: Souham Biswas
Website: https://www.linkedin.com/in/souham/
"""

import os
from datetime import datetime
# from multiprocessing import cpu_count

import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np

from pointnet.tnet import TNet
from pointnet.utils import custom_conv
import utils_
from data_io import PlySet


class KerasData(Sequence):

    def __init__(self, ply_fps):
        super().__init__()
        self.point_data = PlySet(ply_fps)
        # self.point_data.match_scales()

    def __len__(self):
        return utils_.BATCHES_PER_EPOCH

    def __getitem__(self, idx):
        return utils_.sample_data(self.point_data, random_transform=True)


class PointNet:

    def __init__(self, mode='infer'):
        self.bn_momentum = 0.99
        self.model = self.make_model(mode)
        if mode == 'train':
            # self.model.compile('adam', loss='sparse_categorical_crossentropy',
            #                    metrics=['sparse_categorical_accuracy'])
            self.model.compile('adam', loss=self.loss_func,
                               metrics=['sparse_categorical_accuracy'])
        self.keras_train_data = None
        self.val_point_data = None
        self.train_point_data = None
        self.keras_val_data = None

    def loss_func(self, y_true, y_pred):
        loss_ortho = tf.reduce_mean(self.model.get_losses_for(self.model.inputs[0]))
        total_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)(y_true, y_pred)

        # yt = tf.one_hot(y_true, depth=len(utils_.new_labels))
        # y_true_pos = tf.reshape(yt, [-1, ])
        # y_pred_pos = tf.reshape(y_pred, [-1, ])
        # tp = tf.reduce_sum(y_true_pos * y_pred_pos)
        # fn = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
        # fp = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
        # alpha = .5
        # smooth = 1.
        # sc = (tp + smooth) / (tp + alpha * fn + (1 - alpha) * fp + smooth)
        # total_loss = 1. - sc

        return total_loss + loss_ortho

    def load_weights(self, fpath):
        print('---------> Loading weights from', fpath)
        self.model.load_weights(fpath)

    def train(self):
        # x, y = utils_.sample_data(self.val_point_data, random_transform=False)
        # rgbs = utils_.new_colors[y[0]]
        # utils_.write_ply('tmp_.ply', x, rgbs)

        # while True:
        #     x, y = utils_.sample_data(self.train_point_data, random_transform=True)
        #     print('.')

        exps_dec = tf.keras.optimizers.schedules.ExponentialDecay(utils_.BASE_LR, utils_.NUM_TRAIN_STEPS,
                                                                  utils_.LR_EXP_DECAY_POWER)
        lr_sc = tf.keras.callbacks.LearningRateScheduler(exps_dec)
        save_fpath_dir = os.sep.join([utils_.DIR, 'trained_models'])
        utils_.force_makedir(save_fpath_dir)
        save_fpath = os.sep.join([save_fpath_dir,
                                  'aerial-pointnet-weights.{epoch:02d}-{loss:.2f}.hdf5'])
        saver_keras = tf.keras.callbacks.ModelCheckpoint(save_fpath, save_freq='epoch', monitor='loss',
                                                         save_weights_only=True, save_best_only=False)
        logdir = os.sep.join([utils_.DIR, 'logs'])
        utils_.force_makedir(logdir)
        logdir += os.sep + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False, write_images=True,
                                                              histogram_freq=1,
                                                              update_freq=utils_.UPDATE_TENSORBOARD_EVERY_N_STEPS)
        num_epochs = int(np.round(utils_.NUM_TRAIN_STEPS / utils_.BATCHES_PER_EPOCH))
        self.model.fit(self.keras_train_data, callbacks=[lr_sc, saver_keras, tensorboard_callback],
                       steps_per_epoch=utils_.BATCHES_PER_EPOCH, epochs=num_epochs,
                       validation_data=self.keras_val_data, use_multiprocessing=True)

    def load_data(self, train_fps, val_fps):
        # self.keras_train_data = KerasData(train_fps)
        self.train_point_data = PlySet(train_fps)
        self.val_point_data = PlySet(val_fps)
        self.keras_val_data = tf.data.Dataset.from_generator(self.get_val_gen, (tf.float32, tf.int32))
        self.keras_train_data = tf.data.Dataset.from_generator(self.get_train_gen, (tf.float32, tf.int32))

    def infer(self, xyzs_, rescale_scores=False, combine=False):
        if len(xyzs_.shape) == 3:
            ret = np.squeeze(self.model.predict(xyzs_))
        else:
            ret = np.squeeze(self.model.predict(np.expand_dims(xyzs_, 0)))
        if rescale_scores:
            ret_ = ret / ret.max(axis=0)
            if combine:
                return (ret_ + ret) / 2.
            else:
                return ret_
        else:
            return ret

    def get_val_gen(self):
        for _ in range(10):
            yield utils_.sample_data(self.val_point_data, random_transform=False)

    def get_train_gen(self):
        for _ in range(utils_.NUM_TRAIN_STEPS):
            yield utils_.sample_data(self.train_point_data, random_transform=True)

    def make_model(self, mode):
        # (1) input
        if mode == 'infer':
            input = tf.keras.Input(shape=(None, 3))
        else:
            input = tf.keras.Input(shape=(utils_.MAX_TRAIN_IN_POINTS, 3))
        num_points = tf.shape(input)[1]

        # (2)	Input transform. Apply a T-Net module (which outputs a 3x3 transformation matrix)
        # to standardize the input.
        x = TNet(3, add_regularization=True, bn_momentum=self.bn_momentum)(tf.expand_dims(input, -1))
        # (3)	Two point-wise convolution. shared mlp(64,64)
        x = tf.keras.layers.Conv2D(32, activation='relu', kernel_size=[1, 3], strides=1, padding="valid",
                                   use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)(x)
        x = tf.keras.layers.Conv2D(64, activation='relu', kernel_size=[1, 1], strides=1, padding="valid",
                                   use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)(x)

        # (4)	Feature transform. Apply a T-Net module (which outputs a 64x64 transformation matrix)
        # to standardize the feature.
        local_feat = TNet(64, add_regularization=True, bn_momentum=self.bn_momentum)(tf.transpose(x, [0, 1, 3, 2]))
        # (5)	Three point-wise convolution. shared mlp(64,128,1024)

        x = tf.keras.layers.Conv2D(32, activation='relu', kernel_size=[1, 1], strides=1, padding="valid",
                                   use_bias=False)(local_feat)
        x = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)(x)
        x = tf.keras.layers.Conv2D(64, activation='relu', kernel_size=[1, 1], strides=1, padding="valid",
                                   use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)(x)
        x = tf.keras.layers.Conv2D(256, activation='relu', kernel_size=[1, 1], strides=1, padding="valid",
                                   use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)(x)

        # (6)	Max pooling to aggregate information over all points to gain the global descriptor (1024 vector)
        # compare GlobalMaxPool1D with MaxPool1D
        # global_feature = tf.keras.layers.MaxPooling2D(pool_size=(num_points, 1), strides=(2, 2), padding='valid')(x)
        global_feat = tf.keras.layers.GlobalMaxPooling2D()(x)

        global_feat = tf.expand_dims(tf.expand_dims(global_feat, -1), 1)
        global_feat = tf.tile(global_feat, [1, num_points, 1, 1])
        x = tf.concat([local_feat, global_feat], axis=-2)

        x = tf.keras.layers.Conv2D(256, activation='relu', kernel_size=[1, 3], strides=(1, 2), padding="valid",
                                   use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)(x)
        x = tf.keras.layers.Conv2D(128, activation='relu', kernel_size=[1, 3], strides=(1, 2), padding="valid",
                                   use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)(x)
        x = tf.keras.layers.Conv2D(64, activation='relu', kernel_size=[1, 3], strides=(1, 2), padding="valid",
                                   use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)(x)
        x = tf.keras.layers.Conv2D(utils_.n_classes, activation='relu', kernel_size=[1, 39], strides=1, padding="valid",
                                   use_bias=False)(x)[:, :, 0, :]
        output = tf.nn.softmax(x)
        # x = custom_conv(x, 256)
        # x = custom_conv(x, 128)
        # x = custom_conv(x, 64)
        # x = custom_conv(x, 8)
        # output = custom_conv(x, utils_.n_classes, activation='softmax')

        # build the model
        model = tf.keras.models.Model(inputs=input, outputs=output)
        model.summary()
        return model
