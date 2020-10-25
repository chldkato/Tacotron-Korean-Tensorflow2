import librosa
import numpy as np
import tensorflow as tf
from copy import deepcopy
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv1D, MaxPooling1D, GRUCell
from util.hparams import *


class pre_net(tf.keras.Model):
    def __init__(self):
        super(pre_net, self).__init__()
        self.dense1 = Dense(256)
        self.dense2 = Dense(128)

    def call(self, input_data, is_training):
        x = self.dense1(input_data)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x, training=is_training)
        x = self.dense2(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x, training=is_training)
        return x


class CBHG(tf.keras.Model):
    def __init__(self, K, conv_dim):
        super(CBHG, self).__init__()
        self.K = K
        self.conv_bank = []
        for k in range(1, self.K+1):
            x = Conv1D(128, kernel_size=k, padding='same')
            self.conv_bank.append(x)

        self.bn = BatchNormalization()
        self.conv1 = Conv1D(conv_dim[0], kernel_size=3, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv1D(conv_dim[1], kernel_size=3, padding='same')
        self.bn2 = BatchNormalization()

        self.proj = Dense(128)
        self.dense1 = Dense(128)
        self.dense2 = Dense(128, bias_initializer=tf.constant_initializer(-1.0))

        self.gru_fw = GRUCell(128)
        self.gru_bw = GRUCell(128)

    def call(self, input_data, sequence_length, is_training):
        x = tf.concat([
                Activation('relu')(self.bn(
                    self.conv_bank[i](input_data)), training=is_training) for i in range(self.K)], axis=-1)
        x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
        x = self.conv1(x)
        x = self.bn1(x, training=is_training)
        x = Activation('relu')(x)
        x = self.conv2(x)
        x = self.bn2(x, training=is_training)
        highway_input = input_data + x

        if self.K == 8:
            highway_input = self.proj(highway_input)

        for _ in range(4):
            H = self.dense1(highway_input)
            H = Activation('relu')(H)
            T = self.dense2(highway_input)
            T = Activation('sigmoid')(T)
            highway_input = H * T + highway_input * (1.0 - T)

        x, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
            self.gru_fw,
            self.gru_bw,
            highway_input,
            sequence_length=sequence_length,
            dtype=tf.float32)
        x = tf.concat(x, axis=2)

        return x


def attention(query, value):
    alignment = tf.nn.softmax(tf.matmul(query, value, transpose_b=True))
    context = tf.matmul(alignment, value)
    context = tf.concat([context, query], axis=-1)
    alignment = tf.transpose(alignment, [0, 2, 1])
    return context, alignment


def griffin_lim(spectrogram):
    spec = deepcopy(spectrogram)
    for i in range(50):
        est_wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length)
        est_stft = librosa.stft(est_wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        phase = est_stft / np.maximum(1e-8, np.abs(est_stft))
        spec = spectrogram * phase
    wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length)
    return np.real(wav)
