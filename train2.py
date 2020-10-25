import os, glob, random, traceback
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MAE
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.tacotron import post_CBHG
from util.hparams import *


data_dir = './data'
mel_list = glob.glob(os.path.join(data_dir + '/mel', '*.npy'))
spec_list = glob.glob(os.path.join(data_dir + '/spec', '*.npy'))

fn = os.path.join(data_dir + '/mel_len.py')
if not os.path.isfile(fn):
    mel_len_list = []
    for i in range(len(mel_list)):
        mel_length = np.load(mel_list[i]).shape[0]
        mel_len_list.append([mel_length, i])
    mel_len = sorted(mel_len_list)
    np.save(os.path.join(data_dir + '/mel_len.npy'), np.array(mel_len))

text_len = np.load(os.path.join(data_dir + '/text_len.npy'))
mel_len = np.load(os.path.join(data_dir + '/mel_len.npy'))


def DataGenerator():
    while True:
        idx_list = np.random.choice(len(mel_list), batch_size * batch_size, replace=False)
        idx_list = sorted(idx_list)
        idx_list = [idx_list[i : i + batch_size] for i in range(0, len(idx_list), batch_size)]
        random.shuffle(idx_list)

        for idx in idx_list:
            random.shuffle(idx)

            mel = [np.load(mel_list[mel_len[i][1]]) for i in idx]
            spec = [np.load(spec_list[mel_len[i][1]]) for i in idx]

            mel = pad_sequences(mel, padding='post', dtype='float32')
            spec = pad_sequences(spec, padding='post', dtype='float32')

            yield (mel, spec)


@tf.function(experimental_relax_shapes=True)
def train_step(mel_input, spec_target):
    with tf.GradientTape() as tape:
        pred = model(mel_input, is_training=True)
        loss = tf.reduce_mean(MAE(spec_target, pred))
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss, pred[0]


dataset = tf.data.Dataset.from_generator(generator=DataGenerator,
                                         output_types=(tf.float32, tf.float32),
                                         output_shapes=(tf.TensorShape([batch_size, None, mel_dim]),
                                                        tf.TensorShape([batch_size, None, n_fft // 2 + 1])))\
    .prefetch(tf.data.experimental.AUTOTUNE)

model = post_CBHG(K=8, conv_dim=[256, mel_dim])
optimizer = Adam()
step = tf.Variable(0)

checkpoint_dir = './checkpoint/2'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, step=step)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print('Restore checkpoint from {}'.format(manager.latest_checkpoint))

try:
    for mel, spec in dataset:
        loss, pred = train_step(mel, spec)
        checkpoint.step.assign_add(1)
        print("Step: {}, Loss: {:.5f}".format(int(checkpoint.step), loss))

        if int(checkpoint.step) % checkpoint_step == 0:
            checkpoint.save(file_prefix=os.path.join(checkpoint_dir, 'step-{}'.format(int(checkpoint.step))))

except Exception:
    traceback.print_exc()