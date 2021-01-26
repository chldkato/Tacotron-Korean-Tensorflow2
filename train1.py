import os, glob, random, traceback
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MAE
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.tacotron import Tacotron
from util.hparams import *
from util.plot_alignment import plot_alignment
from util.text import sequence_to_text


data_dir = './data'
text_list = glob.glob(os.path.join(data_dir + '/text', '*.npy'))
mel_list = glob.glob(os.path.join(data_dir + '/mel', '*.npy'))
dec_list = glob.glob(os.path.join(data_dir + '/dec', '*.npy'))

fn = os.path.join(data_dir + '/mel_len.npy')
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

            text = [np.load(text_list[mel_len[i][1]]) for i in idx]
            dec = [np.load(dec_list[mel_len[i][1]]) for i in idx]
            mel = [np.load(mel_list[mel_len[i][1]]) for i in idx]
            text_length = [text_len[mel_len[i][1]] for i in idx]

            text = pad_sequences(text, padding='post')
            dec = pad_sequences(dec, padding='post', dtype='float32')
            mel = pad_sequences(mel, padding='post', dtype='float32')

            yield (text, dec, mel, text_length)


@tf.function(experimental_relax_shapes=True)
def train_step(enc_input, dec_input, dec_target, text_length):
    with tf.GradientTape() as tape:
        pred, alignment = model(enc_input, text_length, dec_input, is_training=True)
        loss = tf.reduce_mean(MAE(dec_target, pred))
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss, pred[0], alignment[0]


dataset = tf.data.Dataset.from_generator(generator=DataGenerator,
                                         output_types=(tf.float32, tf.float32, tf.float32, tf.int32),
                                         output_shapes=(tf.TensorShape([batch_size, None]),
                                                        tf.TensorShape([batch_size, None, mel_dim]),
                                                        tf.TensorShape([batch_size, None, mel_dim]),
                                                        tf.TensorShape([batch_size])))\
    .prefetch(tf.data.experimental.AUTOTUNE)

model = Tacotron(K=16, conv_dim=[128, 128])
optimizer = Adam()
step = tf.Variable(0)

checkpoint_dir = './checkpoint/1'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, step=step)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print('Restore checkpoint from {}'.format(manager.latest_checkpoint))

try:
    for text, dec, mel, text_length in dataset:
        loss, pred, alignment = train_step(text, dec, mel, text_length)
        checkpoint.step.assign_add(1)
        print("Step: {}, Loss: {:.5f}".format(int(checkpoint.step), loss))

        if int(checkpoint.step) % checkpoint_step == 0:
            checkpoint.save(file_prefix=os.path.join(checkpoint_dir, 'step-{}'.format(int(checkpoint.step))))

            input_seq = sequence_to_text(text[0].numpy())
            input_seq = input_seq[:text_length[0].numpy()]
            alignment_dir = os.path.join(checkpoint_dir, 'step-{}-align.png'.format(int(checkpoint.step)))
            plot_alignment(alignment, alignment_dir, input_seq)

except Exception:
    traceback.print_exc()
