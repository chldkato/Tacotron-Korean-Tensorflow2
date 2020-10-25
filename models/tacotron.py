from tensorflow.keras.layers import Embedding, GRU
from models.modules import *
from util.hparams import *


class Encoder(tf.keras.Model):
    def __init__(self, K, conv_dim):
        super(Encoder, self).__init__()
        self.embedding = Embedding(symbol_length, embedding_dim)
        self.pre_net = pre_net()
        self.cbhg = CBHG(K, conv_dim)

    def call(self, enc_input, sequence_length, is_training):
        x = self.embedding(enc_input)
        x = self.pre_net(x, is_training=True)
        x = self.cbhg(x, sequence_length, is_training=is_training)
        return x


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.pre_net = pre_net()
        self.attention_rnn = GRU(decoder_dim, return_sequences=True)
        self.proj1 = Dense(decoder_dim)
        self.dec_rnn1 = GRU(decoder_dim, return_sequences=True)
        self.dec_rnn2 = GRU(decoder_dim, return_sequences=True)
        self.proj2 = Dense(mel_dim * reduction)

    def call(self, batch, dec_input, enc_output):
        x = self.pre_net(dec_input, is_training=True)
        x = self.attention_rnn(x)
        context, alignment = attention(x, enc_output)

        dec_rnn_input = self.proj1(context)
        dec_rnn_input += self.dec_rnn1(dec_rnn_input)
        dec_rnn_input += self.dec_rnn2(dec_rnn_input)

        dec_out = self.proj2(dec_rnn_input)
        mel_out = tf.reshape(dec_out, [batch, -1, mel_dim])

        return mel_out, alignment


class Tacotron(tf.keras.Model):
    def __init__(self, K, conv_dim):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(K, conv_dim)
        self.decoder = Decoder()

    def call(self, enc_input, sequence_length, dec_input, is_training):
        batch = dec_input.shape[0]
        x = self.encoder(enc_input, sequence_length, is_training)
        x = self.decoder(batch, dec_input, x)
        return x


class post_CBHG(tf.keras.Model):
    def __init__(self, K, conv_dim):
        super(post_CBHG, self).__init__()
        self.cbhg = CBHG(K, conv_dim)
        self.dense = Dense(n_fft // 2 + 1)

    def call(self, mel_input, is_training):
        x = self.cbhg(mel_input, None, is_training=is_training)
        x = self.dense(x)
        return x