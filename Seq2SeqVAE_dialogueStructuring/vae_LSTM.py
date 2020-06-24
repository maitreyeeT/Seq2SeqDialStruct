import keras
from Seq2SeqVAE_dialogueStructuring.variational_layer import VariationalLayer
l = keras.layers
m = keras.models
K = keras.backend
opt = keras.optimizers
losses = keras.losses
cbk = keras.callbacks
reg = keras.regularizers


class LSTMVAutoencoder:

  def __init__(self, lr, window_size, sequence_size, dropout=0.):
    self.lr = lr
    self.window_size = window_size
    self.sequences_length = sequence_size
   # self.input_convolution_size = 3
   # self.conv_2d_filters = []  # Num filter of the convolution layers
   # self.conv_2d_kernels = []  # Kernel sizes of the convolution layers

    self.dropout = dropout
    self.regul = 1e-4
    self.latent_dim = 256
    self.build()

  def get_attention(self,name, output_size, dropout=0., axis=-2):
    model_att = m.Sequential(name='attention_' + str(name))  # (?, n, k, x)
    model_att.add(l.Dense(1, activation=K.exp))  # (?, n, k, 1)
    model_att.add(l.Lambda(lambda x: x / (K.sum(x, axis=axis, keepdims=True) + \
                                          K.epsilon()),
                           name=name + 'attention'))  # (?, n, k, 1)

    m_att = m.Sequential()
    m_att.add(l.Lambda(lambda x: K.sum(x * model_att(x), axis=axis)))
    m_att.add(l.Dense(output_size, activation='linear'))
    return m_att

  def make_networks(self):
    attention = self.get_attention('xylo',10)
    #from w2vec_utils import train_word2vec, data_transform
    #data_embed = data_transform()
    #w2vmodel, embed_wts = train_word2vec(data_embed)
    encoder = m.Sequential(name='encoder')
    #encoder.add(l.Embedding(output_dim=300, weights=[embed_wts],input_dim=embed_wts.shape[0],
    #                        input_shape=(self.sequences_length,2)))
    #encoder.add(l.Lambda(lambda x: K.mean(x, axis=-2)))
    encoder.add(l.LSTM(10, return_sequences='True'))
    encoder.add(l.LSTM(10, return_sequences='True'))

    encoder.add(attention)
    #encoder.add(l.Dense(5, activity_regularizer=reg.l2(1e-3)))

    self.encoder = encoder

    decoder = m.Sequential(name='decoder')
    decoder.add(l.RepeatVector(self.sequences_length))
    decoder.add(l.LSTM(10, return_sequences=True))
    decoder.add(l.LSTM(self.window_size, activation='sigmoid'
                       , return_sequences=True))

    self.decoder = decoder

  def build(self):

    seq_input = l.Input(shape=(self.sequences_length, self.window_size),
                        dtype='float32')  # (?, n, k)

    self.make_networks()

    latent_in = self.encoder(seq_input)
    latent_in = VariationalLayer(10, kl_gain=1e-3)(latent_in)

    decoder_out = self.decoder(latent_in)

    model = m.Model(inputs=[seq_input], outputs=[decoder_out])

    model_latent = m.Model(inputs=[seq_input], outputs=[latent_in])

    model.compile(opt.Adam(self.lr), loss=[
      lambda x, y: losses.binary_crossentropy(x, y),
    ])

    model_latent.compile(opt.RMSprop(1e-3), loss='binary_crossentropy')
    self.model = model
    self.model_latent = model_latent

  def fit(self, X, epochs, callbacks=None, verbose=2):

    self.model.fit(x=[X],
                   y=[X],
                   batch_size=30,
                   epochs=epochs,
                   validation_split=.2,
                   callbacks=callbacks,
                   verbose=verbose
                   )


