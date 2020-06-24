import keras

l = keras.layers
m = keras.models
K = keras.backend
reg = keras.regularizers
initializers = keras.initializers
activations = keras.activations


class VariationalLayer(l.Layer):


    def __init__(self, output_dim, kl_gain=.01, return_mu_sigma=False, **kwargs):
        super(VariationalLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.kl_gain  = kl_gain
        self.bias_mu    = None
        self.bias_sigma = None
        self.W_mu       = None
        self.W_sigma    = None
        self.return_mu_sigma = return_mu_sigma
        self.instances  = list()



    def build(self, input_shape):

        input_dim = int(input_shape[-1])

        init = initializers.glorot_normal()
        dtype = K.floatx()

        self.W_mu       = self.add_weight(name='{}/w_mu'.format(self.name),
                                          shape=(input_dim, self.output_dim),
                                          initializer=init,
                                          dtype=dtype
                                          )
        self.W_sigma    = self.add_weight(name='{}/w_delta'.format(self.name),
                                          shape=(input_dim, self.output_dim),
                                          initializer=init,
                                          dtype=dtype
                                          )
        self.bias_mu    = self.add_weight(name='{}/bias_mu'.format(self.name),
                                          shape=(self.output_dim,),
                                          initializer=init,
                                          dtype=dtype
                                          )
        self.bias_sigma = self.add_weight(name='{}/bias_delta'.format(self.name),
                                          shape=(self.output_dim,),
                                          initializer=init,
                                          dtype=dtype
                                          )

        super(VariationalLayer, self).build(input_shape)



    def call(self, input_tensor, training=None):

        mu_z    = activations.linear(K.bias_add(K.dot(input_tensor, self.W_mu), self.bias_mu))
        sigma_z = activations.softplus(K.bias_add(K.dot(input_tensor, self.W_sigma), self.bias_sigma))
        eps     = K.random_normal(shape=K.shape(mu_z), mean=0., stddev=1.)

        z = mu_z + eps * K.square(sigma_z)
        out = K.in_train_phase(z, mu_z, training)


        kl_loss = VariationalLayer.kl_divergence(mu_z, sigma_z, 0., 1.)

        self.instances.append((mu_z, sigma_z))
        if self.kl_gain  > 0.:
            self.add_loss(self.kl_gain * kl_loss)

        if self.return_mu_sigma:
            return out, mu_z, sigma_z

        return out


    def add_kl_loss(self, kl_gain, ext_mu_z, ext_sigma_z):
        for mu_z, sigma_z in self.instances:
            kl_loss = VariationalLayer.kl_divergence(mu_z, sigma_z, ext_mu_z, ext_sigma_z)
            self.add_loss(kl_gain * kl_loss)

    def add_kl_loss_between_instances(self, kl_gain, inst_1, inst_2):
        mu_z, sigma_z = self.instances[inst_1]
        ext_mu_z, ext_sigma_z = self.instances[inst_2]
        kl_loss = VariationalLayer.kl_divergence(mu_z, sigma_z, ext_mu_z, ext_sigma_z)
        self.add_loss(kl_gain * kl_loss)


    def compute_output_shape(self, input_shape):
        out_shape = tuple(input_shape[:-1] +(self.output_dim,))
        if self.return_mu_sigma:
            return [out_shape,out_shape, out_shape]
        return out_shape



    def get_config(self):
        base_config = super(VariationalLayer, self).get_config()
        base_config.update({'kl_regul': self.kl_gain})
        return base_config

    @staticmethod
    def kl_divergence(mu_1, sigma_1, mu_2, sigma_2):

        sigma_1_sq = K.square(sigma_1)
        sigma_2_sq = K.square(sigma_2)

        p1 = K.log(sigma_2 / sigma_1)
        p2 = .5 * (sigma_1_sq + K.square(mu_1 - mu_2)) / sigma_2_sq
        p3 = -.5

        return K.mean(p1 + p2 + p3)


class GainCallback(keras.callbacks.Callback):

    def __init__(self, nepochs, range=(-7.5, 7.5), sigmoid_pars=(1., 1., 0.)):

        super(GainCallback, self).__init__()

        self.epoch = K.variable(value=0., dtype=K.floatx())
        self.nepochs = K.variable(value=nepochs, dtype=K.floatx())
        self.range = range
        self.sigmoid_pars = sigmoid_pars

        x = range[0] + self.epoch / self.nepochs * (range[1] - range[0])

        c1 = sigmoid_pars[0]
        c2 = sigmoid_pars[1]
        c3 = sigmoid_pars[2]

        self.gain = c1 / (1. + K.exp(-c2 * x)) - c3

        #self.gain = c1 * K.min([self.epoch / self.nepochs, 1.]) + K.epsilon()

    def on_epoch_begin(self, epoch, logs=None):
        K.set_value(self.epoch, epoch)


    def on_train_begin(self, logs=None):
        K.set_value(self.epoch, 0.)


    def __call__(self, *args, **kwargs):
        assert len(args) == 1
        return self.gain * args[1]
