import tensorflow as tf
import numpy as np

class Autoencoder(object):
    def __init__(self, input_shape,
                 hid_shape, latent_shape,
                 optimizer, dropout = 0.2,
                 batch_size = 512):

        self.keep_prob = 1. - dropout
        self.opt = optimizer
        self.hid_shape = hid_shape
        self.input_shape = input_shape
        self.latent_shape = latent_shape
        self.batch_size = batch_size

        self.build_model()

    def fit_on_batch(self, session, batch):
        feed = {
            self.input: batch,
        }
        ret_values = [self.cost_train, self.to_train]
        cost, _ = session.run(ret_values, feed_dict = feed)
        return cost

    def encode_decode(self, session, batch):
        feed = {
            self.input: batch,
        }
        return session.run(self.decoded_test, feed_dict = feed)

    def evaluate(self, session, batch):
        feed = {
            self.input: batch,
        }
        return session.run(self.cost_test, feed_dict = feed)

    def build_model(self):
        input_shape = self.input_shape

        input = tf.placeholder(tf.float32, shape=[None, input_shape])
        self.input = input

        encoded_train, mu_train, log_sigma_train = \
            self.build_encoder(input, is_train = True)
        encoded_test, mu_test, log_sigma_test = \
            self.build_encoder(input, reuse = True)

        decoded_train = self.build_decoder(encoded_train, is_train = True)
        decoded_test = self.build_decoder(encoded_test, reuse = True)
        self.decoded_test = decoded_test

        self.cost_train = self.build_loss(
            input, decoded_train,
            mu_train, log_sigma_train)

        self.cost_test = self.build_loss(
            input, decoded_test,
            mu_test, log_sigma_test)

        self.to_train = self.opt.minimize(self.cost_train)

    def build_encoder(self, input, reuse = False, is_train = False):
        input_shape, hid_shape, latent_shape, keep_prob = \
            self.input_shape, self.hid_shape, self.latent_shape, self.keep_prob

        with tf.variable_scope('encoder', reuse = reuse):

            w_hid = tf.get_variable(
                "hidden_weights",
                [input_shape, hid_shape],
                initializer=self.get_weight_initializer(input_shape)
            )
            b_hid = tf.get_variable(
                'hidden_bias',
                [hid_shape],
                initializer=self.get_bias_initializer()
            )

            mu_w = tf.get_variable(
                "mu_weights",
                [hid_shape, latent_shape],
                initializer=self.get_weight_initializer(hid_shape)
            )
            mu_b = tf.get_variable(
                'mu_bias',
                [latent_shape],
                initializer=self.get_bias_initializer()
            )

            log_sigma_w = tf.get_variable(
                "log_sigma_weights",
                [hid_shape, latent_shape],
                initializer=self.get_weight_initializer(hid_shape)
            )
            log_sigma_b = tf.get_variable(
                'log_sigma_bias',
                [latent_shape],
                initializer=self.get_bias_initializer()
            )

            if is_train:
                input = tf.nn.dropout(input, keep_prob)

            hid_layer = tf.nn.relu(tf.matmul(input, w_hid) + b_hid)
            layer_mu = tf.matmul(hid_layer, mu_w) + mu_b
            layer_log_sigma = tf.matmul(hid_layer, log_sigma_w) + log_sigma_b

            if is_train:
                epsilon = tf.random_normal(tf.shape(layer_mu))
                z_layer = layer_mu + tf.log(tf.exp(layer_log_sigma) + 1)*epsilon
            else:
                z_layer = layer_mu

        return z_layer, layer_mu, layer_log_sigma

    def build_decoder(self, encoded, reuse = False, is_train = False):
        input_shape, hid_shape, latent_shape, keep_prob = \
            self.input_shape, self.hid_shape, self.latent_shape, self.keep_prob

        with tf.variable_scope('decoder', reuse=reuse):
            w_hid1 = tf.get_variable(
                "hidden1_weights",
                [latent_shape, hid_shape],
                initializer=self.get_weight_initializer(latent_shape)
            )

            b_hid1 = tf.get_variable(
                'hidden1_bias',
                [hid_shape],
                initializer=self.get_bias_initializer()
            )

            w_hid2 = tf.get_variable(
                "hidden2_weights",
                [hid_shape, input_shape],
                initializer=self.get_weight_initializer(hid_shape)
            )

            b_hid2 = tf.get_variable(
                'hidden2_bias',
                [input_shape],
                initializer=self.get_bias_initializer()
            )

            if is_train:
                encoded = tf.nn.dropout(encoded, keep_prob)

            hid_layer = tf.nn.relu(tf.matmul(encoded, w_hid1) + b_hid1)
            output = tf.nn.sigmoid(tf.matmul(hid_layer, w_hid2) + b_hid2)

        return output

    def build_loss(self, x, x_predicted, mu, log_sigma):
        cross_entropy = -tf.reduce_sum(
            x*tf.log(x_predicted + 1e-10) + \
            (1-x)*tf.log(1-x_predicted + 1e-10),
            1)
        latent_loss = -0.5*tf.reduce_sum(
            1 + log_sigma - tf.square(mu) - tf.exp(log_sigma),
            1)

        error = tf.reduce_mean(cross_entropy + latent_loss, name='cost')
        return error

    def get_weight_initializer(self, value):
        return tf.random_normal_initializer(0, 1./np.sqrt(value))

    def get_bias_initializer(self):
        return tf.constant_initializer(1e-4)