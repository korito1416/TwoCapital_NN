import tensorflow as tf
import numpy as np


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config):
        super(FeedForwardSubNet, self).__init__(name = config["nn_name"] + ".init_layer")
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5),
                name = config["nn_name"] + ".bn." + str(_)
            )
            for _ in range(len(config["num_hiddens"]) + 1)]
        
        if config['activation'] is not None and "relu" in config['activation']:
            initializer = tf.keras.initializers.HeNormal(seed=0)
        else:
            initializer = tf.keras.initializers.GlorotUniform(seed=0)

        self.dense_layers = [tf.keras.layers.Dense(config["num_hiddens"][i],
                                                   use_bias=config['use_bias'],
                                                   activation=config['activation'],
                                                   kernel_initializer = initializer,
                                                   name = config["nn_name"] + ".dense." + str(i))
                             for i in range(len(config["num_hiddens"]))]
        # final output should be gradient of size dim
        try:
            if config['final_activation'] is None:
                initializer = tf.keras.initializers.GlorotUniform(seed=0)
            elif "relu" in config['final_activation']:
                initializer = tf.keras.initializers.HeNormal(seed=0)
            else:
                initializer = tf.keras.initializers.GlorotUniform(seed=0)
        except:
            initializer = tf.keras.initializers.GlorotUniform(seed=0)

        self.dense_layers.append(tf.keras.layers.Dense(config["dim"], 
        kernel_initializer = initializer, 
        activation=config['final_activation'], use_bias = True, name = config["nn_name"] + ".output" ))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x, training)
        x_inputs = []
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x_inputs.append(x)
        x = tf.keras.layers.Add()(x_inputs)
        x = self.dense_layers[-1](x)
        return x


def setup_optimizers(params):
    """Create optimizers and store them in params['optimizers'].

    Mutates the passed `params` dict in-place. Supported schedule types:
      - 'None' : constant learning rates (Adam)
      - 'piecewiseconstant'
      - 'sgd+piecewiseconstant'
      - 'sgd'

    Expects params["learning_rates"] to be an iterable of floats and
    params["num_iterations"] to be an int.
    """
    learning_rates = params.get("learning_rates", [10e-4,10e-4,10e-4,10e-4])
    num_iterations = int(params.get("num_iterations", 100000))
    lr_type = params.get("learning_rate_schedule_type", "piecewiseconstant")

    if lr_type == "None":
        params["optimizers"] = [tf.keras.optimizers.Adam(learning_rate=float(lr)) for lr in learning_rates]
        return

    if lr_type == "piecewiseconstant":
        boundaries = [int(round(x)) for x in np.linspace(0, num_iterations, 8)][1:-1]
        values_list = [[float(lr) / np.power(4, x) for x in range(len(boundaries) + 1)] for lr in learning_rates]
        lr_schedulers = [tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values) for values in values_list]
        params["optimizers"] = [tf.keras.optimizers.Adam(learning_rate=s) for s in lr_schedulers]
        return

    if lr_type == "sgd+piecewiseconstant":
        boundaries = [int(round(x)) for x in np.linspace(0, num_iterations, 5)][1:-1]
        values_list = [[float(lr) / np.power(2, x) for x in range(len(boundaries) + 1)] for lr in learning_rates]
        lr_schedulers = [tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values) for values in values_list]
        params["optimizers"] = [tf.keras.optimizers.legacy.SGD(learning_rate=s) for s in lr_schedulers]
        return

    if lr_type == "sgd":
        params["optimizers"] = [tf.keras.optimizers.legacy.SGD(learning_rate=float(lr)) for lr in learning_rates]
        return

    # fallback
    params["optimizers"] = [tf.keras.optimizers.Adam(learning_rate=float(learning_rates[0]))]
