from tensorflow.python.ops import embedding_ops
from tensorflow.keras.layers import Layer
import tensorflow as tf


class Squeeze(Layer):
    def __init__(self, axis=None):
        self.axis = axis
        super(Squeeze, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        return tf.squeeze(inputs, self.axis)

    def get_config(self):
        config = {
            'axis': self.axis
        }
        base_config = super(Squeeze, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class ExpandDims(Layer):
    def __init__(self, axis):
        self.axis = axis
        super(ExpandDims, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        return tf.expand_dims(inputs, self.axis)

    def get_config(self):
        config = {
            'axis': self.axis
        }
        base_config = super(ExpandDims, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Embedding2D(Layer):
    def __init__(self,
                 input_dim,
                 output_width,
                 output_height,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 **kwargs):
        dtype = kwargs.pop('dtype', tf.keras.backend.floatx())
        super(Embedding2D, self).__init__(dtype=dtype, **kwargs)

        self.input_dim = input_dim
        self.output_width = output_width
        self.output_height = output_height
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = tf.keras.regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = tf.keras.constraints.get(embeddings_constraint)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_width, self.output_height),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            name='embeddings2d', )

    def call(self, inputs, **kwargs):
        return embedding_ops.embedding_lookup(self.embeddings, inputs)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_width': self.output_width,
            'output_height': self.output_height,
            'embeddings_initializer': tf.keras.initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': tf.keras.regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint': tf.keras.constraints.serialize(self.embeddings_constraint),
        }
        base_config = super(Embedding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
