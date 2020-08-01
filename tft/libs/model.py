import typing

import tensorflow as tf


def linear_layer(size: int,
                 activation: str = None,
                 use_time_distributed:bool = False,
                 use_bias:bool = True) -> tf.keras.Model:
    '''Returns simple Keras linear layer.

    Args:
        size: Output size
        activation: Activation function to apply if required
        use_time_distributed: Whether to apply layer across time
        use_bias: Whether bias should be included in layer
    '''
    linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
    if use_time_distributed:
        linear = tf.keras.layers.TimeDistributed(linear)
    return linear

def apply_mlp(inputs: tf.Tensor,
              hidden_size: int,
              output_size: int,
              hidden_activation: str = 'tanh',
              output_activation: str = None,
              use_time_distributed: bool = False) -> tf.Tensor:
    '''Applies simple feed-forward network to an input.

    Args:
        inputs: MLP inputs
        hidden_size: Hidden state size
        output_size: Output size of MLP
        hidden_activation: Activation function to apply on input
        output_activation: Activation function to apply on output
        use_time_distributed: Whether to apply across time
    '''
    hidden_dense = tf.keras.layers.Dense(hidden_size, activation=hidden_activation)
    output_dense = tf.keras.layers.Dense(output_size, activation=output_activation)
    if use_time_distributed:
        hidden = tf.keras.layers.TimeDistributed(hidden_dense)(inputs)
        return tf.keras.layers.TimeDistributed(output_dense)(hidden)
    else:
        hidden = hidden_dense(inputs)
        return output_dense(hidden)

def apply_gating_layer(x: tf.Tensor,
                       hidden_layer_size: int,
                       dropout_rate: float = None,
                       use_time_distributed: bool = True,
                       activation: str = None) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    '''Applies a Gated Linear Unit (GLU) to an input.

    Args:
        x: Input to gating layer
        hidden_layer_size: Dimension of GLU
        dropout_rate: Dropout rate to apply if any
        use_time_distributed: Whether to apply across time
        activation: Activation function to apply to the linear feature if necessary

    Return:
        Tuple of tensors for : (GLU output, gated_layer output)
    '''
    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    activation_dense = tf.keras.layers.Dense(hidden_layer_size, activation=activation)
    gating_dense = tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid')

    if use_time_distributed:
        activated_output = tf.keras.layers.TimeDistributed(activation_dense)(x)
        gated_output = tf.keras.layers.TimeDistributed(gating_dense)(x)
    else:
        activated_output = activation_dense(x)
        gated_output = tf.keras.layers.TimeDistributed(x)

    return tf.keras.layers.Multiply()([activated_output, gated_output]), gated_output

def gated_residual_network():
    pass

