import typing

import tensorflow as tf

K = tf.keras.backend

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

def apply_mlp(input: tf.Tensor,
              hidden_size: int,
              output_size: int,
              hidden_activation: str = 'tanh',
              output_activation: str = None,
              use_time_distributed: bool = False) -> tf.Tensor:
    '''Applies simple feed-forward network to an input.

    Args:
        input: MLP input
        hidden_size: Hidden state size
        output_size: Output size of MLP
        hidden_activation: Activation function to apply on input
        output_activation: Activation function to apply on output
        use_time_distributed: Whether to apply across time
    '''
    hidden_dense = tf.keras.layers.Dense(hidden_size, activation=hidden_activation)
    output_dense = tf.keras.layers.Dense(output_size, activation=output_activation)
    if use_time_distributed:
        hidden = tf.keras.layers.TimeDistributed(hidden_dense)(input)
        return tf.keras.layers.TimeDistributed(output_dense)(hidden)
    else:
        hidden = hidden_dense(input)
        return output_dense(hidden)

def apply_gating_layer(input: tf.Tensor,
                       hidden_layer_size: int,
                       dropout_rate: float = None,
                       use_time_distributed: bool = True,
                       activation: str = None) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    '''Applies a Gated Linear Unit (GLU) to an input.

    Args:
        input: Input to gating layer
        hidden_layer_size: Dimension of GLU
        dropout_rate: Dropout rate to apply if any
        use_time_distributed: Whether to apply across time
        activation: Activation function to apply to the linear feature if necessary

    Return:
        Tuple of tensors for : (GLU output, gated_layer output)
    '''
    if dropout_rate is not None:
        input = tf.keras.layers.Dropout(dropout_rate)(input)

    activation_dense = tf.keras.layers.Dense(hidden_layer_size, activation=activation)
    gating_dense = tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid')

    if use_time_distributed:
        activated_output = tf.keras.layers.TimeDistributed(activation_dense)(input)
        gated_output = tf.keras.layers.TimeDistributed(gating_dense)(input)
    else:
        activated_output = activation_dense(input)
        gated_output = tf.keras.layers.TimeDistributed(input)

    return tf.keras.layers.Multiply()([activated_output, gated_output]), gated_output

def apply_add_and_norm(inputs: typing.List[tf.Tensor]) -> tf.Tensor:
    '''Applies skip connection followed by layer normalization.

    Args:
        inputs: List of input to sum for skip connection
    
    Returns:
        Tensor output from layer.
    '''
    x = tf.keras.layers.Add()(inputs)
    return tf.keras.layers.LayerNormalization()(x)

def apply_gated_residual_network(input: tf.Tensor,
                                 hidden_layer_size: int,
                                 output_size: int = None,
                                 dropout_rate: float = None,
                                 use_time_distributed : bool = True,
                                 additional_context: tf.Tensor = None
                                 ) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    '''Applies the gated residual network (GRN) as defined in paper.

    Args:
        input: Network input
        hidden_layer_size: Internal state size
        output_size: Size of output layer
        dropout_rat: Dropout rat if dropout is applied
        use_time_distributed: Whether to apply network across time dimension
        additional_context: Additional context vector to use if relevant
    
    Returns:
        Tuple of tensors for : (GRN output, GLU output)
    '''

    # Setup skip connection
    if output_size is None:
        output_size = hidden_layer_size
        skip = input
    else:
        linear = tf.keras.layers.Dense(output_size)
        if use_time_distributed:
            linear = tf.keras.layers.TimeDistributed(linear)
        skip = linear(input)
    

    # Apply feedforward network
    hidden = linear_layer(hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed
    )(input)
    if additional_context is not None:
        hidden = hidden + linear_layer(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed
        )(additional_context)

    hidden = tf.keras.layers.Activation('elu')(hidden)
    hidden = linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed
    )(hidden)

    glu_output, gated_output = apply_gating_layer(
        hidden,
        output_size,
        dropout_rate=dropout_rate,
        use_time_distributed=use_time_distributed,
        activation=None)

    return apply_add_and_norm([skip, glu_output]), gated_output

def get_decoder_mask(self_attn_input: tf.Tensor) -> tf.Tensor:
    '''Returns causal mask to apply self-attention layer.

    Args:
        self_attn_input: Input to self attention layer to determine mask shape
    '''
    len_s = tf.shape(self_attn_input)[1]
    bs = tf.shape(self_attn_input)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)

    return mask

class ScaledDotProductAttention(object):
    '''Defines scaled dot product attention layer.

    Attributes:
        dropout: Dropout rate to use
        activation: Normalization function for scaled dot product attention
        (e.g. softmax by default)
    '''
    def __init__(self, attn_dropout=0.0):
        self.dropout = tf.keras.layers.Dropout(attn_dropout)
        self.activation = tf.keras.layers.Activation('softmax')

    def __call__(self,
                 q: tf.Tensor,
                 k: tf.Tensor,
                 v: tf.Tensor, 
                 mask: tf.Tensor
                ) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        '''Applies scaled dot product attention.

        Args:
            q: Queries
            k: Keys
            v: Values
            mask: Masking if required -- sets softmax to very large value

        Returns:
            Tuple of (layer outputs, attention weieghts)
        '''
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
        attn = tf.keras.layers.Lambda(
            lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper
        )([q, k])

        if mask is not None:
            mmask = tf.keras.layers.Lambda(
                lambda x: (-1e+9) * (1. - K.cast(x, 'float32'))
            )(mask)
            attn = tf.keras.layers.Add()([attn, mmask])
        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = tf.keras.layers.Lambda(
            lambda x: K.batch_dot(x[0], x[1])
        )([attn, v])

        return output, attn



if __name__ == '__main__':
    input = tf.keras.layers.Input((50, 3))
    x = apply_gated_residual_network(input, 30, 2)
    print(x)