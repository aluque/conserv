import tensorflow as tf

def main():
    model = buildmodel()
    model.summary(positions=[.25, .6, .7, 1.])


def buildmodel(filters=32,
               lin_conv_size=9,
               nonlin_conv_size=5):
    input_size = (None, None, 1)
    inputs = tf.keras.Input(shape = input_size, name = "inputs")

    # Expansion in the channels direction (copying)
    x = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False,
                               trainable=False,
                               kernel_initializer=tf.keras.initializers.Ones())(inputs)
        
    n = tf.keras.layers.Conv2D(filters, nonlin_conv_size, padding="same",
                               use_bias=True)(inputs)
    n = tf.keras.layers.Activation('relu')(n)
    n = tf.keras.layers.Conv2D(filters, nonlin_conv_size, padding="same",
                               use_bias=True)(n)
    n = tf.keras.layers.Activation('relu')(n)
    n = tf.keras.layers.Softmax(axis=3)(n)

    m = tf.keras.layers.Multiply()([x, n])

    c = tf.keras.layers.DepthwiseConv2D(lin_conv_size, padding="valid", use_bias=False,
                                        depthwise_constraint=FixSum(1.0))(m)
    
    s = tf.keras.layers.Conv2D(1, 1, padding="same", use_bias=False,
                               kernel_initializer=tf.keras.initializers.Ones(),
                               trainable=False)(c)
    
    model = tf.keras.Model(inputs=inputs, outputs=s)
    return model


class FixSum(tf.keras.constraints.Constraint):
  """ Constrains weight tensors to be nonnegative and add up to `ref_value` 
  in the spatial dimensions. """
  
  def __init__(self, ref_value):
      self.ref_value = ref_value

  def __call__(self, w):
      w1 = w * tf.cast(tf.greater_equal(w, 0.0), tf.keras.backend.floatx())
      s = tf.reduce_sum(w1, axis=[0, 1])
      return self.ref_value * w1 / s

  def get_config(self):
      return {'ref_value': self.ref_value}


if __name__ == '__main__':
    main()
