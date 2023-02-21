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

    c = tf.keras.layers.DepthwiseConv2D(lin_conv_size, padding="same", use_bias=False,
                                        kernel_constraint=CenterAround(0.0))(m)
    
    s = tf.keras.layers.Conv2D(1, 1, padding="same", use_bias=False,
                               kernel_initializer=tf.keras.initializers.Constant(value=1./filters),
                               trainable=False)(c)
    
    model = tf.keras.Model(inputs=inputs, outputs=s)
    return model


# Lifted from the keras documentation.  Using ref_value=0 it can be used to
# impose charge conservation in convolution.
class CenterAround(tf.keras.constraints.Constraint):
  """Constrains weight tensors to be centered around `ref_value`."""
  
  def __init__(self, ref_value):
    self.ref_value = ref_value

  def __call__(self, w):
    mean = tf.reduce_mean(w)
    return w - mean + self.ref_value

  def get_config(self):
    return {'ref_value': self.ref_value}



if __name__ == '__main__':
    main()
