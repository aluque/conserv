import tensorflow as tf
from keras import backend as K


def main():
    model = buildmodel()
    model.summary(positions=[.25, .6, .7, 1.])
    model.compile(loss='mse', optimizer='Adam')


def buildmodel(filters=32, l=3, m=3, concat_input=False):
    lin_conv_size = 2 * l + 1
    nonlin_conv_size = 2 * m + 1
    
    input_size = (None, None, 1)
    inputs = tf.keras.Input(shape = input_size, name = "inputs")

    initializer = GaussianInitializer(l, l / 3)

    # Obtaining the weights b and c
    n = Padding2D(name="pad_conv1", padding=[m, m])(inputs)
    n = tf.keras.layers.Conv2D(filters / 4, nonlin_conv_size,
                               name="conv1",
                               padding="valid",
                               use_bias=True)(n)
    n = tf.keras.layers.LeakyReLU(name="activ1", alpha=0.1)(n)
    
    n = tf.keras.layers.Concatenate(name="concat2", axis=3)([n, inputs])
    n = Padding2D(name="pad_conv2", padding=[m, m])(n)
    n = tf.keras.layers.Conv2D(filters / 2, nonlin_conv_size,
                               name="conv2",
                               padding="valid",
                               use_bias=True)(n)
    n = tf.keras.layers.LeakyReLU(name="activ2", alpha=0.1)(n)

    n = tf.keras.layers.Concatenate(name="concat3", axis=3)([n, inputs])
    n = Padding2D(name="pad_conv3", padding=[m, m])(n)
    n = tf.keras.layers.Conv2D(filters, nonlin_conv_size,
                               name="conv3",
                               padding="valid",
                               use_bias=True)(n)
    n = tf.keras.layers.LeakyReLU(name="activ3", alpha=0.1)(n)

    n = tf.keras.layers.LayerNormalization(name="lnorm", axis=3)(n)
    #n = tf.keras.layers.GaussianNoise(0.1, name="noise")(n)
    
    n = tf.keras.layers.Softmax(name="softmax", axis=3)(n)
    # n = tf.keras.layers.SpatialDropout2D(0.1, name="dropout")(n)

    # Expansion in the channels direction (copying)
    x = tf.keras.layers.Conv2D(filters, 1, name="demux",
                               padding="same", use_bias=False,
                               trainable=False,
                               kernel_initializer=tf.keras.initializers.Ones())(inputs)

    p = tf.keras.layers.Multiply(name="partition")([x, n])

    p = tf.keras.layers.ZeroPadding2D(name="pad_K", padding=[l, l])(p)

    homogen = tf.keras.initializers.Constant(1/lin_conv_size**2)
    c = tf.keras.layers.DepthwiseConv2D(lin_conv_size,
                                        name="K",
                                        padding="same",
                                        use_bias=False,
                                        kernel_initializer=initializer,
                                        depthwise_constraint=FixSumBias(1.0))(p)

    c = Fold2D(l, name="fold_K")(c)
    c = tf.keras.layers.Cropping2D(l, name="crop")(c)
    
    y = tf.keras.layers.Conv2D(1, 1,
                               name="mux",
                               padding="same", use_bias=False,
                               kernel_initializer=tf.keras.initializers.Ones(),
                               trainable=False)(c)
    
    if concat_input:
        y = tf.keras.layers.Concatenate()([y, inputs])

    model = tf.keras.Model(inputs=inputs, outputs=y)
    return model


def gaussian_kernel(l, std):
    """Generates a 2D Gaussian kernel."""
    norm = 1 / (std * np.sqrt(2 * np.pi))
    vals = tf.exp(-tf.square(tf.range(start=-l, limit=l + 1, dtype=tf.float32) / std) / 2) / norm
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


class GaussianInitializer(tf.keras.initializers.Initializer):
    """Initializer that generates a Gaussian kernel."""
    def __init__(self, l, std):
        self.l = l
        self.std = std

    def __call__(self, shape, dtype=None):
        # The Gaussian kernel is 2D. Ensure the shape is compatible.
        print(shape)
        assert len(shape) == 4
        kernel = gaussian_kernel(self.l, self.std)
        # Expand to the shape of the convolutional filters.
        gauss_kernel = tf.expand_dims(kernel, -1)
        gauss_kernel = tf.expand_dims(gauss_kernel, -1)
        print(tf.tile(gauss_kernel, [1, 1, shape[2], shape[3]]).shape)
        return tf.tile(gauss_kernel, [1, 1, shape[3], shape[3]])


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


class FixSumSoftmax(tf.keras.constraints.Constraint):
    """ Constrains weight tensors to be nonnegative and add up to `ref_value` 
    in the spatial dimensions. """
  
    def __init__(self, ref_value):
        self.ref_value = ref_value

    def __call__(self, w):
        w1 = tf.keras.activations.softmax(w, axis=[0, 1])
        return self.ref_value * w1

    def get_config(self):
        return {'ref_value': self.ref_value}

class FixSumBias(tf.keras.constraints.Constraint):
    """ Constrains weight tensors to be nonnegative and add up to `ref_value` 
    in the spatial dimensions. """
  
    def __init__(self, ref_value):
        self.ref_value = ref_value

    def __call__(self, w):
        # Add vmin only if negative
        wneg = w * tf.cast(tf.less_equal(w, 0.0), tf.keras.backend.floatx())
        vmin = tf.reduce_min(wneg, axis=[0, 1])
        w = w - vmin
        s = tf.reduce_sum(w, axis=[0, 1])

        # I don't think a nan can result here but just in case we add a small
        # offset.
        w1 = w / (s + 1e-4)
        
        return self.ref_value * w1

    def get_config(self):
        return {'ref_value': self.ref_value}


# Adapted from
# https://stackoverflow.com/questions/49189496/can-symmetrically-paddding-be-done-in-convolution-layers-in-keras (retrieved Thu Feb 23 10:28:53 2023)
class Padding2D(tf.keras.layers.Layer):
    def __init__(self, mode="SYMMETRIC",
                 padding=[1,1], data_format="channels_last", **kwargs):
        self.data_format = data_format
        self.padding = padding
        self.mode = mode
        super(Padding2D, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(Padding2D, self).build(input_shape)

    def call(self, inputs):
        if (self.data_format == "channels_last"):
            #(batch, depth, rows, cols, channels)
            pad = [[0,0]] + [[i,i] for i in self.padding] + [[0,0]]
        
        elif self.data_format == "channels_first":
            #(batch, channels, depth, rows, cols)
            pad = [[0, 0], [0, 0]] + [[i,i] for i in self.padding]

        if tf.keras.backend.backend() == "tensorflow":
            paddings = tf.constant(pad)
            out = tf.pad(inputs, paddings, self.mode)
        else:
            raise Exception("Backend " + tf.keras.backend.backend() + "not implemented")
        return out 
        
    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_last":
            #(batch, depth, rows, cols, channels)
            return (input_shape[0],
                    input.shape[1] + 2 * self.padding[0],
                    input.shape[2] + 2 * self.padding[1],
                    input_shape[3])

        elif self.data_format == "channels_first":
            #(batch, channels, depth, rows, cols)
            return (input_shape[0],
                    input_shape[1],
                    input.shape[2] + 2 * self.padding[0],
                    input.shape[3] + 2 * self.padding[1])


    def get_config(self):
        return {'mode': self.mode,
                'padding': self.padding,
                'data_format': self.data_format,
                'name': self.name}


# # Add the (flipped) values of the padding area.
class Fold2D(tf.keras.layers.Layer):
    def __init__(self, l, data_format="channels_last", **kwargs):
        self.data_format = data_format
        self.l = l
        if data_format != "channels_last":
            raise NotImplementedError(f"data_format={data_format} not implemented")


        super(Fold2D, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(Fold2D, self).build(input_shape)

    def call(self, x):
        input_shape = tf.shape(x)
        h = input_shape[1]
        w = input_shape[2]

        self.hflip = flipmat(self.l, h)
        self.wflip = flipmat(self.l, w)
        
        x = tf.add(x, tf.einsum("ij,kjlm->kilm", self.hflip, x))
        x = tf.add(x, tf.einsum("ij,kljm->klim", self.wflip, x))
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {'l': self.l,
                'data_format': self.data_format,
                'name': self.name}
    
    

def flipmat(l, n, flipend=False):
    """ Construct a matrix that 'flips' values of a vector along the axis 
    between elements l-1 and l (0-based).  Only considers values to the left of 
    the axis: all other values are ignored.

    If endside is True, does it also same for the values at the end of the 
    vector.
    """    
    indices = [[l + i, l - 1 - i] for i in range(l)]
    values = [1 for i in range(l)]

    if flipend is True:
        indices += [[n - l - 1 - i, n - l + i] for i in range(l)]
        values += values

    mat = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(indices, values, [n, n])))
    return tf.cast(mat, tf.keras.backend.floatx())


CUSTOM_OBJECTS = {'FixSum': FixSum,
                  'FixSumSoftmax': FixSumSoftmax,
                  'Padding2D': Padding2D,
                  'Fold2D': Fold2D,
                  'FixSumBias': FixSumBias,
                  '<lambda>': lambda x: None
                  }

if __name__ == '__main__':
    main()
