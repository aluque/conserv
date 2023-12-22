import tensorflow as tf
from keras import backend as K

from conf import CONF

def main():
    model = buildmodel(l=CONF['l'])
    model.summary(positions=[.25, .6, .7, 1.])
    model.compile(loss='mse', optimizer='Adam')
    model.save(CONF["saved_model"])
    print(CONF["saved_model"])


def init_layer(layer):
    session = K.get_session()
    weights_initializer = tf.variables_initializer(layer.weights)
    session.run(weights_initializer)

    
def buildmodel(filters=32, l=4, m=3):
    lin_conv_size = 2 * l + 1
    
    input_size = (None, None, 1)
    inputs = tf.keras.Input(shape = input_size, name = "inputs")

    # Obtaining the weights b and c
    n = Padding2D(name="pad_conv1", padding=[l, l])(inputs)
    homogen = tf.keras.initializers.Constant(1/lin_conv_size**2)
    avg = tf.keras.layers.DepthwiseConv2D(lin_conv_size,
                                        name="averaging",
                                        padding="valid",
                                        use_bias=False,
                                        trainable=False,
                                        kernel_initializer=homogen)(n)
    model = tf.keras.Model(inputs=inputs, outputs=avg)
    optimizer = tf.keras.optimizers.Adam(clipvalue=100,
                                         weight_decay=CONF["weight_decay"])
    layer = model.get_layer("averaging")
    layer.set_weights([layer.kernel_initializer(layer.weights[0].shape)])
    print(layer.get_weights()[0])
    
    return model



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



CUSTOM_OBJECTS = {'Padding2D': Padding2D}

if __name__ == '__main__':
    main()
