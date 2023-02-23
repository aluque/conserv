import os
import numpy as np

import tensorflow as tf
from matplotlib import pyplot as plt

from train import load2
from model import CUSTOM_OBJECTS

def main():
    model = tf.keras.models.load_model('checkpoint.hdf5',
                                       custom_objects=CUSTOM_OBJECTS)
    K = model.get_layer("K").get_weights()[0]
    
    plt.figure("kernels", figsize=(20, 10))
    for i in range(32):
        ax = plt.subplot(4, 8, i + 1)
        ax.matshow(K[:, :, i])

    probing_layer = tf.keras.Model(inputs=model.input,
                                   outputs=model.get_layer("softmax").output)

    path = os.path.expanduser("~/data/denoise/charge_density/x16/original/")
    fname = "background_density10_negative_10kV_claw0021.hdf"
    q1, q = load2(os.path.join(path, fname))
    q1 = q1.reshape((1, *q1.shape))

    c = probing_layer.predict(q1)

    plt.figure("weights", figsize=(20, 10))
    for i in range(32):
        ax = plt.subplot(4, 8, i + 1)
        ax.pcolormesh(c[0, :, :, i], vmin=0, vmax=1, cmap="gnuplot2")
        ticks = ax.get_xticks()
        ax.set_xticklabels(["" for t in ticks])

        ticks = ax.get_yticks()
        ax.set_yticklabels(["" for t in ticks])

    plt.show()
    
if __name__ == '__main__':
    main()
