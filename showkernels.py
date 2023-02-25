import os
import numpy as np

import tensorflow as tf
from matplotlib import pyplot as plt

try:
    plt.style.use("granada")
except OSError:
    pass

from train import load2
from model import CUSTOM_OBJECTS

def main(savefigs=False):    
    model = tf.keras.models.load_model('checkpoint.hdf5',
                                       custom_objects=CUSTOM_OBJECTS)
    K = model.get_layer("K").get_weights()[0]
    
    fig = plt.figure("kernels", figsize=(20, 10))
    plt.suptitle("Kernels $K^{(n)}$", size=16)
    vmax = np.max(K)
    for i in range(32):
        ax = plt.subplot(4, 8, i + 1)
        #p = ax.matshow(K[:, :, i], cmap="inferno", vmin=0, vmax=vmax)
        p = ax.pcolormesh(np.squeeze(K[:, :, i]), cmap="inferno", vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(K.shape[1]))
        ax.set_yticks(np.arange(K.shape[0]))
        ax.set_aspect("equal")
        
    cbar = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    plt.colorbar(p, cax=cbar)
    
    if savefigs:
        plt.savefig("kernels.pdf")

    probing_layer = tf.keras.Model(inputs=model.input,
                                   outputs=model.get_layer("softmax").output)

    path = os.path.expanduser("~/data/denoise/charge_density/x16/original/")
    fname = "background_density10_negative_10kV_claw0021.hdf"
    q1, q = load2(os.path.join(path, fname))
    q1 = q1.reshape((1, *q1.shape))

    c = probing_layer.predict(q1)
    
    fig = plt.figure("weights", figsize=(20, 10))
    plt.suptitle("Partition coeffs $c^{(n)}$", size=16)
    for i in range(4):
        for j in range(8):
            n = i * 8 + j
            ax = plt.subplot(4, 8, n + 1)
            p = ax.pcolormesh(c[0, :, :, n], vmin=0, vmax=1, cmap="inferno")
            
            if i != 3:
                ticks = ax.get_xticks()
                ax.set_xticklabels(["" for t in ticks])
            else:
                ax.set_xlabel("r (px)")
                
            if j != 0:
                ticks = ax.get_yticks()
                ax.set_yticklabels(["" for t in ticks])

            else:
                ax.set_ylabel("z (px)")

    cbar = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    plt.colorbar(p, cax=cbar)

    if savefigs:
        plt.savefig("partition.pdf")

    plt.show()
    
if __name__ == '__main__':
    main()
