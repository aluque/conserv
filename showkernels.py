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
from conf import CONF

def main(savefigs=True):
    os.makedirs(CONF["plots_path"], exist_ok=True)

    model = tf.keras.models.load_model(CONF['saved_model'],
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
        plt.savefig(os.path.join(CONF["plots_path"], "kernels.pdf"))

    probing_layer = tf.keras.Model(inputs=model.input,
                                   outputs=model.get_layer("softmax").output)

    q1, q = load2(CONF["sample_file_path"])
    q1 = q1.reshape((1, *q1.shape))

    c = probing_layer.predict(q1)
    
    fig = plt.figure("weights", figsize=(20, 10))
    plt.suptitle("Partition coeffs $c^{(n)}$", size=16)
    s = 16 // CONF["resample_factor"]
    for i in range(4):
        for j in range(8):
            n = i * 8 + j
            ax = plt.subplot(4, 8, n + 1)
            p = ax.pcolormesh(c[0, ::s, ::s, n], vmin=0, vmax=1, cmap="inferno")
            
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
        plt.savefig(os.path.join(CONF["plots_path"], "partition.pdf"))

    plt.show()
    
if __name__ == '__main__':
    main()
