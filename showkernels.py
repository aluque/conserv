import os
import numpy as np

import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize

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
    print(CONF['saved_model'])

    K = model.get_layer("K").get_weights()[0]

    fig = plt.figure("kernels", figsize=(20, 10))
    plt.suptitle("Kernels $K^{(n)}$", size=16)
    vmax = np.max(K)
    for i in range(K.shape[2]):
        ax = plt.subplot(4, 8, i + 1)
        #p = ax.matshow(K[:, :, i], cmap="inferno", vmin=0, vmax=vmax)
        p = ax.pcolormesh(np.squeeze(K[:, :, i]), cmap="inferno",
                          vmin=0, vmax=vmax)
                          #norm=LogNorm(vmin=vmax/1e2, vmax=vmax))
        print(i, ": ", np.sum(K[:, :, i]))
        for j in range(0, K.shape[0]):
            print("    ", j, ": ", np.sum(K[j, :, i]))
            
        ax.set_xticks(np.arange(K.shape[1]))
        ax.set_yticks(np.arange(K.shape[0]))
        ax.set_aspect("equal")

    # plt.show()
    # return

    cbar = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    plt.colorbar(p, cax=cbar)

    if savefigs:
        plt.savefig(os.path.join(CONF["plots_path"], "kernels.pdf"))

    probing_layer = tf.keras.Model(inputs=model.input,
                                   outputs=model.get_layer("softmax").output)

    q1, q = load2(CONF["sample_file_path"], False)
    q1 = q1.reshape((1, *q1.shape))

    c = probing_layer.predict(q1)

    fig = plt.figure("weights", figsize=(20, 10))
    plt.suptitle("Partition coeffs $c^{(n)}$", size=16)
    s = 16 // CONF["resample_factor"]
    for n in range(K.shape[2]):
        i = n // 8
        j = n % 8

        ax = plt.subplot(4, 8, n + 1)
        p = ax.pcolormesh(c[0, 1::s, 1::s, n], cmap="inferno", norm=Normalize(vmin=1e-3, vmax=1))
        
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
