import os
import numpy as np

import tensorflow as tf
from matplotlib import pyplot as plt

from train import load2, padding
from model import CUSTOM_OBJECTS

def main():
    model = tf.keras.models.load_model('checkpoint.hdf5', 
                                       custom_objects=CUSTOM_OBJECTS)
    model.summary(positions=[.25, .6, .7, 1.])
    path = os.path.expanduser("~/data/denoise/charge_density/x16/original/")
    fname = "background_density10_negative_10kV_claw0021.hdf"

    q1, q = load2(os.path.join(path, fname))

    q1 = q1.reshape((1, *q1.shape))
    q0 = model.predict(q1)

    qtotal0 = np.sum(q0)
    qtotal1 = np.sum(q1)
    print(f"{qtotal0=}\n{qtotal1=}")
    print(q0.shape, q1.shape)
    
    vmax = 1.6
    plotq(q[:, :, 0], name="original", vmax=vmax)
    plotq(q1[0, :, :, 0], name="noisy", vmax=vmax)
    plotq(q0[0, :, :, 0], name="reconstructed", vmax=vmax)

    plt.show()


def plotq(q, name="", vmax=None):
    plt.figure(name)
    plt.suptitle(name, size=14)
    if vmax is None:
        vmax = np.max(abs(q))
    
    plt.pcolormesh(q, vmin=-vmax, vmax=vmax, cmap="bwr")
    plt.colorbar()
    plt.savefig("%s.pdf" % name)
    

if __name__ == '__main__':
    main()
