import os
import numpy as np
import argparse

import tensorflow as tf
from matplotlib import pyplot as plt

from train import load2, padding
from model import CUSTOM_OBJECTS
from conf import CONF

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Input parameters", default=None)

    parser.add_argument("--rlim", "-r", help="Limits in r", default=None)
    parser.add_argument("--zlim", "-z", help="Limits in z", default=None)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    model = tf.keras.models.load_model(CONF["model_weights"], 
                                       custom_objects=CUSTOM_OBJECTS)
    model.summary(positions=[.25, .6, .7, 1.])

    q1, q = load2(CONF["sample_file_path"])

    q1 = q1.reshape((1, *q1.shape))
    q0 = model.predict(q1)

    kwargs = {}
    if args.rlim is not None:
        kwargs["rlim"] = [float(x) for x in args.rlim.split(":")]

    if args.zlim is not None:
        kwargs["zlim"] = [float(x) for x in args.zlim.split(":")]

    vmax = plotq(q[:, :, 0], name="original", vmax=None, **kwargs)
    plotq(q1[0, :, :, 0], name="noisy", vmax=vmax, **kwargs)
    plotq(q0[0, :, :, 0], name="reconstructed", vmax=vmax, **kwargs)

        

    plt.show()


def plotq(q, name="", vmax=None, rlim=None, zlim=None):
    plt.figure(name)
    plt.suptitle(name, size=14)

    if not CONF["cylindrical"]:
        r = 0.5 + np.arange(q.shape[1])
        q = q * r

    qtotal = np.sum(q)
    print(f"[{name}] total charge = {qtotal}")

    if vmax is None:
        vmax = np.max(abs(q))
        
    plt.pcolormesh(q, vmin=-vmax, vmax=vmax, cmap="bwr")
    plt.colorbar()

    if rlim is not None:
        plt.xlim(rlim)

    if zlim is not None:
        plt.ylim(zlim)
        
    #plt.savefig("%s.pdf" % name)

    return vmax

if __name__ == '__main__':
    main()
