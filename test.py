import os

import tensorflow as tf
from matplotlib import pyplot as plt

from scratch import load2

def main():
    model = tf.keras.models.load_model('checkpoints')
    path = "/Volumes/T7/data/denoise/charge_density/original/"
    fname = "background_density10_negative_10kV_claw0021.hdf"
    q1, q = load2(os.path.join(path, fname))
    q0 = model.predict(q1)
    
    plt.figure("original")
    plt.pcolormesh(q[:, :, 0])

    plt.figure("noisy")
    plt.pcolormesh(q1[:, :, 0])

    plt.figure("reconstructed")
    print(q0.shape)
    plt.pcolormesh(q0[:, :, 0, 0])

    plt.show()
    
if __name__ == '__main__':
    main()
