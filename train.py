import os
import numpy as np
import tensorflow as tf
import h5py
from multiprocessing import cpu_count
from model import buildmodel


SEED = 2021             # seed to initialize the random number generator
BATCH_SIZES = (8, 4, 1) # Batch size for datasets (training, validation, test = 1)
WORKERS = cpu_count()   # Number of CPUs for parallel operations

def main():
    model = buildmodel(l=4, m=3)
    model.summary()

    ds = dataset()
    workers = 4 #cpu_count()
    print(f"{workers=}")
    cp = tf.keras.callbacks.ModelCheckpoint('checkpoint.hdf5', monitor='loss', mode='min',
                                            verbose=1, save_best_only=True)
    model.compile(loss='mse', optimizer='Adam')
    r = model.fit(ds, epochs=1000, batch_size=BATCH_SIZES[0],
                  use_multiprocessing=True, verbose=2, workers=workers,
                  callbacks=(cp,))


def dataset(path=os.path.expanduser("~/data/denoise/charge_density/x16/original/")):
    AUTOTUNE = tf.data.AUTOTUNE

    ds = tf.data.Dataset.list_files(os.path.join(path, "*.hdf"), seed=SEED, shuffle=True)
    ds = ds.map(lambda fname: tf.py_function(func = load2, inp=[fname],
                                             Tout=(tf.float32, tf.float32)),
                num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(BATCH_SIZES[0])
    ds = ds.prefetch(AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
    ds = ds.with_options(options)
    return ds


def loadq(filename):
    file = h5py.File(filename, mode='r')   # loading the hdf5 file

    patch = list(file.keys())[0]        # name of the first group contained in the file
    group = file[patch]                 # first group
    dataset = list(group.keys())[0]     # name of the first dataset in the group (c-density)
    q = np.array(group[dataset]).astype('float32', copy = False)
    file.close()

    return q


def load2(filename):
    if not isinstance(filename, str):
        filename = filename.numpy().decode("utf-8") 
    
    q = loadq(filename)
    q = np.reshape(q, (q.shape[0], q.shape[1], 1))

    q1 = loadq(filename.replace('original', 'noisy_50'))
    q1 = np.reshape(q1, (q1.shape[0], q1.shape[1], 1))
    return q1, q


def padding(x, l):
    """
    Add l pixels of padding to all sides of x using reflection.
    """
    
    newsize = [n + 2 * l for n in x.shape]
    x1 = np.zeros(newsize)
    x1[l:-l, l:-l] = x

    x1[0:l, :] = x1[2 * l - 1:l - 1:-1, :]
    x1[-l:, :] = x1[-(l + 1):-(2 * l + 1):-1, :]

    x1[:, 0:l] = x1[:, 2 * l - 1:l - 1:-1]
    x1[:, -l:] = x1[:, -(l + 1):-(2 * l + 1):-1]

    return x1

    
if __name__ == '__main__':
    main()
