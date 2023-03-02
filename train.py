import os
import numpy as np
import tensorflow as tf
import h5py
from multiprocessing import cpu_count
from model import buildmodel
from conf import CONF


def main():
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = buildmodel(l=4, m=3)

    model.summary()

    os.makedirs(CONF["output_path"], exist_ok=True)

    ds = dataset(path=CONF['dataset_path'])
    cp = tf.keras.callbacks.ModelCheckpoint(CONF['saved_model'],
                                            monitor='loss', mode='min',
                                            verbose=1, save_best_only=True)
    log = tf.keras.callbacks.CSVLogger(CONF['training_log'])
    optimizer = tf.keras.optimizers.Adam(clipvalue=100,
                                         weight_decay=CONF["weight_decay"])
    
    model.compile(loss=CONF["loss"], optimizer=optimizer)
    r = model.fit(ds,
                  epochs     = CONF["epochs"],
                  batch_size = CONF["training_batch"],
                  workers    = CONF["workers"],
                  verbose    = CONF["fit_verbose"],
                  use_multiprocessing=True,
                  callbacks=(cp,log))


def dataset(path=""):
    AUTOTUNE = tf.data.AUTOTUNE

    ds = tf.data.Dataset.list_files(os.path.join(path, "original", "*.hdf"),
                                    seed=CONF["seed"], shuffle=True)
    ds = ds.shuffle(buffer_size = 10000, seed=CONF["seed"],
                    reshuffle_each_iteration = True)
    ds = ds.map(lambda fname: tf.py_function(func = load2, inp=[fname],
                                             Tout=(tf.float32, tf.float32)),
                num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(CONF["training_batch"])
    ds = ds.prefetch(AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
    ds = ds.with_options(options)
    return ds


def loadq(filename):
    if "zero_patch" in CONF:
        c = CONF["zero_patch"]
        h = c["height"]
        w = c["width"]

        if c.get("scale_with_resample", False):
            h = h // CONF["resample_factor"]
            w = w // CONF["resample_factor"]
            
        zero_patch = np.s_[0:h], np.s_[0:w]
    else:
        zero_patch = None
    
    file = h5py.File(filename, mode='r')   # loading the hdf5 file

    patch = list(file.keys())[0]        # name of the first group contained in the file
    group = file[patch]                 # first group
    dataset = list(group.keys())[0]     # name of the first dataset in the group (c-density)
    q = np.array(group[dataset]).astype('float32', copy = False)

    if not zero_patch is None:
        q[zero_patch[0], zero_patch[1]] = 0
        
    file.close()

    if CONF["cylindrical"]:
        r = 0.5 + np.arange(q.shape[1])
        q = q * r

    return q


def load2(filename):
    if not isinstance(filename, str):
        filename = filename.numpy().decode("utf-8") 
    
    q = loadq(filename)
    q = np.reshape(q, (q.shape[0], q.shape[1], 1))

    q1 = loadq(filename.replace('original', CONF["noisy_samples"]))
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
