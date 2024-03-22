import os
import numpy as np
import tensorflow as tf
import h5py
from multiprocessing import cpu_count
import hrnet
import model
from conf import CONF

buildmodel = {"model": model.buildmodel,
              "hrnet": hrnet.buildmodel}[CONF["model"]]

def main():
    mirrored_strategy = tf.distribute.MirroredStrategy()

    os.makedirs(CONF["output_path"], exist_ok=True)

    (ds, val) = dataset(path=CONF['dataset_path'])

    with mirrored_strategy.scope():
        model = buildmodel(filters=CONF['filters'], l=CONF['l'], m=CONF['m'],
                           center_scale_norm=CONF["center_scale_norm"], concat_input=True)


        cp = tf.keras.callbacks.ModelCheckpoint(CONF['saved_model'],
                                                monitor='val_loss', mode='min',
                                                verbose=1, save_best_only=True)
        log = tf.keras.callbacks.CSVLogger(CONF['training_log'])
        # optimizer = tf.keras.optimizers.Adam(clipvalue=100,
        #                                      decay=CONF["weight_decay"])
        optimizer = tf.keras.optimizers.Adam(learning_rate=CONF["init_learning_rate"])

        loss = lambda y_true, y_pred: regloss(y_true, y_pred, model,
                                              alpha=CONF["alpha"],
                                              weight_factor=CONF["weight_factor"],
                                              N=128)
        model.compile(optimizer=optimizer, loss=loss)

    r = model.fit(ds,
                  validation_data = val,
                  epochs     = CONF["epochs"],
                  batch_size = CONF["training_batch"],
                  workers    = CONF["workers"],
                  verbose    = CONF["fit_verbose"],
                  use_multiprocessing=True,
                  callbacks=(cp,log))

    model.summary()


def regloss(y_true, y_pred, model, alpha=0.01, weight_factor=0.01, l=6, N=128):
    """ A loss functon with regularization that takes into account the
    convolutional instability.

    alpha is the factor that amplifies this reg. loss before it is added to the
    mse.
    """
    shape = tf.shape(y_true)

    y_input = tf.reshape(y_pred[:, :, :, 1], (shape[0], shape[1], shape[2], 1))
    y_pred = tf.reshape(y_pred[:, :, :, 0], (shape[0], shape[1], shape[2], 1))

    err = y_true - y_pred
    if CONF["cylindrical"]:
        r = tf.constant(0.5) + tf.range(0, shape[2], dtype=tf.float32)
        err = err / tf.reshape(r, (1, 1, shape[2], 1))
        y_true1 = y_true / tf.reshape(r, (1, 1, shape[2], 1))
    else:
        y_true1 = y_true
    mse = tf.reduce_mean(tf.square(err) * (1 + tf.square(y_true1) / tf.constant(weight_factor**2)))

    # Avoid intsbility term
    z_input = tf.signal.fft2d(tf.cast(y_input, tf.complex64))
    z_pred = tf.signal.fft2d(tf.cast(y_pred, tf.complex64))
    z1 = tf.math.conj(z_input) * z_pred
    reg = tf.reduce_mean(tf.nn.relu(-tf.math.real(z1)))

    return mse + alpha * reg


def dataset(path="", train_ratio=0.8):
    AUTOTUNE = tf.data.AUTOTUNE

    all_files = tf.data.Dataset.list_files(os.path.join(path, "original", "*.hdf"),
                                           seed=CONF["seed"], shuffle=True)
    nfiles = len(all_files)
    train_size = int(train_ratio * nfiles)
    train_files = all_files.take(train_size)
    val_files = all_files.skip(train_size)

    def create_dataset(files):
        ds = files.shuffle(buffer_size = 10000, seed=CONF["seed"],
                           reshuffle_each_iteration = True)
        ds = ds.map(lambda fname: tf.py_function(func = load2, inp=[fname],
                                                 Tout=(tf.float32, tf.float32)),
                    num_parallel_calls=AUTOTUNE, deterministic=True)

        ds = ds.batch(CONF["training_batch"])
        ds = ds.prefetch(AUTOTUNE)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
        return ds.with_options(options)

    train = create_dataset(train_files)
    val = create_dataset(val_files)

    return (train, val)

def loadq(filename):
    if "zero_patch" in CONF:
        c = CONF["zero_patch"]
        if "disable" in c and c["disable"]:
            zero_patch = None
        else:
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
