import os
import glob

import h5py
import numpy as np


def get_parser():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path with original files",
                        default=os.path.expanduser("~/data/denoise/charge_density"))
    parser.add_argument("-x", help="Reduction factor", type=int)
    
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    for subset in ["original", "noisy_50"]:
        resample(args.path, subset, args.x)

def resample(path, subset, step):
    rpath = os.path.join(path, "x%d" % step, subset)
    
    os.makedirs(rpath, exist_ok=True)
    
    for pfname in glob.iglob(os.path.join(path, subset, "*.hdf")):
        print(pfname)

        _, fname = os.path.split(pfname)
        rfname = os.path.join(rpath, fname)
        
        q = loadfile(pfname)
        savefile(rfname, q[::step, ::step])


def loadfile(fname):
    with h5py.File(fname, mode='r') as file:
        patch = list(file.keys())[0]        # name of the first group contained in the file
    
        group = file[patch]                 # first group
        dataset = list(group.keys())[0]     # name of the first dataset in the group (c-density)
        q = np.array(group[dataset]).astype('float32', copy = False)

    return q


def savefile(fname, d):
    with h5py.File(fname, mode='w') as file:  # loading the hdf5 file
        g = file.create_group("group1")
        dset = g.create_dataset("charge_density", data=d, compression="lzf",
                                dtype='float32')
    
if __name__ == '__main__':
    main()
