import collections.abc
import os
import platform
import argparse
    
import tomli

def load_conf(fname):
    mypath = os.path.dirname(os.path.realpath(__file__))
    confpath = os.path.join(mypath, "conf")
    hostname = platform.node()
    
    d = maybe_load_toml(os.path.join(confpath, "defaults.toml"))
    d = update(d, maybe_load_toml(os.path.join(confpath, f"{hostname}.toml")))

    if fname is not None:
        name = os.path.splitext(os.path.basename(fname))[0]
        inputpath = os.path.dirname(os.path.realpath(fname))
        d = update(d, {'name': name, 'path': inputpath})
    
        with open(fname, "rb") as f:
            u = tomli.load(f)

        d = update(d, u)

    while expand(d):
        pass
    
    return d

    
def maybe_load_toml(fname):
    try:
        with open(fname, "rb") as f:
            d = tomli.load(f)
    except FileNotFoundError:
        d = {}

    return d


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def expand(d, root=None):
    if root is None:
        root = d

    changed = False
    for k, v in d.items():
        if isinstance(v, collections.abc.Mapping):
            changed1 = expand(v)
            changed2 = expand(v, root)
            changed = changed1 or changed2
            
        elif isinstance(v, str):
            newstr = os.path.expanduser(v.format(**root))
            if newstr != v:
                changed = True
                d[k] = newstr

    return changed
        

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="Input parameters", default=None)
args = parser.parse_args()
CONF = load_conf(args.i)
