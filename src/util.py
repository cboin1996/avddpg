import h5py
import json
import tensorflow as tf
from types import SimpleNamespace

def save_file(fpath, txt):
    with open(fpath, 'w') as f:
        print(f"Saving {txt} to : {fpath}")
        f.write(txt)

def config_writer(fpath, obj):
    with open(fpath, 'w') as f:
        print(f"Saving configuration Config.py as json: outfile -> {fpath}.")
        json.dump(obj.__dict__, f)

def config_loader(fpath):
    with open(fpath, 'r') as f:
        return json.load(f, object_hook=lambda d: SimpleNamespace(**d))

def print_dct(dct):
    for k, v in dct.items():
        print(f"{k} : {v}")
    