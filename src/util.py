import h5py
import tensorflow as tf

def save_file(fpath, txt):
    with open(fpath, 'w') as f:
        print(f"Saving {txt} to : {fpath}")
        f.write(txt)
