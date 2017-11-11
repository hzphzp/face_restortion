#!/usr/bin/env python

from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle
from time import time
import sys
import h5py
from tqdm import tqdm

image_dir_x = '/data/zhoutk/face_restortion/dcgan_resize/img_align_celeba/'
image_dir_y = '/data/zhoutk/face_restortion/dcgan_resize/img_align_celeba/'
try:
    image_locs_x = [join(image_dir_x, f) for f in listdir(image_dir_x) if isfile(join(image_dir_x, f))]
    image_locs_y = [join(image_dir_y, f) for f in listdir(image_dir_y) if isfile(join(image_dir_y, f))]

except:
    print("expected aligned images directory, see README")
# print(image_locs_x[:5])
# print(image_locs_y[:5])

total_imgs = len(image_locs_x)
print("found %i images in directory" % total_imgs)


def process_image(im):
    if im.mode != "RGB":
        im = im.convert("RGB")
    target = np.array(im)
    return target[:, :, :]


def proc_loc(loc_x, loc_y):
    try:
        i = Image.open(loc_x)
        input = process_image(i)
        j = Image.open(loc_y)
        target = process_image(j)
        return (input, target)
    except KeyboardInterrupt:
        raise
    except:
        return None


try:
    hf = h5py.File('faces.hdf5', 'r+')
except:
    hf = h5py.File('faces.hdf5', 'w')

try:
    dset_t = hf.create_dataset("target", (1, 218, 178, 3),
                               maxshape=(1e6, 218, 178, 3), chunks=(1, 218, 178, 3), compression="gzip")
except:
    dset_t = hf['target']

try:
    dset_i = hf.create_dataset("input", (1, 218, 178, 3),
                               maxshape=(1e6, 218, 178, 3), chunks=(1, 218, 178, 3), compression="gzip")
except:
    dset_i = hf['input']

batch_size = 1
num_iter = total_imgs / 1

insert_point = 0
print("STARTING PROCESSING IN BATCHES OF %i" % batch_size)

for i in tqdm(range(num_iter)):
    # sys.stdout.flush()

    X_in = []
    X_ta = []

    a = time()
    # loc_x = [join(image_dir_x,"ori"+str(i+1)+".bmp")]
    loc_x = join(image_dir_x, image_locs_x[i])
    # loc_y = [join(image_dir_y,"ori"+str(i+1)+".bmp")]
    loc_y = join(image_dir_y, image_locs_y[i])
    proc = [proc_loc(loc_x, loc_y)]
    for pair in proc:
        if pair is not None:
            input, target = pair
            X_in.append(input)
            X_ta.append(target)
            # else:
            # print("ERROR")

    X_in = np.array(X_in)
    X_ta = np.array(X_ta)

    dset_i.resize((insert_point + len(X_in), 218, 178, 3))
    dset_t.resize((insert_point + len(X_in), 218, 178, 3))

    dset_i[insert_point:insert_point + len(X_in)] = X_in
    dset_t[insert_point:insert_point + len(X_in)] = X_ta

    insert_point += len(X_in)
print(hf['input'].shape[0])
hf.close()
