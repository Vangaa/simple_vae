import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

SAVER_FILE = 'model.ckpt'
IMG_FOLDER = 'images'

def get_args():
    parser = argparse.ArgumentParser(description='Trainer for simple variation autoencoder.')
    
    parser.add_argument("--make_imgs", help="Store images", action="store_true")
    parser.add_argument("--digit", default = 3, help="Digit example to draw", type=int)
    parser.add_argument("lat_shape", help="Latent shape", type=int)
    parser.add_argument("--batch_size", default=512, help="Batch_size", type=int)
    parser.add_argument("--epoches", default=100, help="Epoches count", type=int)
    parser.add_argument("--hid_shape", default=128, help="Hidden shape", type=int)
    parser.add_argument("--log_after", default=20, help="Show test error after `count` batches", type=int)
    parser.add_argument("--dropout", default=0.2, help="Dropout", type=float)
    parser.add_argument("-s", "--save_model", help="Save session", action="store_true")
    parser.add_argument("--load_model", help="Load session from file", action="store_true")
    parser.add_argument("--lr", default=1e-2, help="Learning rate for AdaGrad optimizer", type=float)
    args = parser.parse_args()
    return args

def gen_data(batch_size, dataset):
    for i in range(0, int(dataset.shape[0]/batch_size)*batch_size, batch_size):
        batch = dataset[i: i+batch_size]
        yield batch

def get_example(digit, dataset):
    while True:
        x, y = dataset.next_batch(1)
        if np.argmax(y[0]) == digit:
            return x
        
def save_image(matrix, path):
    plt.imshow(matrix.reshape((28, 28)), cmap='gray')
    plt.savefig(path)

def prepare_image_folder():
    if not os.path.exists(IMG_FOLDER):
        os.system("mkdir {0}".format(IMG_FOLDER))
    else:
        os.system("rm {0}/*.png".format(IMG_FOLDER))