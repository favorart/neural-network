# from collections import defaultdict
# from math import *
import numpy as np
import random
import sys
import re
import os

import matplotlib.image as mpimg

from visual import ann_visual_weights, rbm_visual_weights
from test_ann import test_ann
from test_rbm import test_rbm


def input_data(data_folder, n_classes=26, test_samples=[6,7,8], valid_samples=[5,6]):
    """ """
    in_file_names = [ fn for fn in os.listdir(data_folder) 
                      if os.path.isfile( os.path.join(data_folder, fn) ) ]

    X_test  = []  # TEST
    y_test  = [] # * n_classes 
    X_train = [] # TRAIN
    y_train = [] # * n_classes
    y_valid = [] # VALID
    X_valid = [] # * n_classes 

    y = [0] * 26
    for fn in in_file_names:

        m = re.match(ur'[a-z]+-(\d+)(?:-(\d+))?(?:-0)?.bmp', fn)
        # print fn, m.group(0)
        if m:
            iy = int(m.group(1))
            iz = int(m.group(2)) if m.group(2) else 0

            image = mpimg.imread( os.path.join(data_folder, fn) )
            b = np.array(image[:,:,0] > 0, dtype=int).flatten()

            y[iy] = 1
            if   iz in test_samples:
                y_test.append(y[:])
                X_test.append(b)
            # elif iz in valid_samples:
            #     y_valid.append(y[:])
            #     X_valid.append(b)
            else:
                y_train.append(y[:])
                X_train.append(b)

            y[iy] = 0

    # inputs by rows
    X_test  = np.array(X_test)
    y_test  = np.array(y_test)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # X_valid = np.array(X_valid)
    # y_valid = np.array(y_valid)
    
    print X_test.shape, y_test.shape, \
          X_train.shape, y_train.shape #, \
          # X_valid.shape, y_valid.shape
    
    return X_train, y_train, X_test, y_test # , X_valid, y_valid


if __name__ == '__main__':

    if sys.argv[1] == '-nt':

        X_train, y_train, X_test, y_test = input_data("./data/", test_samples=[7,8])

        test_ann(X_train, y_train, X_test, y_test, [X_train.shape[1], 30, y_train.shape[1]])


    elif sys.argv[1] == '-nv':

        X_train, y_train, X_test, y_test = input_data("./data/", test_samples=[7,8])

        ann_visual_weights(X_train, y_train, X_test, y_test)


    elif sys.argv[1] == '-rt':

        X_train, y_train, X_test, y_test = input_data("./data/", test_samples=[])

        test_rbm( X_train, y_train, X_test, y_test)


    elif sys.argv[1] == '-rv':

        X_train, y_train, X_test, y_test = input_data("./data/", test_samples=[])

        rbm_visual_weights(X_train, y_train, X_test, y_test)


    else: print 'incorrect console argument'

