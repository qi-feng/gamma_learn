#from gamma_xgb import *
import pandas as pd
import numpy as np
import cPickle as pickle
from itertools import islice
from sklearn.cross_validation import StratifiedShuffleSplit

def quick_oversample2(pixVals, z_index, numX=54):
    z = -np.ones((numX, numX))
    for i_ in range(len(pixVals)):
        x_ = int(z_index.at[i_, 'x1'])
        y_ = int(z_index.at[i_,'y1'])
        z[x_:x_+2, y_:y_+2] = pixVals[i_]
    return z

def get_raw_features(infile='64080.txt', in_pickle_onfile = 'on0.95_64080.pkl', in_pickle_offfile = 'off0.005_64080.pkl', test_ratio=0.3, random_state=1234, dump=False, out_trainx='64080_raw_trainx.pkl', out_trainy='64080_raw_trainy.pkl', out_testx='64080_raw_testx.pkl', out_testy='64080_raw_testy.pkl'):
    inputfile = open(in_pickle_onfile, 'rb')
    on = pickle.load(inputfile)
    inputfile.close()
    inputfile = open(in_pickle_offfile, 'rb')
    off = pickle.load(inputfile)
    inputfile.close()
    on = on.astype(int)
    off = off.astype(int)
    #x_on = np.zeros((on.shape[0], 4016))
    #x_off = np.zeros((off.shape[0], 4016))
    x_on = np.zeros((on.shape[0], 4, 54, 54))
    x_off = np.zeros((off.shape[0], 4, 54, 54))
    y_on = np.ones(on.shape[0])
    y_off = np.zeros(off.shape[0])
    
    #read oversample z index:
    z_index = pd.read_csv("oversample_coordinates.csv")

    # read certain lines in a file, fast skipping
    current_ = 1
    #infile='64080.txt'
    with open(infile) as f:
        for i_, on_ in enumerate(on):
            line = list(islice(f, on_-current_, on_-current_+1))
            try:
                #print i_, on_, np.array(line[0].split())
                x_on_entry = np.array(line[0].split())
                for j_ in range(4):
                    x_on[i_, j_] = quick_oversample2(x_on_entry[504+4016/4*j_:504+500+4016/4*j_], z_index)
                current_ = on_+1
            except:
                print("Problem reading line %d: %s" % (on_, line))
    
    current_ = 1
    with open(infile) as f:
        for i_, off_ in enumerate(off):
            line = list(islice(f, off_-current_, off_-current_+1))
            try:
                #print i_, off_, np.array(line[0].split())
                x_off_entry = np.array(line[0].split())
                for j_ in range(4):
                    x_off[i_, j_] = quick_oversample2(x_off_entry[504+4016/4*j_:504+500+4016/4*j_], z_index)
                current_ = off_+1
            except:
                print("Problem reading line %d: %s" % (off_, line))
    
    x = np.concatenate([x_on, x_off])
    #x = np.delete(x, [0, 4016/4, 4016/4*2, 4016/4*3], 1)
    y = np.concatenate([y_on, y_off])
    
    sss = StratifiedShuffleSplit(y, test_size=test_ratio, random_state=random_state)
    for train_index, test_index in sss:
            break
    
    train_x, train_y = x[train_index], y[train_index]
    test_x, test_y = x[test_index], y[test_index]

    if dump:
        for f_, arr_ in zip([out_trainx, out_trainy, out_testx, out_testy], [train_x, train_y, test_x, test_y]):
            output = open(f_, 'wb')
            pickle.dump(arr_, output, protocol=pickle.HIGHEST_PROTOCOL)
            output.close()

    return train_x, train_y, test_x, test_y

def get_raw_features_on(infile='72044.txt', in_pickle_onfile = 'on72044.pkl', test_ratio=0.3, random_state=1234, 
                        dump=False, out_trainx='72044_raw_trainx_on.pkl', out_trainy='72044_raw_trainy_on.pkl', 
                        out_testx='72044_raw_testx_on.pkl', out_testy='72044_raw_testy_on.pkl'):
    inputfile = open(in_pickle_onfile, 'rb')
    on = pickle.load(inputfile)
    inputfile.close()
    on = on.astype(int)
    x_on = np.zeros((on.shape[0], 4, 54, 54))
    y_on = np.ones(on.shape[0])
    #read oversample z index:
    z_index = pd.read_csv("oversample_coordinates.csv")
    # read certain lines in a file, fast skipping
    current_ = 1
    with open(infile) as f:
        for i_, on_ in enumerate(on):
            line = list(islice(f, on_-current_, on_-current_+1))
            try:
                #print i_, on_, np.array(line[0].split())
                x_on_entry = np.array(line[0].split())
                for j_ in range(4):
                    x_on[i_, j_] = quick_oversample2(x_on_entry[504+4016/4*j_:504+500+4016/4*j_], z_index)
                current_ = on_+1
            except:
                print("Problem reading line %d: %s" % (on_, line))

    x = x_on
    y = np.ones(x.shape[0])
    sss = StratifiedShuffleSplit(y, test_size=test_ratio, random_state=random_state)
    for train_index, test_index in sss:
            break
    train_x, train_y = x[train_index], y[train_index]
    test_x, test_y = x[test_index], y[test_index]
    if dump:
        for f_, arr_ in zip([out_trainx, out_trainy, out_testx, out_testy], [train_x, train_y, test_x, test_y]):
            output = open(f_, 'wb')
            pickle.dump(arr_, output, protocol=pickle.HIGHEST_PROTOCOL)
            output.close()
    return train_x, train_y, test_x, test_y

def load_pickle(f):
    inputfile = open(f, 'rb')
    loaded = pickle.load(inputfile)
    inputfile.close()
    return loaded


