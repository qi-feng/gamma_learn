__author__ = 'qfeng'


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
#from keras.layers.normalization import BatchNormalization
#from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from sklearn.metrics import roc_auc_score, roc_curve, auc
from get_raw_features import *
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.models import model_from_json

import matplotlib.pyplot as plt
import os

def do_cnn(train_x, train_y, test_x, test_y, input_shape=(4, 54, 54), nb_classes=2, loss_func='binary_crossentropy',
           filter_n1=32, filter_size1=6, filter_stride1=2, border_mode1='valid', pool_size1=2, filter_drop1=0.25,
           filter_n2=32, filter_size2=3, filter_stride2=1, border_mode2='same', pool_size2=2, filter_drop2=0.25,
           filter_n3=32, filter_size3=3, filter_stride3=1, border_mode3='same', pool_size3=2, filter_drop3=0.25,
           dense_n1=256, dense_drop1=0.5, dense_n2=64, dense_drop2=0.5, batch_size=128, nb_epoch=5, norm_x=1.,
           lr=0.01, early_stop=10, weights_file= 'mnist_best_weights.hdf5'):

    print("Building a ConvNet model...")
    model = Sequential()

    # input: 54x54 images with 4 tels -> (4, 54, 54) tensors.
    # this applies 64 convolution filters of size 6x6 each.
    model.add(Convolution2D(filter_n1, filter_size1, filter_size1, subsample=(filter_stride1, filter_stride1),
                            border_mode=border_mode1, input_shape=input_shape))
    model.add(Activation('relu'))
    #model.add(Convolution2D(filter_n1, filter_size1, filter_size1, subsample=(filter_stride1, filter_stride1), border_mode=border_mode1))
    model.add(Convolution2D(filter_n2, filter_size2, filter_size2, subsample=(filter_stride2, filter_stride2), border_mode=border_mode2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size1, pool_size1), border_mode='valid'))
    model.add(Dropout(filter_drop1))

    model.add(Convolution2D(filter_n2, filter_size2, filter_size2, subsample=(filter_stride2, filter_stride2), border_mode=border_mode2))
    model.add(Activation('relu'))
    model.add(Convolution2D(filter_n2, filter_size2, filter_size2, subsample=(filter_stride2, filter_stride2), border_mode=border_mode2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size2, pool_size2), border_mode='valid'))
    model.add(Dropout(filter_drop2))

    if filter_n3>0:
        model.add(Convolution2D(filter_n3, filter_size3, filter_size3, subsample=(filter_stride3, filter_stride3), border_mode=border_mode3))
        model.add(Activation('relu'))
        model.add(Convolution2D(filter_n3, filter_size3, filter_size3, subsample=(filter_stride3, filter_stride3), border_mode=border_mode3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(pool_size3, pool_size3), border_mode='valid'))
        model.add(Dropout(filter_drop3))


    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(dense_n1))
    model.add(Activation('relu'))
    model.add(Dropout(dense_drop1))

    if dense_n2>0:
        model.add(Dense(dense_n2))
        #model.add(Activation('relu'))
        model.add(Activation('tanh'))
        model.add(Dropout(dense_drop2))

    model.add(Dense(nb_classes))

    if nb_classes<=2:
        model.add(Activation('sigmoid'))
    else:
        model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd)

    if os.path.isfile(weights_file):
        model.load_weights(weights_file)
        print "load weights successful!"
    model.compile(loss=loss_func, optimizer=sgd)
    #model.compile(loss='binary_crossentropy', optimizer="rmsprop")

    if len(train_y.shape)==1:
        #add a new axis to y
        train_y = np_utils.to_categorical(train_y, nb_classes)

    if len(test_y.shape)==1:
        #add a new axis to y
        test_y = np_utils.to_categorical(test_y, nb_classes)

    #normalize x:
    if train_x.dtype != 'float32':
        train_x = train_x.astype('float32')
    if train_y.dtype != 'float32':
        train_y = train_y.astype('float32')
    if test_x.dtype != 'float32':
        test_x = test_x.astype('float32')
    if test_y.dtype != 'float32':
        test_y = test_y.astype('float32')

    train_x = train_x / norm_x
    test_x = test_x / norm_x

    print("Training the ConvNet model...")

    checkpointer = ModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True)
    early_stop = EarlyStopping(patience=early_stop, verbose=1)
    m_history = model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1,
                          validation_data=(test_x, test_y), callbacks=[checkpointer, early_stop])

    #m_history = model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True)
    #objective_score = model.evaluate(test_x, test_y, batch_size=batch_size)
    #print "The objective score on test data is", objective_score


    if nb_classes==2:
        predict_train_y = model.predict_proba(train_x)
        predict_test_y = model.predict_proba(test_x)
        fpr, tpr, thresh = roc_curve(train_y[1],predict_train_y[1])
        roc_auc = auc(fpr, tpr)
        fpr_test, tpr_test, thresh_test = roc_curve(test_y[1],predict_test_y[1])
        roc_auc_test = auc(fpr_test, tpr_test)
        print 'The training AUC score is {0}, and the test AUC score is: {1}'.format(roc_auc, roc_auc_test)
        print 'The mean prediction for training data is {0}, and for test data is: {1}'.format(np.mean(predict_train_y), np.mean(predict_test_y))

    return model, m_history

def search_cnn(train_x, train_y, test_x, test_y, input_shape=(4, 54, 54), nb_classes=2, loss_func='binary_crossentropy',
           filter_n1s=[16,32], filter_size1s=[8,10,12,14], filter_stride1=1, border_mode1='valid', pool_size1s=[2,], filter_drop1=0.5,
           filter_n2s=[16,32], filter_size2s=[3,4], filter_stride2=1, border_mode2='same', pool_size2=2, filter_drop2=0.5,
           dense_n1=256, dense_drop1=0.5, dense_n2=64, dense_drop2=0.5, batch_sizes=[64,128], nb_epoch=50, norm_x=10.,
           lr=0.01, early_stop=10, log_file="search_gamma_cnn.log"):

    print("Grid searching the parameter space of a ConvNet model...")
    scores = {}
    if log_file is not None:
        flog = open(log_file, 'w+')

    if len(train_y.shape)==1:
        #add a new axis to y
        train_y = np_utils.to_categorical(train_y, nb_classes)

    if len(test_y.shape)==1:
        #add a new axis to y
        test_y = np_utils.to_categorical(test_y, nb_classes)

    if train_x.dtype != 'float32':
        train_x = train_x.astype('float32')
    if train_y.dtype != 'float32':
        train_y = train_y.astype('float32')
    if test_x.dtype != 'float32':
        test_x = test_x.astype('float32')
    if test_y.dtype != 'float32':
        test_y = test_y.astype('float32')

    #normalize x only once:
    train_x = train_x / norm_x
    test_x = test_x / norm_x

    for filter_n1 in filter_n1s:
        for filter_size1 in filter_size1s:
            for pool_size1 in pool_size1s:
                for filter_n2 in filter_n2s:
                    for filter_size2 in filter_size2s:
                        for batch_size in batch_sizes:
                            weights_file= "sim+CR_filter1n%ds%dp%d_filter2n%ds%dp%d_batch%d_norm%.1f_best_weights.hdf5" % (filter_n1, filter_size1, pool_size1, filter_n2, filter_size2, pool_size2, batch_size, norm_x)
                            model, m_history = do_cnn(train_x, train_y, test_x, test_y, input_shape=input_shape, nb_classes=nb_classes, loss_func=loss_func,
                                   filter_n1=filter_n1, filter_size1=filter_size1, filter_stride1=filter_stride1, border_mode1=border_mode1,
                                   pool_size1=pool_size1, filter_drop1=filter_drop1,
                                   filter_n2=filter_n2, filter_size2=filter_size2, filter_stride2=filter_stride2, border_mode2=border_mode2,
                                   pool_size2=pool_size2, filter_drop2=filter_drop2,
                                   dense_n1=dense_n1, dense_drop1=dense_drop1, dense_n2=dense_n2, dense_drop2=dense_drop2,
                                   batch_size=batch_size, nb_epoch=nb_epoch, norm_x=1.,
                                   lr=lr, early_stop=early_stop, weights_file= weights_file)
                            score = model.evaluate(test_x, test_y, batch_size=batch_size, show_accuracy=True, verbose=1)
                            scores["sim+CR_filter1n%ds%dp%d_filter2n%ds%dp%d_batch%d_norm%.1f" % \
                                   (filter_n1, filter_size1, pool_size1, filter_n2, filter_size2, pool_size2, batch_size, norm_x)] = score
                            #print score
                            if log_file is not None:
                                flog.write("sim+CR_filter1n%ds%dp%d_filter2n%ds%dp%d_batch%d_norm%.1f score = %.4f accuracy = %.4f" % \
                                   (filter_n1, filter_size1, pool_size1, filter_n2, filter_size2, pool_size2, batch_size, norm_x, score[0], score[1]))
    if log_file is not None:
        flog.close()
    print(scores)
    return scores

def view_layer(model, layer_num, data_entry, subplot_rows=4, subplot_cols=8, figsize=(12,6), cmap=plt.cm.CMRmap, filename=None):
    get_layer_output = K.function([model.layers[0].input],[model.layers[layer_num].get_output(train=False)])
    layer_output = get_layer_output([data_entry])[0]
    print layer_output.shape

    fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize=figsize)
    for i in range(subplot_rows*subplot_cols):
        ax.flatten()[i].pcolor(layer_output[0,i].T, cmap=cmap)
        ax.flatten()[i].set_xlim(0, layer_output[0,i].shape[0])
        ax.flatten()[i].set_ylim(0, layer_output[0,i].shape[0])
        ax.flatten()[i].set_xticks([])
        ax.flatten()[i].set_yticks([])

    plt.subplots_adjust(hspace = .0)
    plt.subplots_adjust(wspace = .0)
    #plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename,bbox_inches='tight')


def read_data_from_pickle(fs=["64080_raw_trainx.pkl", "64080_raw_trainy.pkl", "64080_raw_testx.pkl", "64080_raw_testy.pkl"]):
    train_x = load_pickle(fs[0])
    train_y = load_pickle(fs[1])
    test_x =  load_pickle(fs[2])
    test_y =  load_pickle(fs[3])
    return train_x, train_y, test_x, test_y

def read_all_data_from_pickle(runs=[64080, 64081, 64082, 64083]):
    train_x, train_y, test_x, test_y = read_data_from_pickle(fs=[str(runs[0])+"_raw_trainx.pkl", str(runs[0])+"_raw_trainy.pkl",
                                                                     str(runs[0])+"_raw_testx.pkl", str(runs[0])+"_raw_testy.pkl"])
    for run in runs[1:]:
        train_x_, train_y_, test_x_, test_y_  = read_data_from_pickle(fs=[str(run)+"_raw_trainx.pkl", str(run)+"_raw_trainy.pkl",
                                                                     str(run)+"_raw_testx.pkl", str(run)+"_raw_testy.pkl"])
        train_x = np.concatenate((train_x, train_x_), axis=0)
        train_y = np.concatenate((train_y, train_y_), axis=0)
        test_x = np.concatenate((test_x, test_x_), axis=0)
        test_y = np.concatenate((test_y, test_y_), axis=0)
    return train_x, train_y, test_x, test_y

def concat_data(train_x1, train_y1, test_x1, test_y1, train_x2, train_y2, test_x2, test_y2):
    train_x = np.concatenate((train_x1, train_x2), axis=0)
    train_y = np.concatenate((train_y1, train_y2), axis=0)
    test_x = np.concatenate((test_x1, test_x2), axis=0)
    test_y = np.concatenate((test_y1, test_y2), axis=0)
    return train_x, train_y, test_x, test_y

def split_train_test(x, y, ratio=0.2, random_state=1234):
    sss = StratifiedShuffleSplit(y, test_size=ratio, random_state=random_state)
    for train_index, test_index in sss:
        break
    train_x, train_y = x[train_index], y[train_index]
    test_x, test_y = x[test_index], y[test_index]
    return train_x.astype('float32'), train_y.astype('float32'), test_x.astype('float32'), test_y.astype('float32')


# save model:
def save_keras_model(model, filename, weight_file):
    """
    :param model:
    :param filename: e.g. 'model_sim5_moreData_architecture.json'
    :param weight_file: e.g. 'model_sim5_moreData_weights.h5'
    :return:
    """
    json_string = model.to_json()
    open(filename, 'w').write(json_string)
    model.save_weights(weight_file)


# load model:
def load_keras_model(filename, weight_file):
    """
    :param filename: see above
    :param weight_file: see above
    :return: keras model obj
    """
    model = model_from_json(open(filename).read())
    model.load_weights(weight_file)
    return model


def predict_stats(model, test_x, norm=255.):
    pred_ = model.predict_proba(test_x/norm)
    print "The mean predictions are", np.mean(pred_, axis=0)
    print "The std dev of the predictions are", np.std(pred_, axis=0)

def clean_negative(x, y=None):
    x_positve = x[np.where(np.mean(x, axis=(1,2,3))>0)].astype('float32')
    x_positve[np.where(x_positve<0)] = 0
    if y is not None:
        assert x.shape[0] == y.shape[0], "Please provide the same number of x and y entries."
        y_positve = y[np.where(np.mean(x, axis=(1,2,3))>0)].astype('float32')
        return x_positve, y_positve
    return x_positve

