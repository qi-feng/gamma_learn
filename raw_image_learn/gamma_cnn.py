__author__ = 'qfeng'


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
#from keras.layers.normalization import BatchNormalization
#from keras.layers.advanced_activations import PReLU
#from keras.utils import np_utils, generic_utils
from sklearn.metrics import roc_auc_score, roc_curve, auc
from get_raw_features import *

def do_cnn(train_x, train_y, test_x, test_y, input_shape=(4, 54, 54), filter_n1=64, filter_size1=6,
           filter_n2=64, filter_size2=6, pool_size1=3, pool_size2=2, filter_drop1=0.25, filter_drop2=0.25,
           dense_n1=512, dense_drop1=0.5, dense_n2=256, dense_drop2=0.5, batch_size=128, nb_epoch=5):
    print("Building a ConvNet model...")
    model = Sequential()

    # input: 54x54 images with 4 tels -> (4, 54, 54) tensors.
    # this applies 64 convolution filters of size 6x6 each.
    model.add(Convolution2D(filter_n1, filter_size1, filter_size1, border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(filter_n1, filter_size1, filter_size1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size1, pool_size1)))
    model.add(Dropout(filter_drop1))

    model.add(Convolution2D(filter_n2, filter_size2, filter_size2, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(filter_n2, filter_size2, filter_size2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size2, pool_size2)))
    model.add(Dropout(filter_drop2))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(dense_n1))
    model.add(Activation('relu'))
    model.add(Dropout(dense_drop1))

    model.add(Dense(dense_n2))
    model.add(Activation('relu'))
    model.add(Dropout(dense_drop2))

    model.add(Dense(1))
    #model.add(Activation('softmax'))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    #model.compile(loss='binary_crossentropy', optimizer="rmsprop")

    print("Training the ConvNet model...")
    m_history = model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=nb_epoch)

    objective_score = model.evaluate(test_x, test_y, batch_size=batch_size)
    print "The objective score on test data is", objective_score


    predict_train_y = model.predict(train_x)
    fpr, tpr, thresh = roc_curve(train_y,predict_train_y)
    roc_auc = auc(fpr, tpr)
    predict_test_y = model.predict(test_x)
    fpr_test, tpr_test, thresh_test = roc_curve(test_y,predict_test_y)
    roc_auc_test = auc(fpr_test, tpr_test)
    print 'The training AUC score is {0}, and the test AUC score is: {1}'.format(roc_auc, roc_auc_test)

    return model, m_history

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

