from gamma_xgb import *

from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation 
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

import cPickle as pickle
import sys
sys.setrecursionlimit(5000)

def dump_model(model00, f_outname):
    f_out = file(f_outname, 'wb')
    pickle.dump(model00, f_out, protocol=pickle.HIGHEST_PROTOCOL)
    f_out.close()

def read_data_keras(filename="BDT_2_0_V6.txt", fit_transform='linear', test_size=0.2):
    x_ft,y_ft,scale_ft = read_data(filename=filename, fit_transform=fit_transform)
    sss_ft = StratifiedShuffleSplit(y_ft, test_size=test_size)
    for train_index, test_index in sss_ft:
        break

    train_x_ft, train_y_ft = x_ft[train_index], y_ft[train_index]
    test_x_ft, test_y_ft = x_ft[test_index], y_ft[test_index]

    return train_x_ft, train_y_ft, test_x_ft, test_y_ft

def do_keras_file(filename="BDT_2_0_V6.txt", fit_transform='linear', test_size=0.2, layers=[Dense, Dense, Dense], layer_dims=[512, 512, 1024], activations=['relu', 'relu', 'relu'], dropouts=[0.5, 0.5, 0.5], loss='binary_crossentropy', optimizer="rmsprop", out_activation='sigmoid', init='glorot_uniform', nb_epoch=20, batch_size=128, valid_ratio=0.15,dump=False, overwrite=False):
    print("Reading data from file %s ..." % filename)
    train_x, train_y, test_x, test_y = read_data_keras(filename=filename, fit_transform=fit_transform, test_size=test_size)
    modelname='keras_'+filename[4:7]+'_V6_3layers_'+str(layer_dims[0])+'_'+str(layer_dims[1])+'_'+str(layer_dims[2])+'_dropouts_'+str(dropouts[0])+'_'+str(dropouts[1])+'_'+str(dropouts[2])+'_epoch'+str(nb_epoch)+'batch'+str(batch_size)+'.pkl'
    return do_keras(train_x, train_y, test_x, test_y, layers=layers, layer_dims=layer_dims, activations=activations, dropouts=dropouts, loss=loss, optimizer=optimizer, out_activation=out_activation, init=init, nb_epoch=nb_epoch, batch_size=batch_size, valid_ratio=valid_ratio, dump=dump, overwrite=overwrite, modelname=modelname)

def do_keras(train_x, train_y, test_x, test_y, layers=[Dense, Dense, Dense], layer_dims=[512, 512, 1024], activations=['relu', 'relu', 'relu'], dropouts=[0.5, 0.5, 0.5], loss='binary_crossentropy', optimizer="rmsprop", out_activation='sigmoid', init='glorot_uniform', nb_epoch=20, batch_size=128, valid_ratio=0.15, dump=False, overwrite=False, modelname=None):
    dims = train_x.shape[1]
    n_classes = 1
    input_dim = dims
    if os.path.exists(modelname) and not overwrite:
        print('%s already present - load using pickle.' % modelname)
        f_in = file(modelname, 'rb')
        model = pickle.load(f_in)
        f_in.close()
    else:
        print('%s doesn\'t exist - building model...' % modelname)
        model = Sequential()
        for layer_, layer_dim_, activation_, dropouts_ in zip(layers, layer_dims, activations, dropouts):
            model.add(layer_(output_dim=layer_dim_, input_dim=input_dim, init=init, activation=activation_))
            model.add(Dropout(dropouts_))
            model.add(BatchNormalization((layer_dim_,)))
            input_dim = layer_dim_
        model.add(Dense(output_dim=1, input_dim=input_dim, activation=out_activation))
        model.compile(loss=loss, optimizer=optimizer)
        print("Training model...")
        model.fit(train_x, train_y, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=valid_ratio)
        print("Saving model to file %s " % modelname)
        f_out = file(modelname, 'wb')
        pickle.dump(model, f_out, protocol=pickle.HIGHEST_PROTOCOL)
        f_out.close()

    objective_score = model.evaluate(test_x, test_y, batch_size=batch_size)
    print("Objective_score is: %0.3f" % objective_score)
    
    predict_train_y = model.predict(train_x)
    fpr, tpr, thresh = roc_curve(train_y,predict_train_y)
    roc_auc = auc(fpr, tpr)
    predict_test_y = model.predict(test_x)
    fpr_test, tpr_test, thresh_test = roc_curve(test_y,predict_test_y)
    roc_auc_test = auc(fpr_test, tpr_test)
    print 'The training AUC score is {0}, and the test AUC score is: {1}'.format(roc_auc, roc_auc_test)
    return model, objective_score, roc_auc_test

def search_keras(filename="BDT_2_0_V6.txt", fit_transform='linear', test_size=0.2, dump=False, overwrite=False):
    l_list=[[512, 512, 1024], [512, 1024, 2048], [1024, 2048, 2048]]
    d_list=[[0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [0.4, 0.4, 0.4]]
    e_list=[20, 40]
    b_list=[64, 128]
    scores=[]
    aucs=[]
    params=[]
    print("Reading data from file %s ..." % filename)
    train_x, train_y, test_x, test_y = read_data_keras(filename=filename, fit_transform=fit_transform, test_size=test_size)
    for layer_d, drop_, nb_epoch_, batch_ in zip(l_list, d_list, e_list, b_list):
        modelname='keras_'+filename[4:7]+'_V6_3layers_'+str(layer_d[0])+'_'+str(layer_d[1])+'_'+str(layer_d[2])+'_dropouts_'+str(drop_[0])+'_'+str(drop_[1])+'_'+str(drop_[2])+'_epoch'+str(nb_epoch_)+'batch'+str(batch_)+'.pkl'
        model, obj_score, auc_score = do_keras(train_x, train_y, test_x, test_y, layers=[Dense, Dense, Dense], layer_dims=layer_d, activations=['relu', 'relu', 'relu'], dropouts=drop_, loss='binary_crossentropy', optimizer="rmsprop", out_activation='sigmoid', init='glorot_uniform', nb_epoch=nb_epoch_, batch_size=batch_, valid_ratio=0.15, dump=dump, overwrite=overwrite, modelname=modelname)
        params.append([layer_d, drop_, nb_epoch_, batch_])
        scores.append(obj_score)
        aucs.append(auc_score)
        print "model trained: "
        print "params: ", [layer_d, drop_, nb_epoch_, batch_]
        print "objective score: ", obj_score
        print "auc score: ", auc_score
    print params, scores, aucs
    return params, scores, aucs


