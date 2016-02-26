from gamma_xgb import *

from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation 
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
import numexpr as ne
import cPickle as pickle
import sys
sys.setrecursionlimit(5000)

def dump_model(model00, f_outname):
    f_out = file(f_outname, 'wb')
    pickle.dump(model00, f_out, protocol=pickle.HIGHEST_PROTOCOL)
    f_out.close()

def read_data_keras(filename="BDT_2_0_V6.txt", fit_transform='linear', test_size=0.2):
    x_ft,y_ft,scale_ft = read_data(filename=filename, fit_transform=fit_transform)
    if test_size<=0:
        return x_ft,y_ft
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

def do_keras(train_x, train_y, test_x, test_y, layers=[Dense, Dense, Dense], layer_dims=[512, 512, 1024], activations=['relu', 'relu', 'relu'], dropouts=[0.5, 0.5, 0.5], loss='binary_crossentropy', optimizer="rmsprop", out_activation='sigmoid', init='glorot_uniform', nb_epoch=20, batch_size=128, valid_ratio=0.15, dump=False, overwrite=False, modelname=None, conv=True):
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

def search_keras(filename="BDT_2_0_V6.txt", fit_transform='linear', test_size=0.2, dump=False, overwrite=False, 
        l_list=[[16, 32, 64], [16, 64, 128], [32, 128, 256], [32, 128, 512], [64, 256, 512]],
        d_list=[[0.1, 0.2, 0.5]], 
        e_list=[10], 
        b_list=[128, 256]):
    scores=[]
    aucs=[]
    params=[]
    print("Reading data from file %s ..." % filename)
    train_x, train_y, test_x, test_y = read_data_keras(filename=filename, fit_transform=fit_transform, test_size=test_size)
    #for layer_d, drop_, nb_epoch_, batch_ in zip(l_list, d_list, e_list, b_list):
    for layer_d in l_list: 
      for drop_ in d_list:
        for nb_epoch_ in e_list:
          for batch_ in b_list:
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

def plot_keras_pseudo_TMVA(model_file="keras_2_0_V6_3layers_64_256_512_dropouts_0.1_0.2_0.5_epoch10batch128.pkl", train_file="BDT_2_0_V6.txt", test_file="BDT_2_0_Test_V6.txt",
        ifKDE=False, outfile='BDT_2_0_keras', nbins=40, plot_roc=True, plot_tmva_roc=True, norm_hist=True, thresh_IsGamma=0.95):

    f_in = file(model_file, 'rb')
    clf = pickle.load(f_in)
    f_in.close()
    train_x, train_y = read_data_keras(filename=train_file, fit_transform='linear', test_size=0)
    test_x, test_y = read_data_keras(filename=test_file, fit_transform='linear', test_size=0)

    predict_train_y = clf.predict(train_x)
    predict_test_y = clf.predict(test_x)
    # Compute ROC curve and ROC area
    # roc_auc = dict()
    fpr, tpr, thresh = roc_curve(train_y,predict_train_y)
    roc_auc = auc(fpr, tpr)
    fpr_test, tpr_test, thresh_test = roc_curve(test_y,predict_test_y)
    roc_auc_test = auc(fpr_test, tpr_test)
    print 'The training AUC score is {0}, and the test AUC score is: {1}'.format(
            roc_auc, roc_auc_test)
    diff_tpr_fpr=tpr_test-fpr_test
    thresh_index_fpr = np.argmin(fpr_test<=(1-thresh_IsGamma))
    thresh_index_tpr = np.argmax(tpr_test>=thresh_IsGamma)
    thresh_index2 = np.where(diff_tpr_fpr==np.max(diff_tpr_fpr))
    print "Note that below TMVA threshold [-1, 1] is used instead of probability"
    print "However, note that thresh_IsGamma should be given in probability"
    print "Threshold tpr>="+str(thresh_IsGamma)+" is "+str(thresh_test[thresh_index_tpr]*2-1)
    print "Threshold fpr<="+str(1-thresh_IsGamma)+" is "+str(thresh_test[thresh_index_fpr]*2-1)
    print "Threshold index found ", thresh_index2
    for ind_ in thresh_index2:
        for ind in ind_:
            print "TMVA Threshold max diff", thresh_test[ind]*2-1
            thresh_maxdiff = thresh_test[ind]*2-1
    plt.figure()
    sns.distplot(predict_train_y[np.where(train_y==1)]*2.-1.,
                 bins=nbins, hist=True, kde=ifKDE, rug=False,
                 hist_kws={"histtype": "step", "linewidth": 1,
                           "alpha": 1, "color": "darkblue"},
                 label="Training Signal", norm_hist=norm_hist)
    sns.distplot(predict_train_y[np.where(train_y==0)]*2.-1.,
                 bins=nbins, hist=True, kde=ifKDE, rug=False,
                 hist_kws={"histtype": "step", "linewidth": 1,
                           "alpha": 1, "color": "darkred"},
                 label="Training Background", norm_hist=norm_hist)
    sns.distplot(predict_test_y[np.where(test_y==1)]*2.-1.,
                 color='b', bins=nbins, hist=True, kde=ifKDE, rug=False,
                 label="Test Signal", norm_hist=norm_hist)
    sns.distplot(predict_test_y[np.where(test_y==0)]*2.-1.,
                 color='r', bins=nbins, hist=True, kde=ifKDE, rug=False,
                 label="Test Background", norm_hist=norm_hist)
    for ind_ in thresh_index2:
        for ind in ind_:
            plt.axvline(thresh_test[ind]*2-1,color='orange', label='thresh diff_tpr_fpr_test '+str(thresh_test[ind]*2-1))
    plt.axvline(thresh_test[thresh_index_tpr]*2-1,color='green', label='thresh tpr_test '+str(thresh_test[thresh_index_tpr]*2-1))
    plt.axvline(thresh_test[thresh_index_fpr]*2-1,color='magenta', label='thresh fpr_test '+str(thresh_test[thresh_index_fpr]*2-1))
    plt.legend(loc='best');
    if norm_hist:
        sns.axlabel("Pseudo TMVA value","Normalized event counts")
        plt.savefig(outfile+'normed_counts.png', format='png', dpi=500)
    else:
        sns.axlabel("Pseudo TMVA value","Event counts")
        plt.savefig(outfile+'counts.png', format='png', dpi=500)
    if(plot_roc==True):
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='training ROC curve (area = %0.2f)' % roc_auc)
        plt.plot(fpr_test, tpr_test, label='testing ROC curve (area = %0.2f)' % roc_auc_test)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        #plt.savefig(os.path.splitext(outfile)[0]+'_ROC.png', format='png', dpi=1000)
        plt.savefig(outfile+'_ROC.png', format='png', dpi=500)

    if(plot_tmva_roc==True):
        # Plot TMVA ROC curve
        plt.figure(figsize=(6, 6))
        for ind_ in thresh_index2:
            for ind in ind_:
                plt.axvline(thresh_test[ind]*2-1,color='orange', label='thresh diff_tpr_fpr_test '+str(thresh_test[ind]*2-1))
        #thresh_test[thresh_index]*2-1
        np.max(tpr_test-fpr_test)
        plt.plot(thresh*2-1, tpr, 'r--', label='training true positive')
        plt.plot(thresh*2-1, fpr, 'b--', label='training false positive')
        plt.plot(thresh_test*2-1, tpr_test, 'r-', label='test true positive')
        plt.plot(thresh_test*2-1, fpr_test, 'b-', label='test false positive')
        #plt.axvline(thresh_test[thresh_index]*2-1,color='g', label='thresh ratio_tpr_fpr')
        #plt.axvline(thresh_test[thresh_index2]*2-1,color='orange', label='thresh diff_tpr_fpr_test '+str(thresh_test[thresh_index2]*2-1))
        plt.axvline(thresh_test[thresh_index_tpr]*2-1,color='green', label='thresh tpr_test '+str(thresh_test[thresh_index_tpr]*2-1))
        plt.axvline(thresh_test[thresh_index_fpr]*2-1,color='magenta', label='thresh fpr_test '+str(thresh_test[thresh_index_fpr]*2-1))
        print str(model_file)+' has a thresh diff_tpr_fpr_test '+str(thresh_test[thresh_index2[0]]*2-1)
        plt.xlim([-1.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Threshold')
        plt.ylabel('True/False Positive Rate')
        plt.title('ROC curve in the TMVA way')
        plt.legend(loc="best")
        plt.savefig(outfile+'_efficiency.png', format='png', dpi=500)

    return thresh_test[thresh_index_tpr]*2-1, thresh_test[thresh_index_fpr]*2-1, thresh_maxdiff

def pred_4_models(E=2, Z=0, weights=None):
    model_x_file="BDT"+str(E)+str(Z)+".model"
    model_k_file="keras_"+str(E)+"_"+str(Z)+"_V6_3layers_64_256_512_dropouts_0.1_0.2_0.5_epoch10batch128.pkl"
    model_xLT_file="BDT"+str(E)+str(Z)+"LT.model"
    model_xLT2_file="BDT"+str(E)+str(Z)+"LT2.model"
    model_rf_file="RF"+str(E)+str(Z)+".pkl"
    train_file="BDT_"+str(E)+"_"+str(Z)+"_V6.txt"
    test_file="BDT_"+str(E)+"_"+str(Z)+"_Test_V6.txt"
    outfile="BDT_"+str(E)+"_"+str(Z)+"_combined"

    print "Predicting xgb non trans"
    clfx = xgb.Booster() #init model
    clfx.load_model(model_x_file) # load data
    trainx_x, trainx_y =read_data_xgb(train_file, predict=True)
    testx_x, testx_y =read_data_xgb(test_file, predict=True)
    predict_trainx_y = clfx.predict(trainx_x)
    predict_testx_y = clfx.predict(testx_x)
    del trainx_x

    print "Predicting xgb linear trans 1"
    clfx1 = xgb.Booster() #init model
    clfx1.load_model(model_xLT_file) # load data
    trainx1_x, trainx1_y =read_data_xgb(train_file, fit_transform='linear', predict=True)
    testx1_x, testx1_y =read_data_xgb(test_file, fit_transform='linear', predict=True)
    predict_trainx1_y = clfx1.predict(trainx1_x)
    predict_testx1_y = clfx1.predict(testx1_x)

    print "Predicting xgb linear trans 2"
    clfx2 = xgb.Booster() #init model
    clfx2.load_model(model_xLT2_file) # load data
    #trainx2_x, trainx2_y = trainx1_x, trainx1_y
    #testx2_x, testx2_y = testx1_x, testx1_y
    predict_trainx2_y = clfx2.predict(trainx1_x)
    predict_testx2_y = clfx2.predict(testx1_x)  
    del trainx1_x

    print "Predicting keras"
    f_in = file(model_k_file, 'rb')
    clfk = pickle.load(f_in)
    f_in.close()
    traink_x, traink_y = read_data_keras(filename=train_file, fit_transform='linear', test_size=0)
    testk_x, testk_y = read_data_keras(filename=test_file, fit_transform='linear', test_size=0)
    
    np.testing.assert_array_equal(trainx_y, traink_y, err_msg='Training data read differently for xgb and keras')
    np.testing.assert_array_equal(testx_y, testk_y, err_msg='Test data read differently for xgb and keras')
    predict_traink_y = clfk.predict(traink_x).T[0]
    predict_testk_y = clfk.predict(testk_x).T[0]

    print "Predicting rf"
    f_in = file(model_rf_file, 'rb')
    clf_rf_isotonic = pickle.load(f_in)
    f_in.close()

    predict_trainrf_y = clf_rf_isotonic.predict_proba(traink_x)[:, 1]
    predict_testrf_y = clf_rf_isotonic.predict_proba(testk_x)[:, 1]

    del traink_x

    # Compute ROC curve and ROC area
    #fprx, tprx, threshx = roc_curve(trainx_y,predict_trainx_y)
    #roc_aucx = auc(fprx, tprx)
    #fprx_test, tprx_test, threshx_test = roc_curve(testx_y,predict_testx_y)
    #roc_aucx_test = auc(fprx_test, tprx_test)
    #print 'The XGB training AUC score is {0}, and the test AUC score is: {1}'.format(
    #        roc_aucx, roc_aucx_test)
    #fprk, tprk, threshk = roc_curve(traink_y,predict_traink_y)
    #roc_auck = auc(fprk, tprk)
    #fprk_test, tprk_test, threshk_test = roc_curve(testk_y,predict_testk_y)
    #roc_auck_test = auc(fprk_test, tprk_test)
    #print 'The Keras training AUC score is {0}, and the test AUC score is: {1}'.format(   
    #        roc_auck, roc_auck_test)

    for pred_, y_, pred_test_, y_test_, name_ in zip([predict_trainx_y, predict_trainx1_y, predict_trainx2_y, predict_traink_y, predict_trainrf_y], [trainx_y, trainx1_y, trainx1_y, traink_y, traink_y], [predict_testx_y, predict_testx1_y, predict_testx2_y, predict_testk_y, predict_testrf_y], [testx_y, testx1_y, testx1_y, testk_y, testk_y], ['xgb', 'xgbLT1', 'xgbLT2', 'keras', 'rf']):
        fpr_, tpr_, thresh_ = roc_curve(y_, pred_)
        roc_auc_ = auc(fpr_, tpr_)
        fpr_test_, tpr_test_, thresh_test_ = roc_curve(y_test_, pred_test_)
        roc_auc_test_ = auc(fpr_test_, tpr_test_)
        print "pred train: ", pred_[:5]
        print 'The model '+str(name_)+' training AUC score is {0}, and the test AUC score is: {1}'.format(roc_auc_, roc_auc_test_)

    #print "Now calc aucs"
    #for pred_, y_, name_ in zip([predict_trainx_y, predict_trainx1_y, predict_trainx2_y, predict_traink_y, predict_trainrf_y], [trainx_y, trainx1_y, trainx1_y, traink_y, traink_y],  ['xgb', 'xgbLT1', 'xgbLT2', 'keras', 'rf']):
    #    fpr_, tpr_, thresh_ = roc_curve(y_, pred_)
    #    roc_auc_ = auc(fpr_, tpr_)
    #    print 'The model '+str(name_)+' training AUC score is {0}'.format(roc_auc_,)

    #if weights==None:
    #    try:
    #        weights=np.array([(roc_aucx-0.5)/(roc_aucx+roc_auck-1.), (roc_auck-0.5)/(roc_aucx+roc_auck-1.)])
    #    except:
    #        print "error doing ensemble of the two models"
    #        raise
    #else:
        #weights=optimize_weights([predict_trainx_y, predict_traink_y], traink_y)
    weights=optimize_weights([predict_trainx_y, predict_trainx1_y, predict_trainx2_y, predict_traink_y, predict_trainrf_y], traink_y)
    wts_test=optimize_weights([predict_testx_y, predict_testx1_y, predict_testx2_y, predict_testk_y, predict_testrf_y], testk_y)

    #assert len(weights)==5, "Weights provided are longer than 5"
    assert np.sum(weights)-1.<1e-3;
    predict_train_y = predict_trainx_y*weights[0] + predict_trainx1_y*weights[1]+ predict_trainx2_y*weights[2] + predict_traink_y*weights[3] + predict_trainrf_y*weights[4]
    predict_test_y = predict_testx_y*weights[0]+ predict_testx1_y*weights[1]+ predict_testx2_y*weights[2]+ predict_testk_y*weights[3]+ predict_testrf_y*weights[4]

    print "test weights:", wts_test
    fpr, tpr, thresh = roc_curve(trainx_y,predict_train_y)
    roc_auc = auc(fpr, tpr)
    fpr_test, tpr_test, thresh_test = roc_curve(testx_y,predict_test_y)
    roc_auc_test = auc(fpr_test, tpr_test)
    print 'The combined training AUC score is {0}, and the test AUC score is: {1}'.format(roc_auc, roc_auc_test)
    #print 'The combined training AUC score is {0}'.format(roc_auc)

    return weights, wts_test

def plot_2models_pseudo_TMVA(model_x_file="BDT20.model", model_k_file="keras_2_0_V6_3layers_64_256_512_dropouts_0.1_0.2_0.5_epoch10batch128.pkl", train_file="BDT_2_0_V6.txt", test_file="BDT_2_0_Test_V6.txt",
        ifKDE=False, outfile='BDT_2_0_combined', nbins=40, plot_roc=True, plot_tmva_roc=True, norm_hist=True, thresh_IsGamma=0.95, weights=None):
    clfx = xgb.Booster() #init model
    clfx.load_model(model_x_file) # load data
    trainx_x, trainx_y =read_data_xgb(train_file, predict=True)
    testx_x, testx_y =read_data_xgb(test_file, predict=True)
    predict_trainx_y = clfx.predict(trainx_x)
    predict_testx_y = clfx.predict(testx_x)

    f_in = file(model_k_file, 'rb')
    clfk = pickle.load(f_in)
    f_in.close()
    traink_x, traink_y = read_data_keras(filename=train_file, fit_transform='linear', test_size=0)
    testk_x, testk_y = read_data_keras(filename=test_file, fit_transform='linear', test_size=0)
    np.testing.assert_array_equal(trainx_y, traink_y, err_msg='Training data read differently for xgb and keras')
    np.testing.assert_array_equal(testx_y, testk_y, err_msg='Test data read differently for xgb and keras')
    #assert trainx_y == traink_y, "Data read differently for xgb and keras"
    #assert testx_y == testk_y, "Test data read differently for xgb and keras"
    predict_traink_y = clfk.predict(traink_x).T[0]
    predict_testk_y = clfk.predict(testk_x).T[0]
    # Compute ROC curve and ROC area
    # roc_auc = dict()
    fprx, tprx, threshx = roc_curve(trainx_y,predict_trainx_y)
    roc_aucx = auc(fprx, tprx)
    fprx_test, tprx_test, threshx_test = roc_curve(testx_y,predict_testx_y)
    roc_aucx_test = auc(fprx_test, tprx_test)
    print 'The XGB training AUC score is {0}, and the test AUC score is: {1}'.format(
            roc_aucx, roc_aucx_test)
    fprk, tprk, threshk = roc_curve(traink_y,predict_traink_y)
    roc_auck = auc(fprk, tprk)
    fprk_test, tprk_test, threshk_test = roc_curve(testk_y,predict_testk_y)
    roc_auck_test = auc(fprk_test, tprk_test)
    print 'The Keras training AUC score is {0}, and the test AUC score is: {1}'.format(   
            roc_auck, roc_auck_test)
    if weights==None:
    #    try:
    #        weights=np.array([(roc_aucx-0.5)/(roc_aucx+roc_auck-1.), (roc_auck-0.5)/(roc_aucx+roc_auck-1.)])
    #    except:
    #        print "error doing ensemble of the two models"
    #        raise
    #else:
        weights=optimize_weights([predict_trainx_y, predict_traink_y], traink_y)
    
    assert len(weights)==2, "Weights provided are longer than 2"
    print weights[0], weights[1]
    assert weights[0]+weights[1]-1.<1e-3;
    w0 = weights[0]
    w1 = weights[1]
    #predict_trainx_y = np.add(predict_trainx_y*weights[0], predict_traink_y*weights[1])
    #predict_testx_y = np.add(predict_testx_y*weights[0], predict_testk_y*weights[1])
    predict_trainx_y = ne.evaluate('predict_trainx_y*w0+predict_traink_y*w1')
    predict_testx_y = ne.evaluate('predict_testx_y*w0+predict_testk_y*w1')

    fpr, tpr, thresh = roc_curve(trainx_y,predict_trainx_y)
    roc_auc = auc(fpr, tpr)
    fpr_test, tpr_test, thresh_test = roc_curve(testx_y,predict_testx_y)
    roc_auc_test = auc(fpr_test, tpr_test)
    print 'The combined training AUC score is {0}, and the test AUC score is: {1}'.format(   
            roc_auc, roc_auc_test)

    diff_tpr_fpr=tpr_test-fpr_test
    thresh_index_fpr = np.argmin(fpr_test<=(1-thresh_IsGamma))
    thresh_index_tpr = np.argmax(tpr_test>=thresh_IsGamma)
    thresh_index2 = np.where(diff_tpr_fpr==np.max(diff_tpr_fpr))
    print "Note that below TMVA threshold [-1, 1] is used instead of probability"
    print "However, note that thresh_IsGamma should be given in probability"
    print "Threshold tpr>="+str(thresh_IsGamma)+" is "+str(thresh_test[thresh_index_tpr]*2-1)
    print "Threshold fpr<="+str(1-thresh_IsGamma)+" is "+str(thresh_test[thresh_index_fpr]*2-1)
    print "Threshold index found ", thresh_index2
    for ind_ in thresh_index2:
        for ind in ind_:
            print "TMVA Threshold max diff", thresh_test[ind]*2-1
            thresh_maxdiff = thresh_test[ind]*2-1
    plt.figure()
    sns.distplot(predict_trainx_y[np.where(trainx_y==1)]*2.-1.,
                 bins=nbins, hist=True, kde=ifKDE, rug=False,
                 hist_kws={"histtype": "step", "linewidth": 1,
                           "alpha": 1, "color": "darkblue"},
                 label="Training Signal", norm_hist=norm_hist)
    sns.distplot(predict_trainx_y[np.where(trainx_y==0)]*2.-1.,
                 bins=nbins, hist=True, kde=ifKDE, rug=False,
                 hist_kws={"histtype": "step", "linewidth": 1,
                           "alpha": 1, "color": "darkred"},
                 label="Training Background", norm_hist=norm_hist)
    sns.distplot(predict_testx_y[np.where(testx_y==1)]*2.-1.,
                 color='b', bins=nbins, hist=True, kde=ifKDE, rug=False,
                 label="Test Signal", norm_hist=norm_hist)
    sns.distplot(predict_testx_y[np.where(testx_y==0)]*2.-1.,
                 color='r', bins=nbins, hist=True, kde=ifKDE, rug=False,
                 label="Test Background", norm_hist=norm_hist)
    for ind_ in thresh_index2:
        for ind in ind_:
            plt.axvline(thresh_test[ind]*2-1,color='orange', label='thresh diff_tpr_fpr_test '+str(thresh_test[ind]*2-1))
    plt.axvline(thresh_test[thresh_index_tpr]*2-1,color='green', label='thresh tpr_test '+str(thresh_test[thresh_index_tpr]*2-1))
    plt.axvline(thresh_test[thresh_index_fpr]*2-1,color='magenta', label='thresh fpr_test '+str(thresh_test[thresh_index_fpr]*2-1))
    plt.legend(loc='best');
    if norm_hist:
        sns.axlabel("Pseudo TMVA value","Normalized event counts")
        plt.savefig(outfile+'normed_counts.png', format='png', dpi=500)
    else:
        sns.axlabel("Pseudo TMVA value","Event counts")
        plt.savefig(outfile+'counts.png', format='png', dpi=500)
    if(plot_roc==True):
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='training ROC curve (area = %0.2f)' % roc_auc)
        plt.plot(fpr_test, tpr_test, label='testing ROC curve (area = %0.2f)' % roc_auc_test)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        #plt.savefig(os.path.splitext(outfile)[0]+'_ROC.png', format='png', dpi=1000)
        plt.savefig(outfile+'_ROC.png', format='png', dpi=500)

    if(plot_tmva_roc==True):
        # Plot TMVA ROC curve
        plt.figure(figsize=(6, 6))
        for ind_ in thresh_index2:
            for ind in ind_:
                plt.axvline(thresh_test[ind]*2-1,color='orange', label='thresh diff_tpr_fpr_test '+str(thresh_test[ind]*2-1))
        #thresh_test[thresh_index]*2-1
        np.max(tpr_test-fpr_test)
        plt.plot(thresh*2-1, tpr, 'r--', label='training true positive')
        plt.plot(thresh*2-1, fpr, 'b--', label='training false positive')
        plt.plot(thresh_test*2-1, tpr_test, 'r-', label='test true positive')
        plt.plot(thresh_test*2-1, fpr_test, 'b-', label='test false positive')
        #plt.axvline(thresh_test[thresh_index]*2-1,color='g', label='thresh ratio_tpr_fpr')
        #plt.axvline(thresh_test[thresh_index2]*2-1,color='orange', label='thresh diff_tpr_fpr_test '+str(thresh_test[thresh_index2]*2-1))
        plt.axvline(thresh_test[thresh_index_tpr]*2-1,color='green', label='thresh tpr_test '+str(thresh_test[thresh_index_tpr]*2-1))
        plt.axvline(thresh_test[thresh_index_fpr]*2-1,color='magenta', label='thresh fpr_test '+str(thresh_test[thresh_index_fpr]*2-1))
        print 'Combined model has a thresh diff_tpr_fpr_test '+str(thresh_test[thresh_index2[0]]*2-1)
        plt.xlim([-1.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Threshold')
        plt.ylabel('True/False Positive Rate')
        plt.title('ROC curve in the TMVA way')
        plt.legend(loc="best")
        plt.savefig(outfile+'_efficiency.png', format='png', dpi=500)

    return thresh_test[thresh_index_tpr]*2-1, thresh_test[thresh_index_fpr]*2-1, thresh_maxdiff

### finding the optimum weights

def log_loss_func(weights, predictions=[], y=np.ones(1)):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction
    return log_loss(y, final_prediction)

def auc_func(weights, predictions=[], y=np.ones(1)):
    ''' scipy minimize will pass the weights as a numpy array 
        note the negative sign for minimize
    '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction
    fpr, tpr, thresh = roc_curve(y,final_prediction)
    return -auc(fpr, tpr)


def optimize_weights(predictions, y):
    #predictions = []
    #for pred_ in preds:
    #    predictions.append(pred_)
    
    #the algorithms need a starting value, right not we chose 0.5 for all weights
    #its better to choose many random starting points and run minimize a few times
    starting_values = [1.0/len(predictions)]*len(predictions)
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    #our weights are bound between 0 and 1
    bounds = [(0,1)]*len(predictions)
    
    #res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)
    res = minimize(auc_func, starting_values, args=(predictions, y,), method='SLSQP', bounds=bounds, constraints=cons)
    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))
    return res['x']


