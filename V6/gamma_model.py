__author__ = 'qfeng'

import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.metrics import log_loss
from scipy.optimize import minimize
import os

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import identity
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

from sklearn.metrics import roc_auc_score, roc_curve, auc
import seaborn as sns

import ROOT

class PyVData:
    ############################################################################
    #                       Initialization of the object.                      #
    ############################################################################
    def __init__(self, filename="Crab_V6_BDTModerate.RE.root", package='ED'):
        self.package=package
        self.E_grid=np.array([0.08, 0.32, 0.5, 1.0, 50.0])
        self.Zen_grid=np.array([0.0, 25.0, 32.5, 42.5, 75.])
        if filename:
            self.filename = filename
            self.readEDfile(filename=filename)
            #
    def readEDfile(self,filename="Crab_V6_BDTModerate.RE.root"):
        self.Rfile = ROOT.TFile(filename, "read");
    def get_run_on(self):
        self.runOn = []
        runSum = self.Rfile.Get("total_1/stereo/tRunSummary");
        for event in runSum:
            if event.runOn > 0:
                self.runOn.append(event.runOn)
        if not self.runOn:
            print "No runs found in data_on tree! "
            raise RuntimeError
    def get_data_on(self, runNum=0):
        if runNum==0:
            if not hasattr(self, 'runOn'):
                self.get_run_on()
            for run_ in self.runOn:
                self.get_data_on(runNum=run_)
        else:
            if not hasattr(self, 'Rfile'):
                print "No file has been read. Run self.readEDfile(filename=\"rootfile\") first. "
                raise
            data_on_treeName = "run_"+str(runNum)+"/stereo/data_on"
            pointingData_treeName = "run_"+str(runNum)+"/stereo/pointingDataReduced"
            data_on = self.Rfile.Get(data_on_treeName);
            pointingData = self.Rfile.Get(pointingData_treeName);
            ptTime=[]
            for ptd in pointingData:
                ptTime.append(ptd.Time);
            ptTime=np.array(ptTime)
            columns=['runNumber','eventNumber', 'MJD', 'Time', 'Elevation', 'theta2',
                     'MSCW','MSCL','log10_EChi2S_','EmissionHeight',
                     'log10_EmissionHeightChi2_','log10_SizeSecondMax_',
                     'sqrt_Xcore_T_Xcore_P_Ycore_T_Ycore_', 'NImages','Xoff','Yoff',
                     'ErecS', 'MVA', 'IsGamma']
            df_ = pd.DataFrame(np.array([np.zeros(data_on.GetEntries())]*len(columns)).T,
                               columns=columns)
            for i, event in enumerate(data_on):
                time_index=np.argmax(ptTime>event.Time)
                pointingData.GetEntry(time_index)
                # convert some quantities to BDT input
                log10_EChi2S_ = np.log10(event.EChi2S);
                log10_EmissionHeightChi2_ = np.log10(event.EmissionHeightChi2);
                log10_SizeSecondMax_ = np.log10(event.SizeSecondMax);
                sqrt_Xcore_T_Xcore_P_Ycore_T_Ycore_ = np.sqrt(event.Xcore*event.Xcore + event.Ycore*event.Ycore);
                # fill the pandas dataframe
                df_.runNumber[i] = event.runNumber
                df_.eventNumber[i] = event.eventNumber
                df_.MJD[i] = event.MJD
                df_.Time[i] = event.Time
                df_.Elevation[i] = pointingData.TelElevation
                df_.theta2[i] = event.theta2
                df_.MSCW[i] = event.MSCW
                df_.MSCL[i] = event.MSCL
                df_.ErecS[i] = event.ErecS
                df_.log10_EChi2S_[i] = log10_EChi2S_
                df_.log10_EmissionHeightChi2_[i] = log10_EmissionHeightChi2_
                df_.log10_SizeSecondMax_[i] = log10_SizeSecondMax_
                df_.sqrt_Xcore_T_Xcore_P_Ycore_T_Ycore_[i] = sqrt_Xcore_T_Xcore_P_Ycore_T_Ycore_
                # NImages, Xoff, Yoff not used:
                df_.NImages[i] = 0.0
                df_.Xoff[i] = 0.0
                df_.Yoff[i] = 0.0
                df_.IsGamma[i] = event.IsGamma
                df_.MVA[i] = 0.0
            if not hasattr(self, 'OnEvts'):
                self.OnEvts=df_
            else:
                self.OnEvts=pd.concat([self.OnEvts, df_])
    def make_BDT_on(self):
        if not hasattr(self, 'OnEvts'):
            print "No data frame for on events found, running self.get_data_on() now!"
            self.get_data_on()
        self.BDTon = self.OnEvts.drop(['Elevation','runNumber','eventNumber','MJD', 'Time', 'theta2', 'MVA', 'IsGamma'],axis=1)
        self.BDT_ErecS = self.OnEvts.ErecS
        self.BDT_Elevation = self.OnEvts.Elevation
        self.E_bins=np.digitize(self.BDT_ErecS, self.E_grid)-1
        self.Z_bins=np.digitize((90.-self.BDT_Elevation), self.Zen_grid)-1
    def predict_BDT_on(self, modelpath='.', modelbase='BDT', modelext='.model', scaler=None,fit_transform='linear'):
        if not hasattr(self, 'OnEvts'):
            print "No data frame for on events found, running self.get_data_on() now!"
            self.get_data_on()
        if not scaler:
            print "Warning: scaler not provided for prediction data!"
            scaler = StandardScaler()
        if fit_transform=='log':
            print "log transform the input features"
            self.BDTon = scaler.fit_transform(np.log(self.BDTon + 1.)).astype(np.float32)
        elif fit_transform=='linear':
            self.BDTon = scaler.fit_transform(self.BDTon).astype(np.float32)
        else:
            self.BDTon = self.BDTon.astype(np.float32)
        # Now divide into bins in ErecS and Zen
        # Note that if ErecS>50TeV or Zen>75, put them in the highest bin (bias! but rare)
        #for E in np.unique(self.E_bins):
        for E in [0,1,2,3]:
            for Z in np.unique(self.Z_bins):
                predict_x = self.BDTon[np.where((self.E_bins==E) & (self.Z_bins==Z))]
                modelname = modelpath+str('/')+modelbase+str(E)+str(Z)+modelext
                print "Using model %s" % modelname
                clf = xgb.Booster() #init model
                clf.load_model(modelname) # load model
                predict_y = clf.predict(xgb.DMatrix(predict_x))
                self.OnEvts.MVA.values[np.where((self.E_bins==E) & (self.Z_bins==Z))] = predict_y
    def get_run_off(self):
        self.runOff = []
        runSum = self.Rfile.Get("total_1/stereo/tRunSummary");
        for event in runSum:
            if event.runOff > 0:
                self.runOff.append(event.runOff)
        if not self.runOff:
            print "No runs found in data_off tree! "
            raise RuntimeError
    def get_data_off(self, runNum=0):
        if runNum==0:
            if not hasattr(self, 'runOff'):
                self.get_run_off()
            for run_ in self.runOff:
                self.get_data_off(runNum=run_)
        else:
            if not hasattr(self, 'Rfile'):
                print "No file has been read. Run self.readEDfile(filename=\"rootfile\") first. "
                raise
            data_off_treeName = "run_"+str(runNum)+"/stereo/data_off"
            pointingData_treeName = "run_"+str(runNum)+"/stereo/pointingDataReduced"
            data_off = self.Rfile.Get(data_off_treeName);
            pointingData = self.Rfile.Get(pointingData_treeName);
            ptTime=[]
            for ptd in pointingData:
                ptTime.append(ptd.Time);
            ptTime=np.array(ptTime)
            columns=['runNumber','eventNumber', 'MJD', 'Time', 'Elevation', 'theta2',
                     'MSCW','MSCL','log10_EChi2S_','EmissionHeight',
                     'log10_EmissionHeightChi2_','log10_SizeSecondMax_',
                     'sqrt_Xcore_T_Xcore_P_Ycore_T_Ycore_', 'NImages','Xoff','Yoff',
                     'ErecS', 'MVA', 'IsGamma']
            df_ = pd.DataFrame(np.array([np.zeros(data_off.GetEntries())]*len(columns)).T,
                               columns=columns)
            for i, event in enumerate(data_off):
                time_index=np.argmax(ptTime>event.Time)
                pointingData.GetEntry(time_index)
                # convert some quantities to BDT input
                log10_EChi2S_ = np.log10(event.EChi2S);
                log10_EmissionHeightChi2_ = np.log10(event.EmissionHeightChi2);
                log10_SizeSecondMax_ = np.log10(event.SizeSecondMax);
                sqrt_Xcore_T_Xcore_P_Ycore_T_Ycore_ = np.sqrt(event.Xcore*event.Xcore + event.Ycore*event.Ycore);
                # fill the pandas dataframe
                df_.runNumber[i] = event.runNumber
                df_.eventNumber[i] = event.eventNumber
                df_.MJD[i] = event.MJD
                df_.Time[i] = event.Time
                df_.Elevation[i] = pointingData.TelElevation
                df_.theta2[i] = event.theta2
                df_.MSCW[i] = event.MSCW
                df_.MSCL[i] = event.MSCL
                df_.ErecS[i] = event.ErecS
                df_.log10_EChi2S_[i] = log10_EChi2S_
                df_.log10_EmissionHeightChi2_[i] = log10_EmissionHeightChi2_
                df_.log10_SizeSecondMax_[i] = log10_SizeSecondMax_
                df_.sqrt_Xcore_T_Xcore_P_Ycore_T_Ycore_[i] = sqrt_Xcore_T_Xcore_P_Ycore_T_Ycore_
                # NImages, Xoff, Yoff not used:
                df_.NImages[i] = 0.0
                df_.Xoff[i] = 0.0
                df_.Yoff[i] = 0.0
                df_.IsGamma[i] = event.IsGamma
                df_.MVA[i] = 0.0
            if not hasattr(self, 'OffEvts'):
                self.OffEvts=df_
            else:
                self.OffEvts=pd.concat([self.OffEvts, df_])
    def make_BDT_off(self):
        if not hasattr(self, 'OffEvts'):
            print "No data frame for off events found, running self.get_data_off() now!"
            self.get_data_off()
        self.BDToff = self.OffEvts.drop(['Elevation','runNumber','eventNumber','MJD', 'Time', 'theta2', 'MVA', 'IsGamma'],axis=1)
        self.BDT_ErecS_off = self.OffEvts.ErecS
        self.BDT_Elevation_off = self.OffEvts.Elevation
        self.E_bins_off=np.digitize(self.BDT_ErecS_off, self.E_grid)-1
        self.Z_bins_off=np.digitize((90.-self.BDT_Elevation_off), self.Zen_grid)-1
    def predict_BDT_off(self, modelpath='.', modelbase='BDT', modelext='.model', scaler=None,fit_transform='linear'):
        if not hasattr(self, 'OffEvts'):
            print "No data frame for off events found, running self.get_data_off() now!"
            self.get_data_off()
        if not scaler:
            print "Warning: scaler not provided for prediction data!"
            scaler = StandardScaler()
        if fit_transform=='log':
            print "log transform the input features"
            self.BDToff = scaler.fit_transform(np.log(self.BDToff + 1.)).astype(np.float32)
        elif fit_transform=='linear':
            self.BDToff = scaler.fit_transform(self.BDToff).astype(np.float32)
        else:
            self.BDToff = self.BDToff.astype(np.float32)
        # Now divide into bins in ErecS and Zen
        # Note that if ErecS>50TeV or Zen>75, ignore them (rare)
        #for E in np.unique(self.E_bins_off):
        for E in [0,1,2,3]:
            for Z in np.unique(self.Z_bins_off):
                predict_x = self.BDToff[np.where((self.E_bins_off==E) & (self.Z_bins_off==Z))]
                modelname = modelpath+str('/')+modelbase+str(E)+str(Z)+modelext
                print "Using model %s" % modelname
                clf = xgb.Booster() #init model
                clf.load_model(modelname) # load model
                predict_y = clf.predict(xgb.DMatrix(predict_x))
                self.OffEvts.MVA.values[np.where((self.E_bins_off==E) & (self.Z_bins_off==Z))] = predict_y

    def write_data(self, newfile=None, runNum=0):
        if newfile==None:
            base = os.path.splitext(self.filename)[0]
            newfile = base+"_xgb.root"
        if not os.path.isfile(newfile):
            os.system("cp %s %s" % (self.filename, newfile))
        self.xgbfile = ROOT.TFile(newfile, "UPDATE");
        if runNum==0:
            #here assuming that runOn and runOff have the same list of runs
            if not hasattr(self, 'runOn'):
                self.get_run_on()
            for run_ in self.runOn:
                self.write_data(runNum=run_)
        else:
            data_on_treeName = "run_"+str(runNum)+"/stereo/data_on"
            data_on = self.xgbfile.Get(data_on_treeName);
            data_off_treeName = "run_"+str(runNum)+"/stereo/data_off"
            data_off = self.xgbfile.Get(data_off_treeName);
            mva_ = np.zeros(1, dtype=float)
            mva_off_ = np.zeros(1, dtype=float)
            mva_onlist = np.zeros(data_on.GetEntries(), dtype=float)
            mva_offlist = np.zeros(data_off.GetEntries(), dtype=float)
            Bran_MVAon = data_on.Branch('xMVA', mva_, 'xMVA/D')
            Bran_MVAoff = data_off.Branch('xMVA', mva_off_, 'xMVA/D')
            data_on.SetBranchStatus("*", 1)
            data_off.SetBranchStatus("*", 1)
            #for event in data_on:
            #    mva_[0] = self.OnEvts.MVA.values[np.where((self.OnEvts.eventNumber==event.eventNumber) & (self.OnEvts.runNumber==event.runNumber))]
            for i in range(data_on.GetEntries()):
                data_on.GetEntry(i)
                mva_[0] = self.OnEvts.MVA.values[np.where((self.OnEvts.eventNumber==data_on.eventNumber) & (self.OnEvts.runNumber==data_on.runNumber))]
                #mva_onlist[i] = self.OnEvts.MVA.values[np.where((self.OnEvts.eventNumber==data_on.eventNumber) & (self.OnEvts.runNumber==data_on.runNumber))]
                #data_on.Fill()
                Bran_MVAon.Fill()
            #for i in range(data_on.GetEntries()):
            #    mva_[0] = mva_onlist[i]
            #    data_on.Fill()
            #for event in data_off:
            for i in range(data_off.GetEntries()):
                data_off.GetEntry(i)
                mva_off_[0] = self.OffEvts.MVA.values[np.where((self.OffEvts.eventNumber==data_off.eventNumber) & (self.OffEvts.runNumber==data_off.runNumber))]
                #mva_offlist[i] = self.OffEvts.MVA.values[np.where((self.OffEvts.eventNumber==data_off.eventNumber) & (self.OffEvts.runNumber==data_off.runNumber))]
                #mva_off_[0] = self.OffEvts.MVA.values[np.where((self.OffEvts.eventNumber==event.eventNumber) & (self.OffEvts.runNumber==event.runNumber))]
                #data_off.Fill()
                Bran_MVAoff.Fill()
            #for i in range(data_off.GetEntries()):
            #    mva_off_[0] = mva_offlist[i]
            #    data_off.Fill()
            data_on.Write()
            data_off.Write()
            self.xgbfile.Write()
            self.xgbfile.Close()


def read_data(filename='BDT_1_1.txt', predict=False, scaler=None, fit_transform='linear'):
    if predict==True:
        print "Read data for prediction..."
        x = pd.read_csv(filename)
        x = np.array(x.values)
        if not scaler:
            print "Warning: scaler not provided for prediction data!"
            scaler = StandardScaler()
        if fit_transform=='log':
            print "log transform the input features"
            x = scaler.fit_transform(np.log(x + 1.)).astype(np.float32)
        elif fit_transform=='linear':
            x = scaler.fit_transform(x).astype(np.float32)
        else:
            x = x.astype(np.float32)
        return x
    print "Read train/test data..."
    data = pd.read_csv(filename, header=None, sep=r"\s+")
    data.columns=['classID','className','MSCW','MSCL','log10_EChi2S_','EmissionHeight',
                  'log10_EmissionHeightChi2_','log10_SizeSecondMax_','sqrt_Xcore_T_Xcore_P_Ycore_T_Ycore_',
                  'NImages','Xoff','Yoff','ErecS','weight','BDT_0',]
    scaler = StandardScaler()
    #x = np.array(data.drop(['classID','className','BDT_0'],axis=1).values)
    # x has columns: ['MSCW','MSCL','log10_EChi2S_','EmissionHeight',
    #              'log10_EmissionHeightChi2_','log10_SizeSecondMax_','sqrt_Xcore_T_Xcore_P_Ycore_T_Ycore_',
    #              'NImages','Xoff','Yoff','ErecS']
    x = np.array(data.drop(['classID','className','weight','BDT_0'],axis=1).values)
    if fit_transform=='log':
        print "log transform the input features"
        x = scaler.fit_transform(np.log(x + 1.)).astype(np.float32)
    elif fit_transform=='linear':
        x = scaler.fit_transform(x).astype(np.float32)
    else:
        x = x.astype(np.float32)
    y = data['classID']
    y = y.values.astype(np.int32)
    return x, y, scaler

def read_ED_anasum_data(filename='test_Crab_V6_ED_RE.txt', scaler=None, fit_transform='linear'):
    print "Reading EventDisplay anasum data..."
    data = pd.read_csv(filename, header=None, sep=r"\s+")
    data.columns=['runNum','evtNum','MSCW','MSCL','log10_EChi2S_','EmissionHeight',
                  'log10_EmissionHeightChi2_','log10_SizeSecondMax_','sqrt_Xcore_T_Xcore_P_Ycore_T_Ycore_',
                  'NImages','Xoff','Yoff','ErecS','IsGamma']
    x = np.array(data.drop(['runNum','evtNum','IsGamma'],axis=1).values)
    if not scaler:
        print "Warning: scaler not provided for prediction data!"
        scaler = StandardScaler()
    if fit_transform=='log':
        print "log transform the input features"
        x = scaler.fit_transform(np.log(x + 1.)).astype(np.float32)
    elif fit_transform=='linear':
        x = scaler.fit_transform(x).astype(np.float32)
    else:
        x = x.astype(np.float32)
    y = data['IsGamma']
    y = y.values.astype(np.int32)
    return x, y, scaler

def read_data_xgb(filename='BDT_1_1.txt', predict=False, cv_ratio=0.1, scaler=None, fit_transform='linear', random_state=1234): 
    if predict:
        x, y, _ = read_data(filename=filename, predict=False, scaler=scaler, fit_transform=fit_transform)
        dtestx = xgb.DMatrix(x)
        #dtesty = xgb.DMatrix(y)
        return dtestx, y
    else:
        x,y,_ = read_data(filename=filename, predict=predict, scaler=scaler, fit_transform=fit_transform)
        sss = StratifiedShuffleSplit(y, test_size=cv_ratio, random_state=random_state)
        for train_index, test_index in sss:
            break
        train_x, train_y = x[train_index], y[train_index]
        test_x, test_y = x[test_index], y[test_index]
        dtrain = xgb.DMatrix(train_x, label= train_y)
        deval  = xgb.DMatrix(test_x, label=test_y)
        watchlist  = [(dtrain,'train'),(deval,'eval')]
        return watchlist

def do_RF3(train_x, train_y, test_x=None, test_y=None, n_estimators=2000, max_depth=20, max_features=20,
          criterion='entropy', method='isotonic', cv=5,
          min_samples_leaf=1, min_samples_split=13, random_state=4141, n_jobs=-1, load=False, save=True,
          outfile=None, search=False, log=False):
    if search == False:
        #mdl_name = 'rf_train_n' + str(n_estimators) + '_maxdep' + str(max_depth) + '_maxfeat' + str(max_features) \
        if log==True:
            mdl_name = 'rf_log_isotonic_train_n' + str(n_estimators) + '_maxdep' + str(max_depth) + '_maxfeat' + str(max_features) \
                + '_minSamLeaf' + str(min_samples_leaf) + '_minSamSplit' + str(min_samples_split) + '.pkl'
        else:
            mdl_name = 'rf_isotonic_train_n' + str(n_estimators) + '_maxdep' + str(max_depth) + '_maxfeat' + str(max_features) \
                   + '_minSamLeaf' + str(min_samples_leaf) + '_minSamSplit' + str(min_samples_split) + '.pkl'
        if os.path.exists(mdl_name) == True:
            clf_rf_isotonic = joblib.load(mdl_name)
        else:
            clf_rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                     max_features=max_features, criterion=criterion,
                                                     min_samples_leaf=min_samples_leaf,
                                                     min_samples_split=min_samples_split, random_state=random_state,
                                                     n_jobs=n_jobs)
            clf_rf_isotonic = CalibratedClassifierCV(clf_rf, cv=cv, method=method)
            clf_rf_isotonic.fit(train_x, train_y)
            if save == True:
                try:
                    _ = joblib.dump(clf_rf_isotonic, mdl_name, compress=1)
                except:
                    print("*** Save RF model to pickle failed!!!")
                    if outfile != None:
                        outfile.write("*** Save RF model to pickle failed!!!")
        if test_x != None and test_y != None:
            probas_rf = clf_rf_isotonic.predict_proba(test_x)[:, 1]
            score_rf = roc_auc_score(test_y, probas_rf)
            print("RF ROC score", score_rf)
        return clf_rf_isotonic
    else:
        if test_x == None or test_y == None:
            print "Have to provide test_x and test_y to do grid search!"
            return -1

        min_samples_split = [10, 11, 12]
        max_depth_list = [13, 15, 17]
        n_list = [2000]
        max_feat_list = [8, 10, 12]
        info = {}
        for mss in min_samples_split:
            for max_depth in max_depth_list:
                #for n in n_list:
                for max_features in max_feat_list:
                    print 'max_features = ', max_features
                    n=2000
                    print 'n = ', n
                    print 'min_samples_split = ', mss
                    print 'max_depth = ', max_depth
                    clf_rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth, max_features=max_features,
                                                    criterion=criterion, min_samples_leaf=min_samples_leaf,
                                                    min_samples_split=mss, random_state=random_state, n_jobs=n_jobs)
                    #clf_rf.fit(train_x, train_y)
                    clf_rf_isotonic = CalibratedClassifierCV(clf_rf, cv=cv, method=method)
                    mdl_name = 'rf_n'+str(n)+'_mss'+str(mss)+'_md'+str(max_depth)+'mf'+str(max_features)+'.pkl'
                    if os.path.exists(mdl_name) == True:
                        clf_rf_isotonic = joblib.load(mdl_name)
                    else:
                        clf_rf_isotonic = CalibratedClassifierCV(clf_rf, cv=cv, method=method)
                        clf_rf_isotonic.fit(train_x, train_y)
                        _ = joblib.dump(clf_rf_isotonic, mdl_name, compress=1)
                    probas_rf = clf_rf_isotonic.predict_proba(test_x)[:, 1]
                    scores = roc_auc_score(test_y, probas_rf)
                    info[max_features, mss, max_depth] = scores
        for mss in info:
            scores = info[mss]
            print(
                'clf_rf_isotonic: max_features = %d, min_samples_split = %d, max_depth = %d, ROC score = %.5f(%.5f)' % (mss[0], mss[1], mss[2], scores.mean(), scores.std()))


def do_gbdt4(train_x, train_y, test_x=None, test_y=None, learning_rate=0.03, max_depth=8, max_features=25,
            n_estimators=600, load=False, save=True, outfile=None, search=False, log=False):
    if search == False:
        if log==True:
            mdl_name = 'gbdt_log_train_lr' + str(learning_rate) + '_n' + str(n_estimators) + '_maxdep' + str(max_depth) + '.pkl'
        else:
            mdl_name = 'gbdt_train_lr' + str(learning_rate) + '_n' + str(n_estimators) + '_maxdep' + str(max_depth) + '.pkl'
        if os.path.exists(mdl_name) == True:
            clf_gbdt = joblib.load(mdl_name)
        else:
            # create gradient boosting
            clf_gbdt = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth,
                                                  max_features=max_features, n_estimators=n_estimators)
            #n_estimators=500, learning_rate=0.5, max_depth=3)
            clf_gbdt.fit(train_x, train_y)
            if save == True:
                try:
                    _ = joblib.dump(clf_gbdt, mdl_name, compress=1)
                except:
                    print("*** Save GBM model to pickle failed!!!")
                    if outfile != None:
                        outfile.write("*** Save RF model to pickle failed!!!")
        if test_x != None and test_y != None:
            probas_gbdt = clf_gbdt.predict_proba(test_x)[:, 1]
            score_gbdt = roc_auc_score(test_y, probas_gbdt)
            print("GBDT ROC score", score_gbdt)
        return clf_gbdt
    else:
        max_depth_list = [ 6, 7, 8, 9, 10]
        n_list = [2000]
        lr_list = [0.005,0.003]
        max_feat_list = [15, 16, 17, 18, 20]
        info = {}
        for md in max_depth_list:
            for n in n_list:
                for lr in lr_list:
                  for mf in max_feat_list:
                    print 'max_depth = ', md
                    print 'n = ', n
                    print 'learning rate = ', lr
                    print 'max feature = ', mf
                    # n_estimators=500, learning_rate=0.5, max_depth=3)
                    mdl_name = 'gbdt_n'+str(n)+'_lr'+str(lr)+'_md'+str(md)+'mf'+str(mf)+'.pkl'
                    if os.path.exists(mdl_name) == True:
                        clf_gbdt = joblib.load(mdl_name)        
                    else:
                        clf_gbdt = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=md,max_features=mf, n_estimators=n_estimators)
                        clf_gbdt.fit(train_x, train_y)
                        _ = joblib.dump(clf_gbdt, mdl_name, compress=1)
                    probas_gbdt = clf_gbdt.predict_proba(test_x)[:, 1]
                    score_gbdt = roc_auc_score(test_y, probas_gbdt)
                    info[md, n, lr, mf] = score_gbdt
        for md in info:
            scores = info[md]
            print('GBDT max_depth = %d, n = %d, lr = %.5f, max_feature = %d, ROC score = %.5f(%.5f)' % (
                md[0], md[1], md[2], md[3], scores.mean(), scores.std()))


def do_nn(xTrain, yTrain, test_x=None, test_y=None, dropout_in=0.2, dense0_num=800, dropout_p=0.5, dense1_num=500,
           update_learning_rate=0.0002, update_momentum=0.9, test_ratio=0.1, max_epochs=40, search=False):
    num_features = len(xTrain[0, :])
    num_classes = 2
    print num_features
    if search == False:
        layers0 = [('input', InputLayer),
                   ('dropoutin', DropoutLayer),
                   ('dense0', DenseLayer),
                   ('dropout', DropoutLayer),
                   ('dense1', DenseLayer),
                   ('output', DenseLayer)]
        clf = NeuralNet(layers=layers0,
                        input_shape=(None, num_features),
                        dropoutin_p=dropout_in,
                        dense0_num_units=dense0_num,
                        dropout_p=dropout_p,
                        dense1_num_units=dense1_num,
                        output_num_units=num_classes,
                        output_nonlinearity=softmax,
                        update=nesterov_momentum,
                        update_learning_rate=update_learning_rate,
                        update_momentum=update_momentum,
                        eval_size=test_ratio,
                        verbose=1,
                        max_epochs=max_epochs)
        clf.fit(xTrain, yTrain)
        if test_x != None and test_y != None:
            probas_nn = clf.predict_proba(test_x)[:, 1]
            score_nn = roc_auc_score(test_y, probas_nn)
            print("NN ROC score", score_nn)
        return clf
    else:
        dropout_in_list = [0.1]
        dense0_num_list = [ 1000, 1200, 1400]
        dropout_p_list = [0.5, 0.4]
        dense1_num_list = [50, 100, 150, 200]
        info = {}
        for d_in in dropout_in_list:
            for d_01 in dropout_p_list:
                for d0 in dense0_num_list:
                    for d1 in dense1_num_list:
                        print 'dropout_in = ', d_in
                        print 'dense0_num = ', d0
                        print 'dropout_p = ', d_01
                        print 'dense1_num = ', d1
                        layers0 = [('input', InputLayer),
                                   ('dropoutin', DropoutLayer),
                                   ('dense0', DenseLayer),
                                   ('dropout', DropoutLayer),
                                   ('dense1', DenseLayer),
                                   ('output', DenseLayer)]
                        clf = NeuralNet(layers=layers0,
                                        input_shape=(None, num_features),
                                        dropoutin_p=d_in,
                                        dense0_num_units=d0,
                                        dropout_p=d_01,
                                        dense1_num_units=d1,
                                        output_num_units=num_classes,
                                        output_nonlinearity=softmax,
                                        update=nesterov_momentum,
                                        update_learning_rate=update_learning_rate,
                                        update_momentum=update_momentum,
                                        eval_size=test_ratio,
                                        verbose=1,
                                        max_epochs=max_epochs)
                        clf.fit(xTrain, yTrain)
                        probas_nn = clf.predict_proba(test_x)[:, 1]
                        score_nn = roc_auc_score(test_y, probas_nn)
                        print 'dropout_in = ', d_in
                        print 'dense0_num = ', d0
                        print 'dropout_p = ', d_01
                        print 'dense1_num = ', d1
                        print("NN ROC score", score_nn)
                        info[d_in, d0, d_01, d1] = score_nn
        for md in info:
            scores = info[md]
            print('NN dropout_in = %.5f, dense0_num = %d, dropout_p = %.5f, dense1_num = %d, ROC score = %.5f(%.5f)' % \
                  (md[0], md[1], md[2], md[3], scores.mean(), scores.std()))

def do_xgb(filename='BDT_1_1_V6.txt',search=False, logfile=None, max_depth=15, eta=0.04, gamma=5, subsample=0.6,colsample_bytree=0.7, num_round=200, predict_file=None, early_stop=0, test_ratio=0.1):
    x,y,_ = read_data(filename=filename)
    sss = StratifiedShuffleSplit(y, test_size=test_ratio, random_state=1234)
    for train_index, test_index in sss:
        break
    train_x, train_y = x[train_index], y[train_index]
    test_x, test_y = x[test_index], y[test_index]
    dtrain = xgb.DMatrix(train_x, label= train_y)
    deval  = xgb.DMatrix(test_x, label=test_y)
    watchlist  = [(dtrain,'train'),(deval,'eval')]
    if search==True:
        #num_round = 200
        info3 = {}
        for md in max_depth:
        #for md in [12, 14, 16, 18, 20]:
            for eta_ in eta:
                for gamma_ in gamma:
                    for sam in subsample:
                        for col in colsample_bytree:
                            param = {'max_depth':md, 'eta':eta_,'eval_metric':'auc', 'silent':1, 'objective':'binary:logistic', 'gamma':gamma_, 'subsample':sam, 'colsample_bytree':col }
                            if early_stop>0:
                                clf_xgb = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=early_stop)
                                info3[md,eta_,gamma_,sam,col,clf_xgb.best_iteration] = clf_xgb.best_score
                            else:
                                clf_xgb = xgb.train(param, dtrain, num_round, watchlist)
                                info3[md,eta_,gamma_,sam,col] = clf_xgb.eval(deval)
        if early_stop>0:
            score3 = np.array(info3.values())
        else:
            score3 = [float(info3.values()[i][13:]) for i in range(len(info3))]
            score3 = np.array(score3)
        print info3, score3
        if logfile!=None:
            out_df=pd.concat([pd.DataFrame(info3.keys()), pd.DataFrame(score3)], axis=1)
            out_df.columns=['max_depth', 'eta', 'gamma', 'subsample', 'colsample_bytree', 'best_iteration', 'best_score']
            out_df.to_csv(logfile, index=False)
        return info3, score3

    param = {'max_depth':max_depth, 'eta':eta,'eval_metric':'auc', 'silent':1, 'objective':'binary:logistic', 'gamma':gamma, 'subsample':subsample, 'colsample_bytree':colsample_bytree}
    if early_stop>0:
        clf_xgb = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=early_stop)
    else:
        clf_xgb = xgb.train(param, dtrain, num_round, watchlist)
    #param2 = {'max_depth':3, 'eta':0.5, 'eval_metric':'auc', 'silent':1, 'objective':'binary:logistic' }
    #clf_xgb2 = xgb.train(param2, dtrain, num_round, watchlist)
    #param3 = {'max_depth':4, 'eta':0.3, 'eval_metric':'auc', 'silent':1, 'objective':'binary:logistic' }
    #clf_xgb3 = xgb.train(param2, dtrain, num_round, watchlist)
    if predict_file != None:
        #Not ready yet
        _x,_y,_ = read_data(filename=filename)
        dtest = xgb.DMatrix(_x)
        preds = bst.predict(dtest)
    return clf_xgb

def print_best_xgb(lfile, num=8):
    df_ = pd.read_csv(lfile)
    pd.set_option('display.width', 1000)
    print df_.sort(columns='best_score', ascending=False).head(num)

def plot_pseudo_TMVA(model_file="BDT11.model", train_file="V6/BDT_1_1_V6.txt", test_file="V6/BDT_1_1_Test_V6.txt", ifKDE=False, outfile='BDT_1_1_xgb_counts.png', nbins=40):
    clf = xgb.Booster() #init model
    clf.load_model(model_file) # load data
    train_x, train_y =read_data_xgb(train_file, predict=True)
    test_x, test_y =read_data_xgb(test_file, predict=True)
    predict_train_y = clf.predict(train_x)
    predict_test_y = clf.predict(test_x)
    sns.distplot(predict_train_y[np.where(train_y==1)]*2.-1.,
                 bins=nbins, hist=True, kde=ifKDE, rug=False, 
                 hist_kws={"histtype": "step", "linewidth": 1,
                           "alpha": 1, "color": "darkblue"}, 
                 label="Training Signal")
    sns.distplot(predict_train_y[np.where(train_y==0)]*2.-1.,
                 bins=nbins, hist=True, kde=ifKDE, rug=False, 
                 hist_kws={"histtype": "step", "linewidth": 1,
                           "alpha": 1, "color": "darkred"},
                 label="Training Background")
    sns.distplot(predict_test_y[np.where(test_y==1)]*2.-1.,
                 color='b', bins=nbins, hist=True, kde=ifKDE, rug=False, 
                 label="Test Signal")
    sns.distplot(predict_test_y[np.where(test_y==0)]*2.-1.,
                 color='r', bins=nbins, hist=True, kde=ifKDE, rug=False, 
                 label="Test Background")
    sns.axlabel("Pseudo TMVA value","Event counts")
    plt.legend(loc='best');
    plt.savefig(outfile, format='png', dpi=1000)

def roc_func(weights, predictions, test_y):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction
    return roc_auc_score(test_y, final_prediction)

def make_predictions(clfs, predict_x, enrollment_id, test_x=None, test_y=None, outfile='test_sub.csv', weights=[]):
    scores = []    
    predictions = []
    for clf in clfs:
        _probas = clf.predict_proba(test_x)[:,1]
        _score = roc_auc_score(test_y, _probas)
        print("ROC score", _score)
        predictions.append(_probas)
        scores.append(_score)
    starting_values = [0.5]*len(predictions)
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    #our weights are bound between 0 and 1    
    bounds = [(0,1)]*len(predictions)
    print len(predictions), len(test_y)
    res = minimize(roc_func, starting_values, args= (predictions, test_y), method='SLSQP', bounds=bounds, constraints=cons)
    predict_y=[]
    for clf, wt in zip(clfs, res['x']):
        pred_test_current_clf = pd.DataFrame((clf.predict_proba(predict_x))) * wt
        if not predict_y:
            predict_y = pred_test_current_clf
        else:
            predict_y = predict_y + pred_test_current_clf
    sub = pd.concat([pd.DataFrame(enrollment_id), pd.DataFrame(predict_y)],axis=1)
    sub.to_csv(outfile, index=False, header=False)

def plotPseudoMvaValue(y):
    sns.distplot(y*2.-1.)
    return y*2.-1.

def run(train_file='train_xy2.csv', test_file='test_features2.csv', outfile='test_sub2.csv'):
    train_ratio = 0.9
    test_ratio = 1 - train_ratio
    x, y = read_train(file=train_file, test=0.1)
    sss = StratifiedShuffleSplit(y, test_size=test_ratio, random_state=1234)
    for train_index, test_index in sss:
        break

    train_x, train_y = x[train_index], y[train_index]
    test_x, test_y = x[test_index], y[test_index]

    predict_x, enrollment_id = read_test(file=test_file)

    clf_nn = do_nn(train_x, train_y, test_x=test_x, test_y=test_y, dropout_in=0.1, dense0_num=200, dropout_p=0.5,
                   dense1_num=400, update_learning_rate=0.00003, update_momentum=0.9, test_ratio=0.1, max_epochs=20)
    clf_rf = do_RF3(train_x, train_y, test_x=test_x, test_y=test_y)
    clf_gbdt = do_gbdt3(train_x, train_y, test_x=test_x, test_y=test_y)

    make_predictions([clf_rf, clf_gbdt,clf_nn], predict_x, enrollment_id, test_x=test_x, test_y=test_y,
                     outfile=outfile)

def tune3(train_x, train_y, test_x, test_y):
    do_nn(train_x, train_y, test_x=test_x, test_y=test_y, search=True)
    do_RF(train_x, train_y, test_x=test_x, test_y=test_y, search=True)
    do_gbdt(train_x, train_y, test_x=test_x, test_y=test_y, search=True)

def rocFunc(weights, predictions, test_y):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
        final_prediction += weight*prediction
    return -roc_auc_score(test_y, final_prediction)

def makeSub(clfs, predict_x, enrollment_id, scaler, test_x, test_y, deval, dtest, outfile='test_sub.csv'):
    scores = []
    predictions = []

    for ii, clf in enumerate(clfs):
        if ii < 6:
                #print ii, clf
                #_probas = clf.predict(xgb.DMatrix(test_x))[:,1]
                _probas = clf.predict(deval)
        else:
                #print ii, clf
                _probas = clf.predict_proba(test_x)[:,1] #drop out prob
        _score = roc_auc_score(test_y, _probas)
        print("ROC score", _score)
        predictions.append(_probas)
        scores.append(_score)

    starting_values = [1.0/len(predictions)]*len(predictions)

    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    bounds = [(0.,1.)]*len(predictions)

    res = minimize(rocFunc, starting_values, args= (predictions, test_y), method='SLSQP',
            options={'disp': True, 'eps': 5e-2},
            bounds=bounds, constraints=cons)
    print res

    print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))


    #predict_x, enrollment_id = loadData(test_file,test=True,scaler=scaler)
    predict_y = 0
    kk = 0
    for clf, wt in zip(clfs, res['x']):
        if kk < 6:
            predict_y += clf.predict(dtest)
        else:
            predict_y += clf.predict_proba(predict_x)[:,1] * wt
        kk += 1
    sub = pd.concat([pd.DataFrame(enrollment_id), pd.DataFrame(predict_y)],axis=1)
    sub.to_csv(outfile, index=False, header=False)



if __name__ == '__main__':
    #make_feature(test=False, outfile='train_xy2.csv')
    #make_feature(test=True, outfile='test_features2.csv')
    #tune(train_file='train_xy2.csv')

    #run(train_file='train_xy2.csv', test_file='test_features2.csv', outfile='test_sub2.csv')
    import os
    for f in os.listdir("/project/veritas/qfeng/gamma_learn/V6"):
        if f.endswith(".txt"):
            print "working on file "+str(f)+"..."
            do_xgb(file=f, search=True,logfile=f.split('.')[0]+".log")
