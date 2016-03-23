__author__ = 'qfeng'

#get all VEGAS events from on region as on events:
import ROOT
import cPickle as pickle
import numpy as np
from get_raw_features import *

def getVEGASonEvtNum(f, runnum=72045, thetasq=0.01):
    #f = ROOT.TFile.Open("config__1ES1215_/results__1ES1215_s6.root", "READ")
    f = ROOT.TFile.Open(f, "READ")
    EvtTree=f.Get("EventStatsTree")

    events=[]
    for entry in EvtTree:
        if entry.OnEvent==1 and entry.RunNum==runnum and entry.ThetaSq<thetasq:
            #print entry.ArrayEventNum, entry.ThetaSq
            events.append(int(entry.ArrayEventNum))

    output = open('VEGASon'+str(runnum)+'.pkl', 'wb')
    pickle.dump(np.array(events), output)
    output.close()
    return np.array(events)


def getVEGASonOneRun(f, runnum=72045, thetasq=0.01):
    getVEGASonEvtNum(f, runnum=72045, thetasq=thetasq)
    train_x1, train_y1, test_x1, test_y1 = get_raw_features_on(infile=str(runnum)+'.txt',
                    in_pickle_onfile = 'on'+str(runnum)+'.pkl', test_ratio=0.3, random_state=1234,
                    dump=False, out_trainx=str(runnum)+'_raw_trainx_on.pkl', out_trainy=str(runnum)+'_raw_trainy_on.pkl',
                    out_testx=str(runnum)+'_raw_testx_on.pkl', out_testy=str(runnum)+'_raw_testy_on.pkl')
    return train_x1, train_y1, test_x1, test_y1

def getVEGASon(f, runs, thetasq=0.01):
    for i, runnum in enumerate(runs):
        if i==0:
            train_x, train_y, test_x, test_y = getVEGASonOneRun(f, runnum=runnum)
        else:
            train_x_, train_y_, test_x_, test_y_ = getVEGASonOneRun(f, runnum=runnum)
            train_x, train_y, test_x, test_y = concat_data(train_x, train_y, test_x, test_y, train_x_, train_y_, test_x_, test_y_)

    return train_x, train_y, test_x, test_y
