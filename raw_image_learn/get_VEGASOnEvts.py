__author__ = 'qfeng'

#get all VEGAS events from on region as on events:
import ROOT
import cPickle as pickle
import numpy as np
import pandas as pd
from get_raw_features import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    ROOT.gSystem.Load("$VEGAS/common/lib/libSP24sharedLite.so")
except:
    print "Hey man, problem loading VEGAS, can't play with VEGAS root files..."

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
    getVEGASonEvtNum(f, runnum=runnum, thetasq=thetasq)
    train_x1, train_y1, test_x1, test_y1 = get_raw_features_on(infile=str(runnum)+'.txt',
                    in_pickle_onfile = 'VEGASon'+str(runnum)+'.pkl', test_ratio=0.3, random_state=1234,
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

def concat_data(train_x1, train_y1, test_x1, test_y1, train_x2, train_y2, test_x2, test_y2):
    train_x = np.concatenate((train_x1, train_x2), axis=0)
    train_y = np.concatenate((train_y1, train_y2), axis=0)
    test_x = np.concatenate((test_x1, test_x2), axis=0)
    test_y = np.concatenate((test_y1, test_y2), axis=0)
    return train_x, train_y, test_x, test_y

def read_st2_charge(f, tels=[0,1,2,3]):
    #io = ROOT.VARootIO("Oct2012_ua_ATM21_vegasv250rc5_7samples_20deg_050wobb_730noise.root", 1)
    io = ROOT.VARootIO(f, 1)
    q = io.loadTheQStatsData()
    #tel loop
    for tel in tels:
        telQBase = q.fTelQBaseColl.at(tel)
        telSliceQStats = q.fTimeSliceColl.at(0).fTelColl.at(tel)
        #chan loop
        for chan in range(telQBase.fChanColl.size()):
            #telChanQ = telQBase.fChanColl.at(chan)
            telChanQStats = telSliceQStats.fChanColl.at(chan)
            #sample loop
            for sample in range(telChanQStats.fSumWinColl.size()):
                try:
                    telChanSample = telChanQStats.fSumWinColl.at(sample)
                    print sample, telChanSample.fChargeMean
                except:
                    print("Can't get charge from tel %d, channel %d, sample %d" % (tel, chan, sample))

def quick_oversample2(pixVals, z_index, numX=54):
    #z = -np.ones((numX, numX))
    z = np.zeros((numX, numX))
    for i_ in range(len(pixVals)):
        x_ = int(z_index.at[i_, 'x1'])
        y_ = int(z_index.at[i_,'y1'])
        z[x_:x_+2, y_:y_+2] = pixVals[i_]
    return z

def read_st2_calib_charge(f, tels=[0,1,2,3], maskL2=True, getEvtNumber=False,
                          l2channels=[[110, 249, 255, 404, 475, 499], [128, 173, 259, 498, 499], [37, 159, 319, 451, 499], [99, 214, 333, 499]],
                          start_event=None, stop_event=None, evtlist=None, outfile=None):
    calib_io = ROOT.VARootIO(f, 1)
    calibTree = calib_io.loadTheCalibratedEventTree()
    calibEvtData = ROOT.VACalibratedArrayEvent()
    calibTree.SetBranchAddress("C", calibEvtData)
    #calibTree.SetBranchAddress("C", calibEvtData)

    #evtNum = []
    if start_event is None:
        start_event=0
    if stop_event is None and evtlist is None:
        totalEvtNum = calibTree.GetEntries()
        print "You want to get charge from all events."
    elif evtlist is None:
        assert start_event<stop_event, "Please specify sensible start_event and stop_event numbers. "
        totalEvtNum = stop_event+1-start_event
        evtlist = range(start_event, stop_event+1)
    else:
        totalEvtNum = len(evtlist)

    print("Processing %d events." % totalEvtNum)

    try:
        allCharge = np.zeros((4, 500, totalEvtNum))
        oversampledCharge = np.zeros((totalEvtNum, 4, 54, 54))
        if getEvtNumber:
            evtNums = np.zeros(totalEvtNum)
    except MemoryError:
        print("Such a large number of events caused a MemoryError... "
              "Let's try passing start_event and stop_event or evtlist to analyze a smaller set of events.")
        raise

    try:
        z_index = pd.read_csv("oversample_coordinates.csv")
    except:
        print "Can't load square coordinates for oversampled camera"

    if maskL2:
        try:
            neighborIDs = pd.read_csv("neighborID.csv")
        except:
            print "Can't load neighbor pixel IDs from neighborID.csv, not masking L2s"
            maskL2=False

    for evt_count, evt in enumerate(evtlist):
        try:
            calibTree.GetEntry(evt)
        except:
            print("Can't get calibrated event number %d" % evt)
        #evtNum.append(int(calibEvtData.fArrayEventNum))
        for telID in tels:
            try:
                for chanID in range(500):
                    try:
                        allCharge[telID][chanID][evt_count] = calibEvtData.fTelEvents.at(telID).fChanData.at(chanID).fCharge
                    except:
                        print("Can't get charge from tel %d channel %d for calibrated event number %d" % (telID, chanID, evt))
                    #hiLo[telID][chanID][evt] = calibEvtData.fTelEvents.at(telID).fChanData.at(chanID).fHiLo
                if maskL2:
                    for l2chan in l2channels[telID]:
                        neighborCharges = []
                        for nc in neighborIDs.iloc[l2chan,1][1:-1].split():
                            neighborCharges.append(allCharge[telID][int(nc)][evt_count])
                        allCharge[telID][l2chan][evt_count] = np.mean(neighborCharges)
                oversampledCharge[evt_count, telID] = quick_oversample2(allCharge[telID, :, evt_count], z_index)
            except:
                print "tel ", telID, "chan ",chanID, " event ", evt, " failed to get charge "
                allCharge[telID][chanID][evt_count]=0.
        if getEvtNumber:
            evtNums[evt_count] = calibEvtData.fArrayEventNum
    if outfile is not None:
        output = open(outfile, 'wb')
        pickle.dump(oversampledCharge, output)
        output.close()
        if getEvtNumber:
            output = open("ArrayEvtNum"+outfile, 'wb')
            pickle.dump(evtNums, output)
            output.close()
        #pd.DataFrame(allCharge).to_csv(outfile, index=False, header=None)
    if getEvtNumber:
        return oversampledCharge, evtNums
    return oversampledCharge

def mask_L2_channels_square(x, l2channels=[[110, 249, 255, 404, 475, 499], [128, 173, 259, 498], [37, 159, 319, 451, 499], [99, 214, 333, 499]]):
    assert len(x.shape)==4, "Expected a four dimension input features"
    z_index = pd.read_csv("oversample_coordinates.csv")
    z_index = z_index.drop(['x2', 'y2'], axis=1)
    for evt in range(x.shape[0]):
        for i, t in enumerate(l2channels):
            for c in t:
                #get neighbor pixels
                neighbor_index = np.where((abs(z_index.values[:,0] - z_index.values[c, 0])< 3) & (abs(z_index.values[:,1] - z_index.values[c,1]) < 3) & (abs(z_index.values[:,0] - z_index.values[c, 0])+abs(z_index.values[:,1] - z_index.values[c,1])>0))
                x[evt, i, z_index.values[c, 0].astype('int'):z_index.values[c, 0].astype('int')+2, z_index.values[c, 1].astype('int'):z_index.values[c, 1].astype('int')+2] = np.mean(x[evt, i, z_index.values[neighbor_index, 0].astype('int'), z_index.values[neighbor_index, 1].astype('int')])
    return x

def dump_neighbor_pixels(outf):
    z_index = pd.read_csv("oversample_coordinates.csv")
    z_index = z_index.drop(['x2', 'y2'], axis=1)
    chans = range(500)
    neighbor_indices = []
    #df = pd.DataFrame(np.zeros((500, 2)), columns=['chanID', 'neighborID'])
    for c in chans:
        neighbor_index = np.where((abs(z_index.values[:,0] - z_index.values[c, 0])< 3) & (abs(z_index.values[:,1] - z_index.values[c,1]) < 3) & (abs(z_index.values[:,0] - z_index.values[c, 0])+abs(z_index.values[:,1] - z_index.values[c,1])>0))
        neighbor_indices.append(np.array(neighbor_index[0]))
        #f.iloc[c] = [c, neighbor_index]
    df = pd.DataFrame({'chanID' : chans, 'neighborID' : neighbor_indices},index=chans)
    df.to_csv(outf, index=False)


def read_st4_evt_coord(f, tels=[0,1,2,3], start_event=None, stop_event=None, evtlist=None, outfile=None, getEventNumbers=True):
    io = ROOT.VARootIO(f, 1)
    showerTree = io.loadTheShowerEventTree()
    showerEvtData = ROOT.VAShowerData()
    showerTree.SetBranchAddress("S", showerEvtData)

    paramTree = io.loadTheParameterisedEventTree()
    paramEvtData = ROOT.VAParameterisedEventData()
    paramTree.SetBranchAddress("P", paramEvtData)
    #calibTree.SetBranchAddress("C", calibEvtData)

    #evtNum = []
    if start_event is None:
        start_event=0
    if stop_event is None and evtlist is None:
        totalEvtNum = paramTree.GetEntries()
        print "You want to get coordinates of all events."
    elif evtlist is None:
        assert start_event<stop_event, "Please specify sensible start_event and stop_event numbers. "
        totalEvtNum = stop_event+1-start_event
        evtlist = range(start_event, stop_event+1)
    else:
        totalEvtNum = len(evtlist)

    print("Processing %d events." % totalEvtNum)

    try:
        allCoords = np.zeros((totalEvtNum, 2)) #first [0] RA then [1] DEC
    except MemoryError:
        print("Such a large number of events caused a MemoryError... "
              "Let's try passing start_event and stop_event or evtlist to analyze a smaller set of events.")
        raise

    ReconstructedEvents = []
    for evt_count, evt in enumerate(evtlist):
        try:
            #paramTree.GetEntry(evt)
            showerTree.GetEntry(evt)
        except:
            print("Can't get shower tree event number %d" % evt)
        #evtNum.append(int(calibEvtData.fArrayEventNum))
        #for telID in tels:
        try:
            #allCoords[evt_count] = (paramEvtData.vfTels.at(telID).pfHillasData.fOriginRA, paramEvtData.vfTels.at(telID).pfHillasData.fOriginDec)
            allCoords[evt_count] = (showerEvtData.fDirectionRA_J2000_Rad, showerEvtData.fDirectionDec_J2000_Rad)
            if showerEvtData.fIsReconstructed:
                ReconstructedEvents.append(evt)
        except:
            print "Event ", evt, " failed to get coordinates "
            allCoords[evt_count]=(0., 0.)
    if outfile is not None:
        output = open(outfile, 'wb')
        pickle.dump(allCoords, output)
        output.close()
        #pd.DataFrame(allCharge).to_csv(outfile, index=False, header=None)
    #if getEventNumbers:
    #    return allCoords, evtNums
    return allCoords

def read_st4_evt_params(f, tels=[0,1,2,3], start_event=None, stop_event=None, evtlist=None, outfile=None,
                        isEnergy=True, getCoordinates=False, getParams=True, getEventNumbers=True, return_nonZero=False):
    io = ROOT.VARootIO(f, 1)
    showerTree = io.loadTheShowerEventTree()
    showerEvtData = ROOT.VAShowerData()
    showerTree.SetBranchAddress("S", showerEvtData)

    #paramTree = io.loadTheParameterisedEventTree()
    #paramEvtData = ROOT.VAParameterisedEventData()
    #paramTree.SetBranchAddress("P", paramEvtData)
    #calibTree.SetBranchAddress("C", calibEvtData)

    rh = io.loadTheRunHeader()
    sourceRA = rh.getSourceRA()
    sourceDec = rh.getSourceDec()

    #evtNum = []
    if start_event is None:
        start_event=0
    if stop_event is None and evtlist is None:
        totalEvtNum = showerTree.GetEntries()
        print "You want to get coordinates of all events."
    elif evtlist is None:
        assert start_event<stop_event, "Please specify sensible start_event and stop_event numbers. "
        totalEvtNum = stop_event+1-start_event
        evtlist = range(start_event, stop_event+1)
    else:
        totalEvtNum = len(evtlist)

    print("Processing %d events." % totalEvtNum)

    try:
        #Parameters of interest are: S.fMSW, S.fMSL, S.fMSW_RMS, S.fMSL_RMS, S.fShowerMaxHeight_KM, S.fEnergyRMS_GeV/S.fEnergy_GeV
        #maybes: impact distance (S.fCoreXEastMirrPlane_M, fCoreYNorthMirrPlane_M), S.fShowerMaxHeight_RMS_KM
        if getParams:
            allParams = np.zeros((totalEvtNum, 6))
        if getEventNumbers:
            evtNums = np.zeros(totalEvtNum)
        if getCoordinates:
            allCoords = np.zeros((totalEvtNum, 2)) #first [0] RA then [1] DEC
    except MemoryError:
        print("Such a large number of events caused a MemoryError... "
              "Let's try passing start_event and stop_event or evtlist to analyze a smaller set of events.")
        raise

    ReconstructedEvents = []
    for evt_count, evt in enumerate(evtlist):
        try:
            #paramTree.GetEntry(evt)
            showerTree.GetEntry(evt)
        except:
            print("Can't get shower tree event number %d" % evt)
        #evtNum.append(int(calibEvtData.fArrayEventNum))
        #for telID in tels:
        try:
            if showerEvtData.fEnergy_GeV > 0:
                Erms_over_E = showerEvtData.fEnergyRMS_GeV/showerEvtData.fEnergy_GeV
            else:
                if isEnergy:
                    continue
                else:
                    Erms_over_E = 2.
            if getParams:
                allParams[evt_count] = (showerEvtData.fMSW, showerEvtData.fMSL, showerEvtData.fMSW_RMS,
                                    showerEvtData.fMSL_RMS, showerEvtData.fShowerMaxHeight_KM,
                                    Erms_over_E)
            if getEventNumbers:
                evtNums[evt_count] = showerEvtData.fArrayEventNum
            if getCoordinates:
                allCoords[evt_count] = (showerEvtData.fDirectionRA_J2000_Rad, showerEvtData.fDirectionDec_J2000_Rad)
            if showerEvtData.fIsReconstructed:
                ReconstructedEvents.append(evt)
        except:
            print "Event ", evt, "ArrayEvtNum ", showerEvtData.fArrayEventNum, " failed to get params "
            if getParams:
                allParams[evt_count] = 0.
            if getCoordinates:
                allCoords[evt_count]=(0., 0.)
    if return_nonZero:
        if getParams:
            allParams = allParams[np.where(evtNums!=0)]
        if getEventNumbers:
            evtNums = evtNums[np.where(evtNums!=0)]
        if getCoordinates:
            allCoords = allCoords[np.where(evtNums!=0)]

    if outfile is not None:
        if getParams:
            output = open(outfile, 'wb')
            pickle.dump(allParams, output)
            output.close()
        if getEventNumbers:
            output = open("ArrayEvtNum_"+outfile, 'wb')
            pickle.dump(allParams, output)
            output.close()
        if getCoordinates:
            output = open("Coordinates_"+outfile, 'wb')
            pickle.dump(allCoords, output)
            output.close()
        #pd.DataFrame(allCharge).to_csv(outfile, index=False, header=None)
    if getParams and not getEventNumbers and not getCoordinates:
        return allParams
    elif getParams and getEventNumbers and not getCoordinates:
        return allParams, evtNums
    elif getParams and not getEventNumbers and getCoordinates:
        return allParams, allCoords
    elif not getParams and not getEventNumbers and getCoordinates:
        return allCoords
    elif not getParams and getEventNumbers and not getCoordinates:
        return evtNums
    elif not getParams and getEventNumbers and getCoordinates:
        return evtNums, allCoords


def read_st4_sourceCoord(f):
    io = ROOT.VARootIO(f, 1)
    rh = io.loadTheRunHeader()
    sourceRA = rh.getSourceRA()
    sourceDec = rh.getSourceDec()
    return  sourceRA, sourceDec

def view_raw_skymap(allCoords, evtlist):
    plt.hist2d(allCoords[:,0][np.where(allCoords[:,0]>0)], allCoords[:,1][np.where(allCoords[:,1]>0)], bins=40, norm=LogNorm())
    plt.colorbar()
    plt.show()
