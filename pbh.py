__author__ = 'qfeng'
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy import stats
import random

try:
    import ROOT
except:
    print "Can't import ROOT, no related functionality possible"


import time

def deg2rad(deg):
    return deg/180.*np.pi

def rad2deg(rad):
    return rad*180./np.pi


class Pbh(object):
    def __init__(self):
        # the cut on -2lnL, consider smaller values accepted for events coming from the same centroid
        self.ll_cut = -9.5
        # set the hard coded PSF width table from the hyperbolic secant function
        # 4 rows are Energy bins 0.08 to 0.32 TeV (row 0), 0.32 to 0.5 TeV, 0.5 to 1 TeV, and 1 to 50 TeV
        # 3 columns are Elevation bins 50-70 (column 0), 70-80 80-90 degs
        self.psf_lookup = np.zeros((4,3))
        self.E_grid=np.array([0.08, 0.32, 0.5, 1.0, 50.0])
        self.EL_grid=np.array([50.0, 70.0, 80., 90.])
        # for later reference
        #self.E_bins=np.digitize(self.BDT_ErecS, self.E_grid)-1
        #self.Z_bins=np.digitize((90.-self.BDT_Elevation), self.Zen_grid)-1
        #  0.08 to 0.32 TeV
        self.psf_lookup[0,:] = np.array([0.052, 0.051, 0.05])
        #  0.32 to 0.5 TeV
        self.psf_lookup[1,:] = np.array([0.047, 0.042, 0.042])
        #   0.5 to 1 TeV
        self.psf_lookup[2,:] = np.array([0.041, 0.035, 0.034])
        #   1 to 50 TeV
        self.psf_lookup[3,:] = np.array([0.031, 0.028, 0.027])

    #This below is not used as we are dealing with stupid root files
    def read_photon_list(self, ts, RAs, Decs, Es, ELs):
        N_ = len(ts)
        assert N_==len(RAs) and N_==len(Decs) and N_==len(Es) and N_==len(ELs), \
            "Make sure input lists (ts, RAs, Decs, Es, ELs) are of the same dimension"
        columns=['MJDs', 'ts', 'RAs', 'Decs', 'Es', 'ELs', 'coords', 'psfs']
        df_ = pd.DataFrame(np.array([np.zeros(N_)]*len(columns)).T,
                      columns=columns)
        df_.ts = ts
        df_.RAs = RAs
        df_.Decs = Decs
        df_.Es = Es
        df_.ELs = ELs
        df_.coords = np.concatenate([df_.RAs.reshape(N_,1), df_.Decs.reshape(N_,1)], axis=1)
        df_.psfs = np.zeros(N_)
        self.photon_df = df_
        self.photon_df.psfs = self.get_psf_lists()

    def readEDfile(self, runNum=None, filename=None):
        self.runNum = runNum
        self.filename = str(runNum)+".anasum.root"
        if not os.path.isfile(self.filename) and filename is not None:
            if os.path.isfile(filename):
                self.filename = filename
        self.Rfile = ROOT.TFile(self.filename, "read");


    def get_TreeWithAllGamma(self, runNum=None):
        if not hasattr(self, 'Rfile'):
            print "No file has been read."
            if runNum is not None:
                try:
                    self.readEDfile(runNum=runNum)
                    print "Read file "+self.filename+"..."
                except:
                    print "Can't read file with runNum "+str(runNum)
                    raise
            else:
                print "Run self.readEDfile(\"rootfile\") first; or provide a runNum"
                raise
        all_gamma_treeName = "run_"+str(self.runNum)+"/stereo/TreeWithAllGamma"
        #pointingData_treeName = "run_"+str(self.runNum)+"/stereo/pointingDataReduced"
        all_gamma_tree = self.Rfile.Get(all_gamma_treeName);
        #pointingData = self.Rfile.Get(pointingData_treeName);
        #ptTime=[]
        #for ptd in pointingData:
        #    ptTime.append(ptd.Time);
        #ptTime=np.array(ptTime)
        #columns=['runNumber','eventNumber', 'MJD', 'Time', 'Elevation', ]
        columns=['MJDs', 'ts', 'RAs', 'Decs', 'Es', 'ELs', 'coords', 'psfs']
        N_ = all_gamma_tree.GetEntries()
        df_ = pd.DataFrame(np.array([np.zeros(N_)]*len(columns)).T,
                           columns=columns)
        for i, event in enumerate(all_gamma_tree):
            #time_index=np.argmax(ptTime>event.Time)
            # fill the pandas dataframe
            df_.MJDs[i] = event.dayMJD
            #df_.eventNumber[i] = event.eventNumber
            df_.ts[i] = event.timeOfDay
            df_.RAs[i] = event.GammaRA
            df_.Decs[i] = event.GammaDEC
            df_.Es[i] = event.Energy
            df_.ELs[i] = event.TelElevation

        df_.coords = np.concatenate([df_.RAs.reshape(N_,1), df_.Decs.reshape(N_,1)], axis=1)
        df_.psfs = np.zeros(N_)
        self.photon_df = df_
        self.photon_df.psfs = self.get_psf_lists()

    def scramble(self):
        if not hasattr(self, 'photon_df'):
            print "Call get_TreeWithAllGamma first..."
        self.ts = random.shuffle(self.ts)


    def psf_func(self, theta2, psf_width, N=100):
        return 1.71*N/2./np.pi/psf_width**2/np.cosh(np.sqrt(theta2)/psf_width)

    #use hard coded width table from the hyperbolic secant function
    def get_psf(self, E=0.1, EL=80):
        E_bin=np.digitize(E, self.E_grid)-1
        EL_bin=np.digitize(EL, self.EL_grid)-1
        return self.psf_lookup[E_bin, EL_bin]

    def get_psf_lists(self):
        if not hasattr(self, 'photon_df'):
            print "Call get_TreeWithAllGamma first..."
        for i, EL_ in enumerate(self.photon_df.ELs):
            self.photon_df.psfs.at[i] = self.get_psf(E=self.photon_df.Es[i], EL=EL_)

    def get_angular_distance(self, coord1, coord2):
        """
        coord1 and coord2 are in [ra, dec] format in degrees
        """
        return rad2deg(np.arccos(np.sin(deg2rad(coord1[1]))*np.sin(deg2rad(coord2[1]))
                         + np.cos(deg2rad(coord1[1]))*np.cos(deg2rad(coord2[1]))*
                         np.cos(deg2rad(coord1[0])-deg2rad(coord2[0]))))

    def get_all_angular_distance(self, coords, cent_coord):
        assert coords.shape[1] == 2
        dists = np.zeros(coords.shape[0])
        for i, coord in enumerate(coords):
            dists[i]=self.get_angular_distance(coord, cent_coord)
        return dists

    def centroid_log_likelihood(self, cent_coord, coords, psfs):
        """
        returns ll=-2*ln(L)
        """
        ll = 0
        dists = self.get_all_angular_distance(coords, cent_coord)
        theta2s = dists**2
        ll = -2.*np.sum(np.log(psfs)) + np.sum(np.log(1./np.cosh(np.sqrt(theta2s)/psfs)))
        ll += len(theta2s)*np.log(1.71/np.pi)
        ll = -2.*ll
        return ll

    def minimize_centroid_ll(self, coords, psfs):
        init_centroid = np.mean(coords, axis=0)
        results = minimize(self.centroid_log_likelihood, init_centroid, args=(coords, psfs), method='L-BFGS-B')
        return results.x

    def search_angular_window(self):
        centroid = self.minimize_centroid_ll(self.coords, self.photon_df.psfs)

    #def search_time_window


    def plot_theta2(self, theta2s=np.arange(0, 2, 0.01), psf_width=0.1, N=100, const=1, ax=None, ylog=True):
        const_ = np.ones(theta2s.shape[0])*const
        if ax is None:
            fig=plt.figure()
            ax=plt.subplot(111)
        ax.plot(theta2s, self.psf_func(theta2s, psf_width, N=N), 'r--')
        ax.plot(theta2s, const_, 'b:')
        ax.plot(theta2s, self.psf_func(theta2s, 0.1)+const_, 'k-')
        if ylog:
            ax.set_yscale('log')
        ax.set_xlabel(r'$\theta^2$ (deg$^2$)')
        ax.set_ylabel("Count")
        return ax

    def plot_skymap(self, coords, Es, ELs, ax=None, color='r'):
        if ax is None:
            fig=plt.figure(figsize=(5,5))
            ax=plt.subplot(111)
        ax.plot(coords[:,0], coords[:,1], color+'.')
        for coor, E_, EL_ in zip(coords, Es, ELs):
            circ=plt.Circle(coor, radius=self.get_psf(E_, EL_), color=color, fill=False)
            ax.add_patch(circ)

        ax.set_xlabel('RA')
        ax.set_ylabel("Dec")
        return ax


def test1():
    pbh = Pbh()
    ras = np.random.random(size=10)*2.0+180.
    decs = np.random.random(size=10)*1.5+30.
    coords = np.concatenate([ras.reshape(10,1), decs.reshape(10,1)], axis=1)
    psfs = np.ones(10)*0.1
    centroid = pbh.minimize_centroid_ll(coords,psfs)

    print centroid
    print centroid.reshape(1,2)[:,0], centroid.reshape(1,2)[:,1]

    ax = pbh.plot_skymap(coords, [0.1]*10, [0.2]*10)
    pbh.plot_skymap(centroid.reshape(1,2), [0.1], [0.2], ax=ax, color='b')
    plt.show()

def test2():
    pbh = Pbh()
    pbh.get_TreeWithAllGamma(runNum=47717)
    print pbh.photon_df.head()
    return pbh

if __name__ == "__main__":
    pbh = test2()