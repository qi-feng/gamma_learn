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
        #equivalently:
        #return (stats.hypsecant.pdf(np.sqrt(theta2s)/psf_width)*1.71/2./psf_width**2)

    def psf_cdf(self, psf_width, fov=1.75):
        """
        :param psf_width: same as psf_func
        :param fov: given so that we calculate cdf from 0 to fov
        :return:
        """
        theta2s=np.arange(0, fov, 0.01)
        cdf = np.cumsum(self.psf_func(theta2s, psf_width, N=1))
        cdf = cdf/np.max(cdf)
        return cdf

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

    def gen_one_random_coords(self, cent_coord, theta):
        """
        *** Here use small angle approx, as it is only a sanity check ***
        :return a pair of uniformly random RA and Dec at theta deg away from the cent_coord
        """
        _phi = np.random.random()*np.pi*2.
        _ra = cent_coord[0]+np.sin(_phi)*theta
        _dec = cent_coord[1]+np.cos(_phi)*theta
        return np.array([_ra, _dec])

    def gen_one_random_theta(self, psf_width, prob="psf", fov=1.75):
        """
        :prob can be either "psf" that uses the hyper-sec function, or "uniform", or "gauss"
        """
        if prob.lower()=="psf" or prob=="hypersec" or prob=="hyper secant":
            #_rand_theta = np.random.random()*fov
            _rand_test_cdf = np.random.random()
            _theta2s=np.arange(0, fov, 0.01)
            _cdf = np.cumsum(self.psf_func(_theta2s, psf_width, N=1))
            _cdf = _cdf/np.max(_cdf)
            #y_interp = np.interp(x_interp, x, y)
            _theta = np.interp(_rand_test_cdf, _cdf, _theta2s)
            return _theta
        elif prob.lower()=="uniform" or prob=="uni":
            return np.random.random()*fov
        #gauss may have a caveat as this is not important
        elif prob.lower()=="gauss" or prob=="norm" or prob.lower()=="gaussian":
            return abs(np.random.normal())*fov
        else:
            return "Input prob value not supported"

    def centroid_log_likelihood(self, cent_coord, coords, psfs):
        """
        returns ll=-2*ln(L)
        """
        ll = 0
        dists = self.get_all_angular_distance(coords, cent_coord)
        theta2s = dists**2
        ll = -2.*np.sum(np.log(psfs)) + np.sum(np.log(1./np.cosh(np.sqrt(theta2s)/psfs)))
        ll += psfs.shape[0]*np.log(1.71/np.pi)
        ll = -2.*ll
        return ll

    def minimize_centroid_ll(self, coords, psfs):
        init_centroid = np.mean(coords, axis=0)
        results = minimize(self.centroid_log_likelihood, init_centroid, args=(coords, psfs), method='L-BFGS-B')
        centroid = results.x
        ll_centroid = self.centroid_log_likelihood(centroid, coords, psfs)
        return centroid, ll_centroid

    def search_angular_window(self, coords, psfs):
        centroid, ll_centroid = self.minimize_centroid_ll(coords, psfs)
        if ll_centroid < self.ll_cut:
            return centroid, ll_centroid
        else:
            return False, False

    def search_time_window(self, ts=None, RAs=None, Decs=None, Es=None, ELs=None):
        if ts is not None:
            assert RAs is not None
            assert Decs is not None
            assert Es is not None
            assert ELs is not None
        else:
            ts = self.photon_df.ts
            RAs = self.photon_df.RAs
            Decs = self.photon_df.Decs
            Es = self.photon_df.Es
            ELs = self.photon_df.ELs


    def plot_theta2(self, theta2s=np.arange(0, 2, 0.01), psf_width=0.1, N=100, const=1, ax=None, ylog=True):
        const_ = np.ones(theta2s.shape[0])*const
        if ax is None:
            fig=plt.figure()
            ax=plt.subplot(111)
        ax.plot(theta2s, self.psf_func(theta2s, psf_width, N=N), 'r--')
        ax.plot(theta2s, const_, 'b:')
        ax.plot(theta2s, self.psf_func(theta2s, psf_width)+const_, 'k-')
        if ylog:
            ax.set_yscale('log')
        ax.set_xlabel(r'$\theta^2$ (deg$^2$)')
        ax.set_ylabel("Count")
        return ax

    def plot_skymap(self, coords, Es, ELs, ax=None, color='r', fov_center=None, fov=1.75, fov_color='gray',
                    cent_coords=None, cent_marker='+', cent_ms=1.8, cent_mew=4.0, cent_radius=0.01, cent_color='b', label=None):
        if ax is None:
            fig=plt.figure(figsize=(5,5))
            ax=plt.subplot(111)
        ax.plot(coords[:,0], coords[:,1], color+'.')
        label_flag=False
        for coor, E_, EL_ in zip(coords, Es, ELs):
            if label_flag==False:
                circ=plt.Circle(coor, radius=self.get_psf(E_, EL_), color=color, fill=False, label=label)
                label_flag=True
            else:
                circ=plt.Circle(coor, radius=self.get_psf(E_, EL_), color=color, fill=False)
            ax.add_patch(circ)

        label_flag=False
        if fov is not None and fov_center is not None:
            circ_fov=plt.Circle(fov_center, radius=fov, color=fov_color, fill=False)
            ax.add_patch(circ_fov)
            ax.set_xlim(fov_center[0]-fov*1.1, fov_center[0]+fov*1.1)
            ax.set_ylim(fov_center[1]-fov*1.1, fov_center[1]+fov*1.1)
        if cent_coords is not None:
            #circ_cent=plt.Circle(cent_coords, radius=cent_radius, color=cent_color, fill=False)
            #ax.add_patch(circ_cent)
            ax.plot(cent_coords[0], cent_coords[1], marker=cent_marker, ms=cent_ms, markeredgewidth=cent_mew, color=color)

        plt.legend(loc='best')
        ax.set_xlabel('RA')
        ax.set_ylabel("Dec")
        return ax


class powerlaw:
    # Class can calculate a power-law pdf, cdf from x_min to x_max,
    # always normalized so that the integral pdf is 1
    # (so is the diff in cdf between x_max and xmin)
    ############################################################################
    #                       Initialization of the object.                      #
    ############################################################################
    def __init__(self, a, x_min, x_max):
        """
        :param a: photon index, dN/dE = E^a
        :param x_min: e lo
        :param x_max: e hi
        :return:
        """
        self.a = a
        self.x_min = x_min
        self.x_max = x_max

    def pdf(self, x):
        if self.a==-1:
            return -1
        self.pdf_norm = (self.a+1)/(self.x_max**(self.a+1.0) - self.x_min**(self.a+1.0))
        return self.pdf_norm*x**self.a

    def cdf(self, x):
        if self.a==-1:
            return -1
        self.cdf_norm = 1./(self.x_max**(self.a+1.0) - self.x_min**(self.a+1.0))
        return self.cdf_norm*(x**(self.a+1.0) - self.x_min**(self.a+1.0))

    def ppf(self, q):
        if self.a==-1:
            return -1
        norm = (self.x_max**(self.a+1.0) - self.x_min**(self.a+1.0))
        return (q*norm*1.0+self.x_min**(self.a+1.0))**(1.0/(self.a+1.0))

    def random(self, n):
        r_uniform = np.random.random_sample(n)
        return self.ppf(r_uniform)


def test_psf_func(Nburst=10, filename=None, cent_ms=1.8, cent_mew=4.0):
    #Nburst: Burst size to visualize
    pbh = Pbh()
    fov_center = np.array([180., 30.0])
    fov = 1.75

    #spec sim:
    index = -2.5
    E_min = 0.08
    E_max = 50.0
    EL = 15
    pl_nu = powerlaw(index, E_min, E_max)
    rand_Es =  pl_nu.random(Nburst)
    rand_bkg_coords = np.zeros((Nburst, 2))
    rand_sig_coords = np.zeros((Nburst, 2))
    psfs = np.zeros(Nburst)

    for i in range(Nburst):
        psf_width = pbh.get_psf(rand_Es[i], EL)
        psfs[i] = psf_width
        rand_bkg_theta = pbh.gen_one_random_theta(psf_width, prob="uniform", fov=fov)
        rand_sig_theta = pbh.gen_one_random_theta(psf_width, prob="psf", fov=fov)
        rand_bkg_coords[i,:] = pbh.gen_one_random_coords(fov_center, rand_bkg_theta)
        rand_sig_coords[i,:] = pbh.gen_one_random_coords(fov_center, rand_sig_theta)

    cent_bkg, ll_bkg = pbh.minimize_centroid_ll(rand_bkg_coords, psfs)
    cent_sig, ll_sig = pbh.minimize_centroid_ll(rand_sig_coords, psfs)

    ax = pbh.plot_skymap(rand_bkg_coords,rand_Es, [EL]*Nburst, color='b', fov_center=fov_center,
                         cent_coords=cent_bkg, cent_marker='+', cent_ms=cent_ms, cent_mew=cent_mew,
                         label=("bkg ll=%.2f" % ll_bkg))
    pbh.plot_skymap(rand_sig_coords,rand_Es, [EL]*Nburst, color='r', fov_center=fov_center, ax=ax,
                    cent_coords=cent_sig, cent_ms=cent_ms, cent_mew=cent_mew,
                    label=("sig ll=%.2f" % ll_sig))
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    return pbh

def test_sim_likelihood(Nsim=1000, N_burst=3, filename=None, sig_bins=50, bkg_bins=100, ylog=True):
    pbh = Pbh()
    fov_center = np.array([180., 30.0])
    fov = 1.75

    #spec sim:
    index = -2.5
    E_min = 0.08
    E_max = 50.0
    #Burst size to visualize
    #N_burst = 10
    EL = 15
    pl_nu = powerlaw(index, E_min, E_max)

    #Nsim = 1000
    ll_bkg_all = np.zeros(Nsim)
    ll_sig_all = np.zeros(Nsim)

    for j in range(Nsim):
        rand_Es =  pl_nu.random(N_burst)
        rand_bkg_coords = np.zeros((N_burst, 2))
        rand_sig_coords = np.zeros((N_burst, 2))
        psfs = np.zeros(N_burst)

        for i in range(N_burst):
            psf_width = pbh.get_psf(rand_Es[i], EL)
            psfs[i] = psf_width
            rand_bkg_theta = pbh.gen_one_random_theta(psf_width, prob="uniform", fov=fov)
            rand_sig_theta = pbh.gen_one_random_theta(psf_width, prob="psf", fov=fov)
            rand_bkg_coords[i,:] = pbh.gen_one_random_coords(fov_center, rand_bkg_theta)
            rand_sig_coords[i,:] = pbh.gen_one_random_coords(fov_center, rand_sig_theta)

        cent_bkg, ll_bkg = pbh.minimize_centroid_ll(rand_bkg_coords, psfs)
        cent_sig, ll_sig = pbh.minimize_centroid_ll(rand_sig_coords, psfs)
        ll_bkg_all[j] = ll_bkg
        ll_sig_all[j] = ll_sig

    plt.hist(ll_sig_all, bins=sig_bins, color='r', alpha=0.3, label="Burst size "+str(N_burst)+" signal")
    plt.hist(ll_bkg_all, bins=bkg_bins, color='b', alpha=0.3, label="Burst size "+str(N_burst)+" background")
    plt.axvline(x=-9.5, ls="--", lw=0.3)
    plt.legend(loc='best')
    plt.xlabel("Likelihood")
    if ylog:
        plt.yscale('log')
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    return pbh


def test1():
    pbh = Pbh()
    fov_center = np.array([180., 30.0])
    ras = np.random.random(size=10)*2.0+fov_center[0]
    decs = np.random.random(size=10)*1.5+fov_center[1]
    coords = np.concatenate([ras.reshape(10,1), decs.reshape(10,1)], axis=1)
    psfs = np.ones(10)*0.1
    centroid = pbh.minimize_centroid_ll(coords,psfs)

    print centroid
    print centroid.reshape(1,2)[:,0], centroid.reshape(1,2)[:,1]

    ax = pbh.plot_skymap(coords, [0.1]*10, [0.2]*10)
    pbh.plot_skymap(centroid.reshape(1,2), [0.1], [0.2], ax=ax, color='b', fov_center=fov_center)
    plt.show()

def test2():
    pbh = Pbh()
    pbh.get_TreeWithAllGamma(runNum=47717)
    print pbh.photon_df.head()
    return pbh

if __name__ == "__main__":
    pbh = test_psf_func()