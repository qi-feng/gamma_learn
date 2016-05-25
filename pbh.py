__author__ = 'qfeng'
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy import stats
import random

import sys

sys.setrecursionlimit(50000)

try:
    import ROOT
except:
    print "Can't import ROOT, no related functionality possible"

import time


def deg2rad(deg):
    return deg / 180. * np.pi


def rad2deg(rad):
    return rad * 180. / np.pi


class Pbh(object):
    def __init__(self):
        # the cut on -2lnL, consider smaller values accepted for events coming from the same centroid
        self.ll_cut = -9.5
        # set the hard coded PSF width table from the hyperbolic secant function
        # 4 rows are Energy bins 0.08 to 0.32 TeV (row 0), 0.32 to 0.5 TeV, 0.5 to 1 TeV, and 1 to 50 TeV
        # 3 columns are Elevation bins 50-70 (column 0), 70-80 80-90 degs
        self.psf_lookup = np.zeros((4, 3))
        self.E_grid = np.array([0.08, 0.32, 0.5, 1.0, 50.0])
        self.EL_grid = np.array([50.0, 70.0, 80., 90.])
        # for later reference
        # self.E_bins=np.digitize(self.BDT_ErecS, self.E_grid)-1
        #self.Z_bins=np.digitize((90.-self.BDT_Elevation), self.Zen_grid)-1
        #  0.08 to 0.32 TeV
        self.psf_lookup[0, :] = np.array([0.052, 0.051, 0.05])
        #  0.32 to 0.5 TeV
        self.psf_lookup[1, :] = np.array([0.047, 0.042, 0.042])
        #   0.5 to 1 TeV
        self.psf_lookup[2, :] = np.array([0.041, 0.035, 0.034])
        #   1 to 50 TeV
        self.psf_lookup[3, :] = np.array([0.031, 0.028, 0.027])
        self._burst_dict = {}  #{"Burst #": [event # in this burst]}, for internal use
        self.VERITAS_deadtime = 0.33e-3  # 0.33ms

    def read_photon_list(self, ts, RAs, Decs, Es, ELs):
        N_ = len(ts)
        assert N_ == len(RAs) and N_ == len(Decs) and N_ == len(Es) and N_ == len(ELs), \
            "Make sure input lists (ts, RAs, Decs, Es, ELs) are of the same dimension"
        columns = ['MJDs', 'ts', 'RAs', 'Decs', 'Es', 'ELs', 'psfs', 'burst_sizes', 'fail_cut']
        df_ = pd.DataFrame(np.array([np.zeros(N_)] * len(columns)).T,
                           columns=columns)
        df_.ts = ts
        df_.RAs = RAs
        df_.Decs = Decs
        df_.Es = Es
        df_.ELs = ELs
        # df_.coords = np.concatenate([df_.RAs.reshape(N_,1), df_.Decs.reshape(N_,1)], axis=1)
        df_.psfs = np.zeros(N_)
        df_.burst_sizes = np.ones(N_)
        #self.photon_df = df_
        #if event.Energy<E_lo_cut or event.Energy>E_hi_cut or event.TelElevation<EL_lo_cut:
        #    df_.fail_cut.at[i] = 1
        #    continue
        df_.fail_cut = np.zeros(N_)
        #clean events that did not pass cut:
        self.photon_df = df_[df_.fail_cut == 0]

        self.get_psf_lists()

    def readEDfile(self, runNum=None, filename=None):
        self.runNum = runNum
        self.filename = str(runNum) + ".anasum.root"
        if not os.path.isfile(self.filename) and filename is not None:
            if os.path.isfile(filename):
                self.filename = filename
        self.Rfile = ROOT.TFile(self.filename, "read");


    def get_TreeWithAllGamma(self, runNum=None, E_lo_cut=0.08, E_hi_cut=50.0, EL_lo_cut=50.0, nlines=None):
        """
        :param runNum:
        :return: nothing but fills photon_df, except photon_df.burst_sizes
        """
        if not hasattr(self, 'Rfile'):
            print "No file has been read."
            if runNum is not None:
                try:
                    self.readEDfile(runNum=runNum)
                    print "Read file " + self.filename + "..."
                except:
                    print "Can't read file with runNum " + str(runNum)
                    raise
            else:
                print "Run self.readEDfile(\"rootfile\") first; or provide a runNum"
                raise
        all_gamma_treeName = "run_" + str(self.runNum) + "/stereo/TreeWithAllGamma"
        # pointingData_treeName = "run_"+str(self.runNum)+"/stereo/pointingDataReduced"
        all_gamma_tree = self.Rfile.Get(all_gamma_treeName);
        #pointingData = self.Rfile.Get(pointingData_treeName);
        #ptTime=[]
        #for ptd in pointingData:
        #    ptTime.append(ptd.Time);
        #ptTime=np.array(ptTime)
        #columns=['runNumber','eventNumber', 'MJD', 'Time', 'Elevation', ]
        columns = ['MJDs', 'ts', 'RAs', 'Decs', 'Es', 'ELs', 'psfs', 'burst_sizes', 'fail_cut']
        if nlines is not None:
            N_ = nlines
        else:
            N_ = all_gamma_tree.GetEntries()
        df_ = pd.DataFrame(np.array([np.zeros(N_)] * len(columns)).T,
                           columns=columns)
        ###QF short breaker:
        breaker = 0
        for i, event in enumerate(all_gamma_tree):
            if nlines is not None:
                if breaker >= nlines:
                    break
                breaker += 1
            #time_index=np.argmax(ptTime>event.Time)
            #making cut:
            if event.Energy < E_lo_cut or event.Energy > E_hi_cut or event.TelElevation < EL_lo_cut:
                df_.fail_cut.at[i] = 1
                continue
            # fill the pandas dataframe
            df_.MJDs[i] = event.dayMJD
            #df_.eventNumber[i] = event.eventNumber
            df_.ts[i] = event.timeOfDay
            df_.RAs[i] = event.GammaRA
            df_.Decs[i] = event.GammaDEC
            df_.Es[i] = event.Energy
            df_.ELs[i] = event.TelElevation

        #df_.coords = np.concatenate([df_.RAs.reshape(N_,1), df_.Decs.reshape(N_,1)], axis=1)
        df_.psfs = np.zeros(N_)
        # by def all events are at least a singlet
        df_.burst_sizes = np.ones(N_)
        #self.photon_df = df_
        #clean events that did not pass cut:
        self.photon_df = df_[df_.fail_cut == 0]
        #reindexing
        self.photon_df.index = range(self.photon_df.shape[0])

        self.get_psf_lists()

        #If
        #df = df[df.line_race.notnull()]

        ###QF
        #print self.photon_df.head()


    def scramble(self, copy=False):
        if not hasattr(self, 'photon_df'):
            print "Call get_TreeWithAllGamma first..."
        if copy:
            # if you want to keep the original burst_dict, this should only happen at the 1st scramble
            if not hasattr(self, 'photon_df_orig'):
                self.photon_df_orig = self.photon_df.copy()
        ts_ = self.photon_df.ts.values
        random.shuffle(ts_)
        self.photon_df.at[:, 'ts'] = ts_
        # re-init _burst_dict for counting
        self._burst_dict = {}
        # print self.photon_df.head()
        #print self.photon_df.ts.shape, self.photon_df.ts
        return self.photon_df.ts

    def t_rando(self, copy=False):
        """
        throw Poisson distr. ts based on the original ts,
        use 1/delta_t as the expected Poisson rate for each event
        """
        if not hasattr(self, 'photon_df'):
            print "Call get_TreeWithAllGamma first..."
        if copy:
            # if you want to keep the original burst_dict, this should only happen at the 1st scramble
            if not hasattr(self, 'photon_df_orig'):
                self.photon_df_orig = self.photon_df.copy()
        delta_ts = np.diff(self.photon_df.ts)
        for i, _delta_t in enumerate(delta_ts):
            # draw a rando!
            _rando_delta_t = np.random.exponential(1. / _delta_t)
            inf_loop_preventer = 0
            inf_loop_bound = 100
            while _rando_delta_t < self.VERITAS_deadtime:
                _rando_delta_t = np.random.exponential(1. / _delta_t)
                inf_loop_preventer += 1
                if inf_loop_preventer > inf_loop_bound:
                    print "Tried 100 times and can't draw a rando wait time that's larger than VERITAS deadtime,"
                    print "you'd better check your time unit or something..."
            self.photon_df.ts.at[i + 1] = self.photon_df.ts[i] + _rando_delta_t
        return self.photon_df.ts

    def psf_func(self, theta2, psf_width, N=100):
        return 1.71 * N / 2. / np.pi / psf_width ** 2 / np.cosh(np.sqrt(theta2) / psf_width)
        # equivalently:
        #return (stats.hypsecant.pdf(np.sqrt(theta2s)/psf_width)*1.71/2./psf_width**2)

    def psf_cdf(self, psf_width, fov=1.75):
        """
        :param psf_width: same as psf_func
        :param fov: given so that we calculate cdf from 0 to fov
        :return:
        """
        theta2s = np.arange(0, fov, 0.01)
        cdf = np.cumsum(self.psf_func(theta2s, psf_width, N=1))
        cdf = cdf / np.max(cdf)
        return cdf

    # use hard coded width table from the hyperbolic secant function
    def get_psf(self, E=0.1, EL=80):
        E_bin = np.digitize(E, self.E_grid) - 1
        EL_bin = np.digitize(EL, self.EL_grid) - 1
        return self.psf_lookup[E_bin, EL_bin]

    def get_psf_lists(self):
        """
        This thing is slow...
        :return: nothing but filles photon_df.psfs, a number that is repeatedly used later
        """
        if not hasattr(self, 'photon_df'):
            print "Call get_TreeWithAllGamma first..."
        ###QF:
        print "getting psf"
        for i, EL_ in enumerate(self.photon_df.ELs.values):
            #self.photon_df.psfs.at[i] = self.get_psf(E=self.photon_df.loc[i, 'Es'], EL=EL_)
            self.photon_df.at[i, 'psfs'] = self.get_psf(E=self.photon_df.at[i, 'Es'], EL=EL_)
            #if i%10000==0:
            #    print i, "events got psfs"
            #    print self.photon_df.at[i, 'Es'], EL_
            #    print self.photon_df.psfs[i]
            #if self.photon_df.psfs[i] is None:
            #    print "Got a None psf, energy is ", self.photon_df.at[i, 'Es'], "EL is ", EL_
            #print "PSF,", self.photon_df.psfs.at[i]

    def get_angular_distance(self, coord1, coord2):
        """
        coord1 and coord2 are in [ra, dec] format in degrees
        """
        return rad2deg(np.arccos(np.sin(deg2rad(coord1[1])) * np.sin(deg2rad(coord2[1]))
                                 + np.cos(deg2rad(coord1[1])) * np.cos(deg2rad(coord2[1])) *
                                 np.cos(deg2rad(coord1[0]) - deg2rad(coord2[0]))))

    def get_all_angular_distance(self, coords, cent_coord):
        assert coords.shape[1] == 2
        dists = np.zeros(coords.shape[0])
        for i, coord in enumerate(coords):
            dists[i] = self.get_angular_distance(coord, cent_coord)
        return dists

    def gen_one_random_coords(self, cent_coord, theta):
        """
        *** Here use small angle approx, as it is only a sanity check ***
        :return a pair of uniformly random RA and Dec at theta deg away from the cent_coord
        """
        _phi = np.random.random() * np.pi * 2.
        _ra = cent_coord[0] + np.sin(_phi) * theta
        _dec = cent_coord[1] + np.cos(_phi) * theta
        return np.array([_ra, _dec])

    def gen_one_random_theta(self, psf_width, prob="psf", fov=1.75):
        """
        :prob can be either "psf" that uses the hyper-sec function, or "uniform", or "gauss"
        """
        if prob.lower() == "psf" or prob == "hypersec" or prob == "hyper secant":
            #_rand_theta = np.random.random()*fov
            _rand_test_cdf = np.random.random()
            _thetas = np.arange(0, fov, 0.01)
            _theta2s = _thetas ** 2
            #_theta2s=np.arange(0, fov*fov, 0.0001732)
            _psf_pdf = self.psf_func(_theta2s, psf_width, N=1)
            _cdf = np.cumsum(_psf_pdf - np.min(_psf_pdf))
            _cdf = _cdf / np.max(_cdf)
            #y_interp = np.interp(x_interp, x, y)
            _theta2 = np.interp(_rand_test_cdf, _cdf, _theta2s)
            return np.sqrt(_theta2)
        elif prob.lower() == "uniform" or prob == "uni":
            return np.random.random() * fov
        #gauss may have a caveat as this is not important
        elif prob.lower() == "gauss" or prob == "norm" or prob.lower() == "gaussian":
            return abs(np.random.normal()) * fov
        else:
            return "Input prob value not supported"


    def centroid_log_likelihood(self, cent_coord, coords, psfs):
        """
        returns ll=-2*ln(L)
        """
        ll = 0
        dists = self.get_all_angular_distance(coords, cent_coord)
        theta2s = dists ** 2
        ll = -2. * np.sum(np.log(psfs)) + np.sum(np.log(1. / np.cosh(np.sqrt(theta2s) / psfs)))
        ll += psfs.shape[0] * np.log(1.71 / np.pi)
        ll = -2. * ll
        #return ll
        #Normalized by the number of events!
        return ll / psfs.shape[0]

    def minimize_centroid_ll(self, coords, psfs):
        init_centroid = np.mean(coords, axis=0)
        results = minimize(self.centroid_log_likelihood, init_centroid, args=(coords, psfs), method='L-BFGS-B')
        centroid = results.x
        ll_centroid = self.centroid_log_likelihood(centroid, coords, psfs)
        return centroid, ll_centroid


    def search_angular_window(self, coords, psfs, slice_index):
        # Determine if N_evt = coords.shape[0] events are accepted to come from one direction
        # slice_index is the numpy array slice of the input event numbers, used for _burst_dict
        # return: A) centroid, likelihood, and a list of event numbers associated with this burst,
        #            given that a burst is found, or the input has only one event
        #         B) centroid, likelihood, a list of event numbers excluding the outlier, the outlier event number
        #            given that we reject the hypothesis of a burst
        ###QF
        #print coords, slice_index
        assert coords.shape[0] == slice_index.shape[0], "coords shape " + coords.shape[0] + " and slice_index shape " + \
                                                        slice_index.shape[0] + " are different"
        if slice_index.shape[0] == 0:
            #empty
            return None, None, None, None
        if slice_index.shape[0] == 1:
            #one event:
            return coords, 1, np.array([1])
        centroid, ll_centroid = self.minimize_centroid_ll(coords, psfs)
        if ll_centroid <= self.ll_cut:
            # likelihood passes cut
            # all events with slice_index form a burst candidate
            return centroid, ll_centroid, slice_index
        else:
            # if not accepted, find the worst offender and
            # return the better N_evt-1 events and the outlier event
            dists = self.get_all_angular_distance(coords, centroid)
            outlier_index = np.argmax(dists)
            mask = np.ones(len(dists), dtype=bool)
            mask[outlier_index] = False
            #better_coords, better_psfs, outlier_coords, outlier_psfs = coords[mask,:], psfs[mask],\
            #                                                           coords[outlier_index,:], psfs[outlier_index]

            #better_centroid, better_ll_centroid, better_burst_sizes = self.search_angular_window(better_coords, better_psfs)

            #return centroid, ll_centroid, coords[mask,:], psfs[mask], coords[outlier_index,:], psfs[outlier_index]
            #return centroid, ll_centroid, slice_index[mask], slice_index[outlier_index]
            ###QF
            #print "mask", mask
            #print "outlier", outlier_index
            #print "slice_index", slice_index, type(slice_index)
            #print "search_angular_window returning better events", slice_index[mask]
            #print "returning outlier events", slice_index[outlier_index]
            return centroid, ll_centroid, slice_index[mask], slice_index[outlier_index]

    def search_time_window(self, window_size=1):
        """
        Start a burst search for the given window_size in photon_df
        _burst_dict needs to be clean for a new scramble
        :param window_size: in the unit of second
        :return: burst_hist, in the process 1) fill self._burst_dict, and
                                            2) fill self.photon_df.burst_sizes through burst counting; and
                                            3) fill self.photon_df.burst_sizes
        """
        assert hasattr(self,
                       'photon_df'), "photon_df doesn't exist, read data first (read_photon_list or get_TreeWithAllGamma)"
        if len(self._burst_dict) != 0:
            print "You started a burst search while there are already things in _burst_dict,  "
        # Master event loop:
        for t in self.photon_df.ts:
            slice_index = np.where((self.photon_df.ts >= t) & (self.photon_df.ts < (t + window_size)))
            _N = self.photon_df.ts.values[slice_index].shape[0]
            if _N < 1:
                print "Should never happen"
                raise
            if _N == 1:
                #a sparse window
                #self.photon_df.burst_sizes[slice_index] = 1
                #print "L367", slice_index
                #self.photon_df.at[slice_index[0], 'burst_sizes'] = 1
                continue
            burst_events, outlier_events = self.search_event_slice(np.array(slice_index[0]))
            if outlier_events is None:
                #All events of slice_index form a burst, no outliers; or all events are singlet
                continue
            #elif len(outlier_events)==1:
            elif outlier_events.shape[0] == 1:
                #A singlet outlier
                #self.photon_df.burst_sizes[outlier_events[0]] = 1
                #print "L378", outlier_events, outlier_events[0]
                #self.photon_df.at[outlier_events[0], 'burst_sizes'] = 1
                continue
            else:
                #If there is a burst of a subset of events, it's been taken care of, now take care of the outlier slice
                outlier_burst_events, outlier_of_outlier_events = self.search_event_slice(outlier_events)

                while outlier_of_outlier_events is not None:
                    ###QF
                    #print "loop through the outliers "
                    #loop until no outliers are left unprocessed
                    if len(outlier_of_outlier_events) <= 1:
                        #self.photon_df.burst_sizes[outlier_of_outlier_events[0]] = 1
                        outlier_of_outlier_events = None
                        break
                    else:
                        # more than 1 outliers to process,
                        # update outlier_of_outlier_events and repeat the while loop
                        outlier_burst_events, outlier_of_outlier_events = self.search_event_slice(
                            outlier_of_outlier_events)
        # the end of master event loop, self._burst_dict is filled
        # now count bursts and fill self.photon_df.burst_sizes:
        self.burst_counting()
        burst_hist = self.get_burst_hist()
        self.sig_burst_hist = burst_hist
        #return self.photon_df.burst_sizes
        return burst_hist

    def singlet_remover(self, slice_index):
        """
        :param slice_index: a np array of events' indices in photon_df
        :return: new slice_index with singlets (no neighbors in a radius of 5*psf) removed;
                 return None if all events are singlets
        """
        if slice_index.shape[0] == 1:
            #one event, singlet by definition:
            #return Nones
            #self.photon_df.at[slice_index[0], 'burst_sizes'] = 1
            return None
        N_ = self.photon_df.shape[0]
        slice_tuple = tuple(slice_index[:, np.newaxis].T)
        coord_slice = np.concatenate([self.photon_df.RAs.reshape(N_, 1), self.photon_df.Decs.reshape(N_, 1)], axis=1)[
            slice_tuple]
        psf_slice = self.photon_df.psfs.values[slice_tuple]
        #default all events are singlet
        mask_ = np.zeros(slice_index.shape[0], dtype=bool)
        #use a dict of {event_num:neighbor_found} to avoid redundancy
        none_singlet_dict = {}
        for i in range(slice_index.shape[0]):
            if slice_index[i] in none_singlet_dict:
                #already knew not a singlet
                continue
            else:
                psf_5 = psf_slice[i] * 5.0
                for j in range(slice_index.shape[0]):
                    if j == i:
                        #self
                        continue
                    elif self.get_angular_distance(coord_slice[i], coord_slice[j]) < psf_5:
                        #decide this pair isn't singlet
                        none_singlet_dict[slice_index[i]] = slice_index[j]
                        none_singlet_dict[slice_index[j]] = slice_index[j]
                        mask_[i] = True
                        mask_[j] = True
                        continue
        return slice_index[mask_]

    def search_event_slice(self, slice_index):
        """
        _burst_dict needs to be clean before starting a new scramble
        :param slice_index: np array of indices of the events in photon_df that the burst search is carried out upon
        :return: np array of indices of events that are in a burst, indices of outliers (None if no outliers);
                 in the process fill self._burst_dict for later burst counting
        """
        N_ = self.photon_df.shape[0]
        ###QF
        #print "Slice"
        #print slice_index
        #print "Type"
        #print type(slice_index)
        #print "tuple Slice"
        #print tuple(slice_index)
        #print "length", len(tuple(np.array(slice_index)[:,np.newaxis].T))
        #print "Coords"
        #print "Shape"
        #print np.concatenate([self.photon_df.RAs.reshape(N_,1), self.photon_df.Decs.reshape(N_,1)], axis=1).shape
        #print np.concatenate([self.photon_df.RAs.values.reshape(N_,1), self.photon_df.Decs.values.reshape(N_,1)], axis=1)[tuple(slice_index[:,np.newaxis].T)]
        #print "PSFs"
        #print self.photon_df.psfs.values[tuple(slice_index[:,np.newaxis].T)]

        #First remove singlet
        slice_index = self.singlet_remover(slice_index)
        #print slice_index
        if slice_index.shape[0] == 0:
            #all singlets, no bursts, and don't need to check for outliers, go to next event
            return None, None

        ang_search_res = self.search_angular_window(
            np.concatenate([self.photon_df.RAs.reshape(N_, 1), self.photon_df.Decs.reshape(N_, 1)], axis=1)[
                tuple(slice_index[:, np.newaxis].T)], self.photon_df.psfs.values[tuple(slice_index[:, np.newaxis].T)],
            slice_index)
        outlier_evts = []

        if len(ang_search_res) == 3:
            # All events with slice_index form 1 burst
            centroid, ll_centroid, burst_events = ang_search_res
            self._burst_dict[len(self._burst_dict) + 1] = burst_events
            #count later
            #self.photon_df.burst_sizes[slice_index] = len(burst_events)
            #burst_events should be the same as slice_index
            return burst_events, None
        else:
            while (len(ang_search_res) == 4):
                # returned 4 meaning no bursts, and the input has more than one events, shall continue
                # this loop breaks when a burst is found or only one event is left, in which case return values has a length of 3
                better_centroid, better_ll_centroid, _better_events, _outlier_events = ang_search_res
                outlier_evts.append(_outlier_events)
                ###QF
                #print tuple(_better_events), _better_events
                #better_coords = np.concatenate([self.photon_df.RAs.reshape(N_,1), self.photon_df.Decs.reshape(N_,1)], axis=1)[tuple(_better_events)]
                better_coords = \
                    np.concatenate([self.photon_df.RAs.reshape(N_, 1), self.photon_df.Decs.reshape(N_, 1)], axis=1)[
                        (_better_events)]
                #print "in search_event_slice, candidate coords and psfs: ", better_coords, self.photon_df.psfs.values[(_better_events)]
                ang_search_res = self.search_angular_window(better_coords,
                                                            self.photon_df.psfs.values[(_better_events)],
                                                            _better_events)
            # Now that while loop broke, we have a good list and a bad list
            centroid, ll_centroid, burst_events = ang_search_res
            if burst_events.shape[0] == 1:
                # No burst in slice_index found
                #count later
                #self.photon_df.burst_sizes[burst_events[0]] = 1
                return burst_events, np.array(outlier_evts)
            else:
                # A burst with a subset of events of slice_index is found
                self._burst_dict[len(self._burst_dict) + 1] = burst_events
                #self.photon_df.burst_sizes[tuple(burst_events)] = len(burst_events)
                return burst_events, np.array(outlier_evts)


    def duplicate_burst_dict(self):
        #if you want to keep the original burst_dict
        self.burst_dict = self._burst_dict.copy()
        return self.burst_dict

    def burst_counting(self):
        """
        :return: nothing but fills self.photon_df.burst_sizes
        """
        # Only to be called after self._burst_dict is filled
        # Find the largest burst
        largest_burst_number = max(self._burst_dict, key=lambda x: len(set(self._burst_dict[x])))
        for evt in self._burst_dict[largest_burst_number]:
            # Assign burst size to all events in the largest burst
            self.photon_df.burst_sizes[evt] = self._burst_dict[largest_burst_number].shape[0]
            #self.photon_df.burst_sizes[evt] = len(self._burst_dict[largest_burst_number])
            for key in self._burst_dict.keys():
                # Now delete the assigned events in all other candiate bursts to avoid double counting
                if evt in self._burst_dict[key] and key != largest_burst_number:
                    #self._burst_dict[key].remove(evt)
                    self._burst_dict[key] = np.delete(self._burst_dict[key], np.where(self._burst_dict[key] == evt))

        # Delete the largest burst, which is processed above
        self._burst_dict.pop(largest_burst_number, None)
        # repeat while there are unprocessed bursts in _burst_dict
        if len(self._burst_dict) >= 1:
            self.burst_counting()

    def get_burst_hist(self):
        burst_hist = {}
        for i in np.unique(self.photon_df.burst_sizes.values):
            burst_hist[i] = np.sum(self.photon_df.burst_sizes.values == i) / i
        return burst_hist

    def estimate_bkg_burst(self, window_size=1, method="scramble", copy=True):
        """
        :param method: either "scramble" or "rando"
        :return:
        """
        #Note that from now on we are CHANGING the photon_df!
        if method == "scramble":
            self.scramble(copy=copy)
        elif method == "rando":
            self.t_rando(copy=copy)
        bkg_burst_hist = self.search_time_window(window_size=window_size)
        self.bkg_burst_hist = bkg_burst_hist
        return bkg_burst_hist


    def plot_theta2(self, theta2s=np.arange(0, 2, 0.01), psf_width=0.1, N=100, const=1, ax=None, ylog=True):
        const_ = np.ones(theta2s.shape[0]) * const
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111)
        ax.plot(theta2s, self.psf_func(theta2s, psf_width, N=N), 'r--')
        ax.plot(theta2s, const_, 'b:')
        ax.plot(theta2s, self.psf_func(theta2s, psf_width) + const_, 'k-')
        if ylog:
            ax.set_yscale('log')
        ax.set_xlabel(r'$\theta^2$ (deg$^2$)')
        ax.set_ylabel("Count")
        return ax

    def plot_skymap(self, coords, Es, ELs, ax=None, color='r', fov_center=None, fov=1.75, fov_color='gray',
                    cent_coords=None, cent_marker='+', cent_ms=1.8, cent_mew=4.0, cent_radius=0.01, cent_color='b',
                    label=None):
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = plt.subplot(111)
        ax.plot(coords[:, 0], coords[:, 1], color + '.')
        label_flag = False
        for coor, E_, EL_ in zip(coords, Es, ELs):
            if label_flag == False:
                circ = plt.Circle(coor, radius=self.get_psf(E_, EL_), color=color, fill=False, label=label)
                label_flag = True
            else:
                circ = plt.Circle(coor, radius=self.get_psf(E_, EL_), color=color, fill=False)
            ax.add_patch(circ)

        label_flag = False
        if fov is not None and fov_center is not None:
            circ_fov = plt.Circle(fov_center, radius=fov, color=fov_color, fill=False)
            ax.add_patch(circ_fov)
            ax.set_xlim(fov_center[0] - fov * 1.1, fov_center[0] + fov * 1.1)
            ax.set_ylim(fov_center[1] - fov * 1.1, fov_center[1] + fov * 1.1)
        if cent_coords is not None:
            #circ_cent=plt.Circle(cent_coords, radius=cent_radius, color=cent_color, fill=False)
            #ax.add_patch(circ_cent)
            ax.plot(cent_coords[0], cent_coords[1], marker=cent_marker, ms=cent_ms, markeredgewidth=cent_mew,
                    color=color)

        plt.legend(loc='best')
        ax.set_xlabel('RA')
        ax.set_ylabel("Dec")
        return ax


class powerlaw:
    # Class can calculate a power-law pdf, cdf from x_min to x_max,
    # always normalized so that the integral pdf is 1
    # (so is the diff in cdf between x_max and xmin)
    # ###########################################################################
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
        if self.a == -1:
            return -1
        self.pdf_norm = (self.a + 1) / (self.x_max ** (self.a + 1.0) - self.x_min ** (self.a + 1.0))
        return self.pdf_norm * x ** self.a

    def cdf(self, x):
        if self.a == -1:
            return -1
        self.cdf_norm = 1. / (self.x_max ** (self.a + 1.0) - self.x_min ** (self.a + 1.0))
        return self.cdf_norm * (x ** (self.a + 1.0) - self.x_min ** (self.a + 1.0))

    def ppf(self, q):
        if self.a == -1:
            return -1
        norm = (self.x_max ** (self.a + 1.0) - self.x_min ** (self.a + 1.0))
        return (q * norm * 1.0 + self.x_min ** (self.a + 1.0)) ** (1.0 / (self.a + 1.0))

    def random(self, n):
        r_uniform = np.random.random_sample(n)
        return self.ppf(r_uniform)


def test_psf_func(Nburst=10, filename=None, cent_ms=8.0, cent_mew=1.8):
    # Nburst: Burst size to visualize
    pbh = Pbh()
    fov_center = np.array([180., 30.0])
    fov = 1.75

    #spec sim:
    index = -2.5
    E_min = 0.08
    E_max = 50.0
    EL = 15
    pl_nu = powerlaw(index, E_min, E_max)
    rand_Es = pl_nu.random(Nburst)
    rand_bkg_coords = np.zeros((Nburst, 2))
    rand_sig_coords = np.zeros((Nburst, 2))
    psfs = np.zeros(Nburst)

    for i in range(Nburst):
        psf_width = pbh.get_psf(rand_Es[i], EL)
        psfs[i] = psf_width
        rand_bkg_theta = pbh.gen_one_random_theta(psf_width, prob="uniform", fov=fov)
        rand_sig_theta = pbh.gen_one_random_theta(psf_width, prob="psf", fov=fov)
        rand_bkg_coords[i, :] = pbh.gen_one_random_coords(fov_center, rand_bkg_theta)
        rand_sig_coords[i, :] = pbh.gen_one_random_coords(fov_center, rand_sig_theta)

    cent_bkg, ll_bkg = pbh.minimize_centroid_ll(rand_bkg_coords, psfs)
    cent_sig, ll_sig = pbh.minimize_centroid_ll(rand_sig_coords, psfs)

    ax = pbh.plot_skymap(rand_bkg_coords, rand_Es, [EL] * Nburst, color='b', fov_center=fov_center,
                         cent_coords=cent_bkg, cent_marker='+', cent_ms=cent_ms, cent_mew=cent_mew,
                         label=("bkg ll=%.2f" % ll_bkg))
    pbh.plot_skymap(rand_sig_coords, rand_Es, [EL] * Nburst, color='r', fov_center=fov_center, ax=ax,
                    cent_coords=cent_sig, cent_ms=cent_ms, cent_mew=cent_mew,
                    label=("sig ll=%.2f" % ll_sig))
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    return pbh


# def test_psf_func_sim(psf_width=0.05, prob="uniform", Nsim=10000, Nbins=40, filename=None, xlim=None):
def test_psf_func_sim(psf_width=0.05, prob="psf", Nsim=10000, Nbins=40, filename=None, xlim=(0, 0.5)):
    #def test_psf_func_sim(psf_width=0.05, prob="psf", Nsim=10000, Nbins=40, filename=None, xlim=None):
    pbh = Pbh()
    fov = 1.75

    # to store the value of a sim signal!
    rand_thetas = []
    for i in range(Nsim):
        rand_thetas.append(pbh.gen_one_random_theta(psf_width, prob=prob, fov=fov))

    rand_theta2s = np.array(rand_thetas)
    rand_theta2s = rand_theta2s ** 2

    theta2s = np.arange(0, fov, 0.01) ** 2

    theta2_hist, theta2_bins, _ = plt.hist(rand_theta2s, bins=Nbins, alpha=0.3, label="Monte Carlo")
    theta2s_analytical = pbh.psf_func(theta2s, psf_width, N=1)

    plt.yscale('log')
    plt.plot(theta2s, theta2s_analytical / theta2s_analytical[0] * theta2_hist[0], 'r--',
             label="Hyperbolic secant function")
    plt.xlim(xlim)
    plt.xlabel(r'$\theta^2$ (deg$^2$)')
    plt.ylabel("Count")
    plt.legend(loc='best')
    if filename is not None:
        plt.savefig(filename)
    plt.show()


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
        rand_Es = pl_nu.random(N_burst)
        rand_bkg_coords = np.zeros((N_burst, 2))
        rand_sig_coords = np.zeros((N_burst, 2))
        psfs = np.zeros(N_burst)

        for i in range(N_burst):
            psf_width = pbh.get_psf(rand_Es[i], EL)
            psfs[i] = psf_width
            rand_bkg_theta = pbh.gen_one_random_theta(psf_width, prob="uniform", fov=fov)
            rand_sig_theta = pbh.gen_one_random_theta(psf_width, prob="psf", fov=fov)
            rand_bkg_coords[i, :] = pbh.gen_one_random_coords(fov_center, rand_bkg_theta)
            rand_sig_coords[i, :] = pbh.gen_one_random_coords(fov_center, rand_sig_theta)

        cent_bkg, ll_bkg = pbh.minimize_centroid_ll(rand_bkg_coords, psfs)
        cent_sig, ll_sig = pbh.minimize_centroid_ll(rand_sig_coords, psfs)
        ll_bkg_all[j] = ll_bkg
        ll_sig_all[j] = ll_sig

    plt.hist(ll_sig_all, bins=sig_bins, color='r', alpha=0.3, label="Burst size " + str(N_burst) + " signal")
    plt.hist(ll_bkg_all, bins=bkg_bins, color='b', alpha=0.3, label="Burst size " + str(N_burst) + " background")
    plt.axvline(x=-9.5, ls="--", lw=0.3)
    plt.legend(loc='best')
    plt.xlabel("Likelihood")
    if ylog:
        plt.yscale('log')
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    return pbh


def test_burst_finding(window_size=3, runNum=55480, nlines=50):
    pbh = Pbh()
    pbh.get_TreeWithAllGamma(runNum=runNum, nlines=nlines)
    #do a small list
    pbh.photon_df = pbh.photon_df[:nlines]
    sig_burst_hist = pbh.search_time_window(window_size=window_size)
    #sig_burst_hist is actually a dictionary
    plotSig=False
    if plotSig:
        plt.figure()
        plt.errorbar(sig_burst_hist.keys(), sig_burst_hist.values(), xerr=0.5, fmt='bs', capthick=0)
        plt.title("Window size " + str(window_size) + "s")
        plt.xlabel("Burst size")
        plt.ylabel("Counts")
        #plt.ylim(0, np.max(sig_burst_hist.values())*1.2)
        #plt.yscale('log')
        plt.savefig("test_burst_finding_histo_signal_window" + str(window_size) + "s.png")
        plt.show()

    #now scramble:
    N_scramble = 3
    bkg_burst_hists = []
    for i in range(N_scramble):
        bkg_burst_hist = pbh.estimate_bkg_burst(window_size=window_size, method="scramble", copy=True)
        bkg_burst_hists.append(bkg_burst_hist)

    #Get all unique keys in bkg_burst_hists[:]
    all_bkg_burst_sizes = set(k for dic in bkg_burst_hists for k in dic.keys())
    #also a dict
    avg_bkg_hist = {}
    #avg_bkg_hist_count = {}
    for key_ in all_bkg_burst_sizes:
        key_ = int(key_)
        for d_ in bkg_burst_hists:
            if key_ in d_:
                if key_ in avg_bkg_hist:
                    avg_bkg_hist[key_] += d_[key_]
                    #avg_bkg_hist_count[key_] += 1
                else:
                    avg_bkg_hist[int(key_)] = d_[key_]
                    #avg_bkg_hist_count[int(key_)] = 1

    for k in avg_bkg_hist.keys():
        #avg_bkg_hist[k] /= avg_bkg_hist_count[k]*1.0
        avg_bkg_hist[k] /= N_scramble*1.0

    plt.figure()
    plt.errorbar(sig_burst_hist.keys(), sig_burst_hist.values(), xerr=0.5, fmt='bs', capthick=0,
                 label="Data " + str(nlines) + " events")
    plt.errorbar(avg_bkg_hist.keys(), avg_bkg_hist.values(), xerr=0.5, fmt='rv', capthick=0,
                 label="Background " + str(nlines) + " events")
    plt.title("Window size " + str(window_size) + "s")
    plt.xlabel("Burst size")
    plt.ylabel("Counts")
    #plt.ylim(0, np.max(sig_burst_hist.values())*1.2)
    #plt.yscale('log')
    plt.legend(loc='best')
    #plt.show()
    plt.savefig("test_burst_finding_histo_signal_bkg_avg_over"+str(N_scramble)+"scrambles_window"+ str(window_size) +".png")


    return pbh


def test1():
    pbh = Pbh()
    fov_center = np.array([180., 30.0])
    ras = np.random.random(size=10) * 2.0 + fov_center[0]
    decs = np.random.random(size=10) * 1.5 + fov_center[1]
    coords = np.concatenate([ras.reshape(10, 1), decs.reshape(10, 1)], axis=1)
    psfs = np.ones(10) * 0.1
    centroid = pbh.minimize_centroid_ll(coords, psfs)

    print centroid
    print centroid.reshape(1, 2)[:, 0], centroid.reshape(1, 2)[:, 1]

    ax = pbh.plot_skymap(coords, [0.1] * 10, [0.2] * 10)
    pbh.plot_skymap(centroid.reshape(1, 2), [0.1], [0.2], ax=ax, color='b', fov_center=fov_center)
    plt.show()


def test_singlet_remover(Nburst=10, filename=None, cent_ms=8.0, cent_mew=1.8):
    pbh = Pbh()
    fov_center = np.array([180., 30.0])
    fov = 1.75

    #spec sim:
    index = -2.5
    E_min = 0.08
    E_max = 50.0
    EL = 75
    pl_nu = powerlaw(index, E_min, E_max)
    rand_Es = pl_nu.random(Nburst)
    rand_bkg_coords = np.zeros((Nburst, 2))
    rand_sig_coords = np.zeros((Nburst, 2))
    psfs = np.zeros(Nburst)

    for i in range(Nburst):
        psf_width = pbh.get_psf(rand_Es[i], EL)
        psfs[i] = psf_width
        rand_bkg_theta = pbh.gen_one_random_theta(psf_width, prob="uniform", fov=fov)
        rand_sig_theta = pbh.gen_one_random_theta(psf_width, prob="psf", fov=fov)
        rand_bkg_coords[i, :] = pbh.gen_one_random_coords(fov_center, rand_bkg_theta)
        rand_sig_coords[i, :] = pbh.gen_one_random_coords(fov_center, rand_sig_theta)

    pbh.read_photon_list(np.arange(10), rand_bkg_coords[:, 0], rand_bkg_coords[:, 1], rand_Es, np.ones(10) * EL)
    slice = np.arange(10)
    slice = pbh.singlet_remover(slice)
    print slice


def test2():
    pbh = Pbh()
    pbh.get_TreeWithAllGamma(runNum=55480, nlines=1000)
    print pbh.photon_df.head()
    return pbh


if __name__ == "__main__":
    #test_singlet_remover()
    pbh = test_burst_finding(window_size=1, runNum=55480)
    #pbh = test_psf_func(Nburst=10, filename=None)

    #pbh = test_psf_func_sim(psf_width=0.05, Nsim=10000, prob="psf", Nbins=40, xlim=(0,0.5),
    #                        filename="/Users/qfeng/Data/veritas/pbh/reedbuck_plots/test_sim_psf/sim_psf_theta2hist_hypsec_sig.pdf")

    #pbh = test_psf_func_sim(psf_width=0.05, prob="uniform", Nsim=10000, Nbins=40, xlim=None,
    #                        filename="/Users/qfeng/Data/veritas/pbh/reedbuck_plots/test_sim_psf/sim_psf_theta2hist_uniform_bkg.pdf")