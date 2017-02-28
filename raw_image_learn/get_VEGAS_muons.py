from VNN.utils.io import *
from VNN.utils.vegas_io import *
from VNN.utils.squarecam import *
from VNN.utils.image import *
#from PyVAPlotCam import PyVAPlotCam
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import math
import numpy as np

import os.path
import sys

#f = "/raid/biggams/qfeng/data/PG1553/81634UCORmedWinterMuonSt4.root"

def get_muon_charge(f, outfile_base="81635_muon", non_muon_outfile_base="81635_non_muon", num_of_non_muon=5000, dump=True,
                    outdir="muon_hunter_images", dpi=144, load=True, cut_radius=0.0, cut_radius_upper=None,
                    cleaning={'img': 5.0, 'brd': 2.5}):
    if os.path.isfile(outfile_base+"_charge.hdf5") and os.path.isfile(outfile_base+"_evtNum_telID.hdf5") and load:
        print("Loading events")
        m_allCharges = load_hdf5(outfile_base + "_charge.hdf5")
        m_EvtNumsTelIDs = load_hdf5(outfile_base + "_evtNum_telID.hdf5")
        m_evtNums = m_EvtNumsTelIDs[:, 0]
        m_tels = m_EvtNumsTelIDs[:, 1]
    else:
        print("Reading events from root file")
        m_evtNums, m_tels, m_allCharges = read_muon_data(f, tels=[0, 1, 2, 3], read_charge=True, save_muon=True,
                                          cut_radius=cut_radius, cut_radius_upper=cut_radius_upper,
                                          outfile_base=outfile_base, save_non_muon=True, num_of_non_muon=num_of_non_muon,
                                          non_muon_outfile_base=non_muon_outfile_base, cleaning=cleaning)

    nm_allCharges = load_hdf5(non_muon_outfile_base+"_charge.hdf5")
    nm_EvtNumsTelIDs = load_hdf5(non_muon_outfile_base+"_evtNum_telID.hdf5")
    nm_evtNums = nm_EvtNumsTelIDs[:,0]
    nm_tels = nm_EvtNumsTelIDs[:, 1]

    if dump:
        for i in range(m_allCharges.shape[-1]):
            cam = PyVAPlotCam(m_allCharges[:, i], fig=plt.figure(figsize=(7, 7)), ax=plt.subplot(111))
            # cam.buildCamera(draw_pixNum=False)
            # cam.draw(drawColorbar=False, draw_pixNum=False)
            cam.draw(drawColorbar=False, draw_pixNum=False, cm=plt.cm.jet)
            plt.savefig(outdir+"/"+str(outfile_base)+"_evt" + str(int(m_evtNums[i])) + "_tel"+str("{:.0f}".format(m_tels[i]))+".jpeg", dpi=dpi)
            #cam.show()


        for i in range(nm_allCharges.shape[-1]):
            cam = PyVAPlotCam(nm_allCharges[:, i], fig=plt.figure(figsize=(7, 7)), ax=plt.subplot(111))
            cam.draw(drawColorbar=False, draw_pixNum=False, cm=plt.cm.jet)
            plt.savefig(outdir+"/"+str(non_muon_outfile_base)+"_evt" + str(int(nm_evtNums[i])) + "_tel"+str("{:.0f}".format(m_tels[i]))+".jpeg", dpi=dpi)

    return m_evtNums, m_tels, m_allCharges, nm_evtNums, nm_tels, nm_allCharges


def get_muon_charge_load_evt(st2file, save_image_dir="muon_hunter_images_clean", #run_num=81783,
                        outfile_base="hide_label", save_text="clean_muon_events.txt",
                        start_event=None, stop_event=None, evtlist="../81783_muon_evtNum_telID.hdf5",
                        ntubes=5, dpi=144):
    #evt_file = "../"+str(run_num)+"_muon_evtNum_telID.hdf5"
    evt_file = evtlist
    evt_tels = load_hdf5(evt_file)
    evts = evt_tels[:,0]
    tels = evt_tels[:,1]
    print("Reading events from root file")
    evtNums, allCharges = read_st2_calib_channel_charge(st2file, tels=[0, 1, 2, 3], maskL2=True,
                                  l2channels=[[110, 249, 255, 404, 475, 499], [128, 173, 259, 498, 499],
                                              [37, 159, 319, 451, 499], [99, 214, 333, 499]],
                                  start_event=start_event, stop_event=stop_event, evtlist=evts, verbose=False,
                                  cleaning={'img': 5.0, 'brd': 2.5})
    #allCharge[telID][chanID][evt_count]
    try:
        # Opening file stream for light curve detection results
        print('Opening file stream for writing results.\n')
        outfile = open(save_text, 'a')
    except IOError:
        print('There was an error opening file {0}'.format(save_text))
        sys.exit()

    n_evts = allCharges.shape[2]
    print("saving_images")

    for i, this_evt in enumerate(evts):
        telID = tels[i]
        # Ntubes cut
        if np.sum(allCharges[telID, :, i] != 0) < ntubes:
            continue
        #this_image = charge_to_image_one_tel(allCharges[telID, :, i:i+1])
        #this_evt = evtNums[i]
        outfile.write(str(this_evt) + ', ' +str(telID) + '\n')
        cam = PyVAPlotCam(allCharges[telID, :, i], fig=plt.figure(figsize=(7, 7)), ax=plt.subplot(111))
        # cam.buildCamera(draw_pixNum=False)
        # cam.draw(drawColorbar=False, draw_pixNum=False)
        cam.draw(drawColorbar=False, draw_pixNum=False, cm=plt.cm.jet)
        plt.savefig(
            save_image_dir + "/" + str(outfile_base) + "_evt" + str(int(this_evt)) + "_tel" + str(telID) + ".jpeg",
            dpi=dpi)




def test():
    f = "/raid/biggams/qfeng/data/PG1553/81214UCORmedWinterMuonSt4.root"
    m_evtNums, m_tels, m_allCharges = read_muon_data(f, tels=[0, 1, 2, 3], read_charge=True, save_muon=False,
                                              cut_radius=0.5, cut_radius_upper=None,
                                              outfile_base="81214_test_0p5_muon", save_non_muon=False, num_of_non_muon=100,
                                              non_muon_outfile_base="81214_test_0p5_non_muon")

    f = "/raid/biggams/qfeng/data/1ES1959/2015/78598UCORmedWinterMuonSt4.root"
    evtNums, allCharges = read_st2_calib_channel_charge(f, tels=[0, 1, 2, 3], maskL2=True,
                                                        l2channels=[[110, 249, 255, 404, 475, 499],
                                                                    [128, 173, 259, 498, 499],
                                                                    [37, 159, 319, 451, 499], [99, 214, 333, 499]],
                                                        start_event=None, stop_event=None, evtlist=[610043],
                                                        verbose=False,
                                                        cleaning={'img': 5.0, 'brd': 2.5})


class PyVAPlotCam:
    ############################################################################
    #                       Initialization of the object.                      #
    ############################################################################
    def __init__(self, pixVals, fig=None, ax=None, forDQM=False):
        if fig is not None and ax is not None:
            self.fig = fig
            self.ax = ax
        elif forDQM == True:
            self.fig = plt.figure(figsize=(7, 7))
            self.ax = plt.subplot(111)
            # plt.subplots_adjust(bottom=0.1,left=0.0,right=1.015)
            plt.subplots_adjust(bottom=0.05, left=0.0, right=0.975)
        else:
            # self.fig = plt.figure()
            self.fig = plt.figure(figsize=[7, 5.7])
            # self.ax = plt.axes([0.08,0.0095,0.85,0.9275])
            # self.ax = plt.axes([0.08,0.0095,0.85,0.9275])
            self.ax = plt.axes([0.0, 0.0095, 0.95, 0.9275])
        self.make_nice_ax()

        self.cbar = None

        self.patches = []
        self.pixNumArr = []

        self.pixSideLength = 1.
        self.numCamSpirals = 13

        if forDQM == True:
            # self.textOffsetX = -self.pixSideLength*0.225
            self.textOffsetX = -self.pixSideLength * 0.25
            # self.textOffsetY = -self.pixSideLength*0.45
            self.textOffsetY = -self.pixSideLength * 0.35
            self.pixLabelFontSize = 8.5
        else:
            self.textOffsetX = -self.pixSideLength * 0.25
            self.textOffsetY = -self.pixSideLength * 0.3
            self.pixLabelFontSize = 8

        self.pixVals = np.array(pixVals)
        self.forDQM = forDQM
        # self.buildCamera()

    def make_nice_ax(self):
        self.ax.set_xlim(-21, 21)
        self.ax.set_ylim(-21, 21)
        self.ax.set_frame_on(False)
        self.ax.set_xticklabels([])
        self.ax.set_xticks([])
        self.ax.set_yticklabels([])
        self.ax.set_yticks([])

    def switch_tel(self, pixVals, ax):
        self.ax = ax
        self.make_nice_ax()
        self.pixVals = np.array(pixVals)

    ############################################################################
    #                         Build the VERITAS camera.                        #
    ############################################################################
    def buildCamera(self, draw_pixNum=True, cm=plt.cm.CMRmap):
        self.patches = []
        self.pixNumArr = []

        pixVertices = self.getPolygon(radius=self.pixSideLength);
        polygon = Polygon(pixVertices, True)
        self.patches.append(polygon)

        pixNum = 1
        self.pixNumArr.append(pixNum)
        pixNumStr = "%d" % pixNum

        if draw_pixNum:
            plt.text(self.textOffsetX, self.textOffsetY, pixNum, size=self.pixLabelFontSize)

        deltaX = math.sqrt(3) * self.pixSideLength / 2.
        deltaY = (3. / 2. * self.pixSideLength)

        for spiral in range(1, self.numCamSpirals + 1):

            xPos = 2. * float((spiral)) * deltaX
            yPos = 0.

            # For the two outermost spirals, there is not a pixel in the y=0 row.
            if spiral < 12:
                pixVertices = self.getPolygon(radius=self.pixSideLength, xCenter=xPos, yCenter=yPos);
                polygon = Polygon(pixVertices, True)
                self.patches.append(polygon)

                pixNum += 1
                self.pixNumArr.append(pixNum)
                pixNumStr = "%d" % pixNum
                if draw_pixNum:
                    plt.text(xPos + self.textOffsetX * (math.floor(math.log10(pixNum) + 1.)), yPos + self.textOffsetY,
                             pixNum, size=self.pixLabelFontSize)

            nextPixDir = np.zeros((spiral * 6, 2))
            skipPixel = np.zeros((spiral * 6, 1))

            for y in range(spiral * 6 - 1):
                # print "%d" % (y/spiral)
                if (y / spiral < 1):
                    nextPixDir[y, :] = [-1, -1]
                elif (y / spiral >= 1 and y / spiral < 2):
                    nextPixDir[y, :] = [-2, 0]
                elif (y / spiral >= 2 and y / spiral < 3):
                    nextPixDir[y, :] = [-1, 1]
                elif (y / spiral >= 3 and y / spiral < 4):
                    nextPixDir[y, :] = [1, 1]
                elif (y / spiral >= 4 and y / spiral < 5):
                    nextPixDir[y, :] = [2, 0]
                elif (y / spiral >= 5 and y / spiral < 6):
                    nextPixDir[y, :] = [1, -1]

            # The two outer spirals are not fully populated with pixels.
            # The second outermost spiral is missing only six pixels (one was excluded above).
            if (spiral == 12):
                for i in range(1, 6):
                    skipPixel[spiral * i - 1] = 1
            # The outmost spiral only has a total of 36 pixels.  We need to skip over the
            # place holders for the rest.
            if (spiral == 13):
                skipPixel[0:3] = 1
                skipPixel[9:16] = 1
                skipPixel[22:29] = 1
                skipPixel[35:42] = 1
                skipPixel[48:55] = 1
                skipPixel[61:68] = 1
                skipPixel[74:77] = 1

            for y in range(spiral * 6 - 1):

                xPos += nextPixDir[y, 0] * deltaX
                yPos += nextPixDir[y, 1] * deltaY

                if skipPixel[y, 0] == 0:
                    pixVertices = self.getPolygon(radius=self.pixSideLength, xCenter=xPos, yCenter=yPos);
                    polygon = Polygon(pixVertices, True)
                    self.patches.append(polygon)

                    pixNum += 1
                    self.pixNumArr.append(pixNum)
                    pixNumStr = "%d" % pixNum
                    if draw_pixNum:
                        plt.text(xPos + self.textOffsetX * (math.floor(math.log10(pixNum) + 1.)),
                                 yPos + self.textOffsetY, pixNum, size=self.pixLabelFontSize)

        self.patchCollec = PatchCollection(self.patches, cmap=cm, alpha=1.)
        # self.patchCollec = PatchCollection(self.patches, cmap=matplotlib.cm.jet, alpha=1.)
        # self.patchCollec = PatchCollection(self.patches, cmap=matplotlib.cm.spectral, alpha=1.)
        # self.colors = 100*np.random.rand(len(self.patches))

    ############################################################################
    #                        Add title, annotations, etc.                      #
    ############################################################################
    def addTitle(self, title):
        if self.forDQM == True:
            self.fig.suptitle(title, fontsize=16, x=0.415)
        else:
            # self.fig.suptitle(title,fontsize=16,x=0.429)
            self.fig.suptitle(title, fontsize=16, x=0.38)

    def addTitleTopLeft(self, title):
        if self.forDQM == True:
            self.fig.suptitle(title, fontsize=12, x=0.147, y=0.925)
        else:
            # self.fig.suptitle(title,fontsize=12,x=0.147,y=0.925)
            self.fig.suptitle(title, fontsize=12, x=0.09, y=0.925)

    def addTitleTopRight(self, title):
        if self.forDQM == True:
            self.fig.suptitle(title, fontsize=12, x=0.663, y=0.925)
        else:
            # self.fig.suptitle(title,fontsize=12,x=0.71,y=0.925)
            self.fig.suptitle(title, fontsize=12, x=0.68, y=0.925)

    def addTitleBottomLeft(self, title):
        if self.forDQM == True:
            self.fig.suptitle(title, fontsize=12, x=0.147, y=0.1)
        else:
            # self.fig.suptitle(title,fontsize=12,x=0.147, y=0.1)
            self.fig.suptitle(title, fontsize=12, x=0.09, y=0.1)

    def addTitleBottomRight(self, title):
        if self.forDQM == True:
            self.fig.suptitle(title, fontsize=12, x=0.663, y=0.1)
        else:
            # self.fig.suptitle(title,fontsize=12,x=0.71, y=0.1)
            self.fig.suptitle(title, fontsize=12, x=0.68, y=0.1)

    def addColorbarTitle(self, title):
        self.cbar.set_label(title, fontsize=16)

    ############################################################################
    #                      Draw/clear/save the camera map.                     #
    ############################################################################
    def draw(self, drawColorbar=True, masked=False, draw_pixNum=True, cm=plt.cm.CMRmap):
        self.buildCamera(draw_pixNum=draw_pixNum, cm=cm)

        self.pixValsMasked = np.ma.masked_where(self.pixVals == 0, self.pixVals)
        if masked == True:
            self.patchCollec.set_array(self.pixValsMasked)
        else:
            self.patchCollec.set_array(self.pixVals)
        self.ax.add_collection(self.patchCollec)
        if drawColorbar == True:
            if self.forDQM == True:
                self.cbar = plt.colorbar(self.patchCollec, shrink=0.95, pad=0.0)
            else:
                self.cbar = plt.colorbar(self.patchCollec, shrink=0.95)
            for label in self.cbar.ax.get_yticklabels():
                label.set_fontsize(16)

    def redraw(self, newPixVals, drawColorbar=True, masked=False, draw_pixNum=False):
        plt.clf()
        plt.cla()

        self.fig = plt.figure(figsize=(7, 7))
        self.ax = plt.axes([0.09, 0.0095, 0.85, 0.9275])
        self.ax.set_xlim(-21, 21)
        self.ax.set_ylim(-21, 21)
        self.ax.set_frame_on(False)
        self.ax.set_xticklabels([])
        self.ax.set_xticks([])
        self.ax.set_yticklabels([])
        self.ax.set_yticks([])

        self.pixVals = newPixVals
        self.draw(drawColorbar=drawColorbar, masked=masked, draw_pixNum=draw_pixNum)

    def show(self):
        self.fig.show()

    def save(self, figName):
        tmpArr = figName.split('\.')
        # if self.forDQM == True:
        if tmpArr[-1] == 'png':
            plt.savefig(figName, dpi=71)
        else:
            plt.savefig(figName)

    def setScale(self, minVal, maxVal):
        self.patchCollec.set_clim([minVal, maxVal])

    ############################################################################
    #                       Get the verices of a polygon.                      #
    ############################################################################
    # This makes the hexagon for a given pixel.
    def getPolygon(self, numSides=6, radius=1, xCenter=0., yCenter=0., rotAngle=math.pi / 2.):

        pixVertices = np.zeros((numSides, 2))

        for i in range(1, numSides + 1):
            pixVertices[i - 1, :] = [xCenter + radius * math.cos(2 * math.pi * float(i) / float(numSides) + rotAngle),
                                     yCenter + radius * math.sin(2 * math.pi * float(i) / float(numSides) + rotAngle)]

        return pixVertices

