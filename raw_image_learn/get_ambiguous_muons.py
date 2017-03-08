from VNN.utils.vegas_io import *
from VNN.utils.io import *
from VNN.vmodel.cnn import CNN
from VNN.utils.squarecam import *
from VNN.utils.image import *
#from raw_image_learn.PyVAPlotCam import *
import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import math


def train_muon_cnn_4runs(save=True, load=False, outfile="muon_cnn_4file_default.json", weight_file="muon_cnn_4file_default_weights.hdf5"):
    train_x = load_hdf5("muon_train_x_4files.hdf5")
    train_y = load_hdf5("muon_train_y_4files.hdf5")
    test_x = load_hdf5("muon_test_x_4files.hdf5")
    test_y = load_hdf5("muon_test_y_4files.hdf5")

    muon_cnn = CNN()

    if load and not save:
        muon_cnn.load_keras_model(outfile, weight_file)
        return muon_cnn

    muon_cnn.load_data(train_x, train_y, test_x, test_y)
    muon_cnn.init_model(input_shape=(1, 54, 54), nb_classes=2)
    #muon_cnn.train(nb_epoch=5)
    muon_cnn.train(early_stop=10, nb_epoch=50)
    if save:
        muon_cnn.save_keras_model(outfile, weight_file)

    return muon_cnn


def train_muon_cnn_from_hdf5_files(trainx_fs, trainy_fs, testx_fs, testy_fs,
                                   save=True, load=False,
                                   outfile="muon_cnn_4file_default_pandas0.18.json", weight_file="muon_cnn_4file_default_weights_pandas0.18.hdf5"):
    train_x, train_y, test_x, test_y = read_train_xy_list(trainx_fs, trainy_fs, testx_fs, testy_fs)

    muon_cnn = CNN()

    if load and not save:
        muon_cnn.load_keras_model(outfile, weight_file)
        return muon_cnn

    muon_cnn.load_data(train_x, train_y, test_x, test_y)
    muon_cnn.init_model(input_shape=(1, 54, 54), nb_classes=2)
    #muon_cnn.train(nb_epoch=5)
    muon_cnn.train(early_stop=10, nb_epoch=50)
    if save:
        muon_cnn.save_keras_model(outfile, weight_file)

    return muon_cnn



def load_cnn_model(model_file="muon_cnn_16file_update.json", weight_file="muon_cnn_16file_update_weights.hdf5"):
    muon_cnn = CNN()
    muon_cnn.load_keras_model(model_file, weight_file)
    return muon_cnn


def read_train_xy_list(trainx_fs, trainy_fs, testx_fs, testy_fs):
    train_x = np.zeros((0,1,54,54))
    train_y = np.zeros((0))
    test_x = np.zeros((0,1,54,54))
    test_y = np.zeros((0))

    for trainx_f, trainy_f, testx_f, testy_f in zip(trainx_fs, trainy_fs, testx_fs, testy_fs):
        train_x_ = load_hdf5(trainx_f)
        train_y_ = load_hdf5(trainy_f)
        test_x_ = load_hdf5(testx_f)
        test_y_ = load_hdf5(testy_f)
        train_x = np.concatenate([train_x, train_x_])
        train_y = np.concatenate([train_y, train_y_])
        test_x = np.concatenate([test_x, test_x_])
        test_y = np.concatenate([test_y, test_y_])
    return train_x, train_y, test_x, test_y


def train_muon_cnn(train_x, train_y, test_x, test_y, save=True, load=False,
                   outfile="muon_cnn_16file_update.json", weight_file="muon_cnn_16file_update_weights.hdf5"):

    muon_cnn = CNN()

    if load and not save:
        muon_cnn.load_keras_model(outfile, weight_file)
        return muon_cnn

    muon_cnn.load_data(train_x, train_y, test_x, test_y)
    muon_cnn.init_model(input_shape=(1, 54, 54), nb_classes=2)
    #muon_cnn.train(nb_epoch=5)
    muon_cnn.train(early_stop=10, nb_epoch=50)
    if save:
        muon_cnn.save_keras_model(outfile, weight_file)

    return muon_cnn



def get_ambiguous_muons(muon_cnn, st2file, save_image_dir="muon_hunter_ambiguous_images",
                        outfile_base="ambiguous", save_text="ambiguous_muon_events.txt",
                        start_event=None, stop_event=None, evtlist=None,
                        score_lower=0.5, score_upper=0.8, ntubes=5, dpi=144):
    evtNums, allCharges = read_st2_calib_channel_charge(st2file, tels=[0, 1, 2, 3], maskL2=True,
                                  l2channels=[[110, 249, 255, 404, 475, 499], [128, 173, 259, 498, 499],
                                              [37, 159, 319, 451, 499], [99, 214, 333, 499]],
                                  start_event=start_event, stop_event=stop_event, evtlist=evtlist, verbose=False,
                                  cleaning={'img': 5.0, 'brd': 2.5})
    #allCharge[telID][chanID][evt_count]
    model = muon_cnn.get_model()

    #ambiguous_evtNums = []
    #ambiguous_scores = []
    try:
        # Opening file stream for light curve detection results
        print('Opening file stream for writing results.\n')
        outfile = open(save_text, 'a')
    except IOError:
        print('There was an error opening file {0}'.format(save_text))
        sys.exit()

    n_evts = allCharges.shape[2]
    for i in range(n_evts):
        for telID in range(4):
            # Ntubes cut
            if np.sum(allCharges[telID, :, i] != 0) < ntubes:
                continue
            this_image = charge_to_image_one_tel(allCharges[telID, :, i:i+1])
            this_predict = model.predict_proba(this_image)
            #print(this_predict)
            if this_predict[0,1]>score_lower and this_predict[0,1]<score_upper:
                this_evt = evtNums[i]
                outfile.write(str(this_evt) + ', ' +str(telID) + ', ' + str(this_predict) + '\n')

                cam = PyVAPlotCam(allCharges[telID, :, i], fig=plt.figure(figsize=(7, 7)), ax=plt.subplot(111))
                # cam.buildCamera(draw_pixNum=False)
                # cam.draw(drawColorbar=False, draw_pixNum=False)
                cam.draw(drawColorbar=False, draw_pixNum=False, cm=plt.cm.jet)
                plt.savefig(
                    save_image_dir + "/" + str(outfile_base) + "_evt" + str(int(this_evt)) + "_tel" + str(telID) + ".jpeg",
                    dpi=dpi)

def get_cnn_muons(muon_cnn, st2file, save_image_dir="muon_hunter_cnn_muon_images", run_num=0,
                  outfile_base="hide_label", save_text="cnn_muon_events.txt", outfile_base_non_muon="hide_label",
                  save_non_muon_image_dir="muon_hunter_cnn_non_muon_images", save_non_muon_text="cnn_non_muon_events.txt",
                  start_event=None, stop_event=None, evtlist=None, score_lower=0.9, score_upper=1.0,
                  score_lower_non_muon=0.0, score_upper_non_muon=0.1, ntubes=5, dpi=144, non_muon_cap=10000,
                  hdf5_file="cnn_muon_events_oversampled.hdf5", non_muon_hdf5_file="cnn_non_muon_events_oversampled.hdf5",
                  save_hdf5=True
                  ):
    print("reading events from root file")
    evtNums, allCharges = read_st2_calib_channel_charge(st2file, tels=[0, 1, 2, 3], maskL2=True,
                                  l2channels=[[110, 249, 255, 404, 475, 499], [128, 173, 259, 498, 499],
                                              [37, 159, 319, 451, 499], [99, 214, 333, 499]],
                                  start_event=start_event, stop_event=stop_event, evtlist=evtlist, verbose=False,
                                  cleaning={'img': 5.0, 'brd': 2.5})
    #allCharge[telID][chanID][evt_count]
    model = muon_cnn.get_model()

    #ambiguous_evtNums = []
    #ambiguous_scores = []
    #try:
    #    # Opening file stream for light curve detection results
    #    print('Opening file stream for writing results.\n')
    #    outfile = open(save_text, 'a')
    #    outfile_nm = open(save_non_muon_text, 'a')
    #except IOError:
    #    print('There was an error opening file {0}'.format(save_text))
    #    print('or an error opening file {0}'.format(save_non_muon_text))
    #    sys.exit()

    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)
    if not os.path.exists(save_non_muon_image_dir):
        os.makedirs(save_non_muon_image_dir)
        
    n_evts = allCharges.shape[2]
    non_muon_count = 0
    for i in range(n_evts):
        for telID in range(4):
            # Ntubes cut
            if np.sum(allCharges[telID, :, i] != 0) < ntubes:
                continue
            this_image = charge_to_image_one_tel(allCharges[telID, :, i:i+1])
            this_predict = model.predict_proba(this_image)
            #print(this_predict)
            if this_predict[0,1]>score_lower and this_predict[0,1]<score_upper:
                this_evt = evtNums[i]
                if save_hdf5:
                    update_evts_hdf5(hdf5_file, run_num=run_num, n_tels=1,
                                     images=this_image, create=True,
                                     evt_nums=this_evt)
                with open(save_text, "w") as outfile:
                    outfile.write(str(this_evt) + ', ' +str(telID) + ', ' + str(this_predict) + '\n')

                cam = PyVAPlotCam(allCharges[telID, :, i], fig=plt.figure(figsize=(7, 7)), ax=plt.subplot(111))
                # cam.buildCamera(draw_pixNum=False)
                # cam.draw(drawColorbar=False, draw_pixNum=False)
                cam.draw(drawColorbar=False, draw_pixNum=False, cm=plt.cm.jet)
                plt.savefig(
                    save_image_dir + "/" + str(outfile_base) + "_evt" + str(int(this_evt)) + "_tel" + str(telID) + ".jpeg",
                    dpi=dpi)

            elif this_predict[0,1]>score_lower_non_muon and this_predict[0,1]<score_upper_non_muon:
                if non_muon_count > non_muon_cap:
                    continue
                non_muon_count += 1
                this_evt = evtNums[i]
                if save_hdf5:
                    update_evts_hdf5(non_muon_hdf5_file, run_num=run_num, n_tels=1,
                                     images=this_image, create=True,
                                     evt_nums=this_evt)
                with open(save_non_muon_text, "w") as outfile_nm:
                    outfile_nm.write(str(this_evt) + ', ' +str(telID) + ', ' + str(this_predict) + '\n')

                cam = PyVAPlotCam(allCharges[telID, :, i], fig=plt.figure(figsize=(7, 7)), ax=plt.subplot(111))
                # cam.buildCamera(draw_pixNum=False)
                # cam.draw(drawColorbar=False, draw_pixNum=False)
                cam.draw(drawColorbar=False, draw_pixNum=False, cm=plt.cm.jet)
                plt.savefig(
                    save_non_muon_image_dir + "/" + str(outfile_base_non_muon) + "_evt" + str(int(this_evt)) + "_tel" + str(telID) + ".jpeg",
                    dpi=dpi)
    #outfile.close()
    #outfile_nm.close()



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


