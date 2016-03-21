#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import math
import pandas as pd

class PyVAPlotSquareCam:
    
    ############################################################################
    #                       Initialization of the object.                      #
    ############################################################################
    def __init__(self, pixVals, fig=None, ax=None, forDQM=False):
        if fig is not None and ax is not None:
            self.fig = fig
            self.ax = ax
        elif forDQM == True:
            self.fig = plt.figure(figsize=(7,7))
            self.ax = plt.subplot(111)
            # plt.subplots_adjust(bottom=0.1,left=0.0,right=1.015)
            plt.subplots_adjust(bottom=0.05,left=0.0,right=0.975)
        else:
            # self.fig = plt.figure()
            self.fig = plt.figure(figsize=[7,5.7])
            # self.ax = plt.axes([0.08,0.0095,0.85,0.9275])
            # self.ax = plt.axes([0.08,0.0095,0.85,0.9275])
            self.ax = plt.axes([0.0,0.0095,0.95,0.9275])
        self.make_nice_ax()

        self.cbar = None
        
        self.patches = []
        self.pixNumArr = []
        
        self.pixSideLength = 1.
        self.numCamSpirals = 13
        
        self.pos=np.zeros((len(pixVals),2))

        if forDQM == True:
            # self.textOffsetX = -self.pixSideLength*0.225
            self.textOffsetX = -self.pixSideLength*0.25
            # self.textOffsetY = -self.pixSideLength*0.45
            self.textOffsetY = -self.pixSideLength*0.35
            self.pixLabelFontSize = 8.5
        else:
            self.textOffsetX = -self.pixSideLength*0.25
            self.textOffsetY = -self.pixSideLength*0.3
            self.pixLabelFontSize = 8
        
        self.pixVals = np.array(pixVals)
        self.forDQM = forDQM
        # self.buildCamera()

    def make_nice_ax(self):
        self.ax.set_xlim(-21,21)
        self.ax.set_ylim(-21,21)
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
    def buildSquareCamera(self):    
        self.patches = []
        self.pixNumArr = []
        
        pixVertices = self.getPolygon(numSides=4, radius=self.pixSideLength, rotAngle=math.pi/4.);
        polygon = Polygon(pixVertices, True)
        self.patches.append(polygon)
        
        pixNum = 1
        self.pixNumArr.append(pixNum)
        pixNumStr = "%d" % pixNum
        
        self.pos[0,0]=0.
        self.pos[0,1]=0.

        plt.text(self.textOffsetX, self.textOffsetY, pixNum, size=self.pixLabelFontSize)
        
        #deltaX = math.sqrt(3)*self.pixSideLength/2.
        #deltaY = (3./2.*self.pixSideLength)
        deltaX = self.pixSideLength*math.sqrt(2)/2.
        deltaY = self.pixSideLength*math.sqrt(2)

        for spiral in range(1,self.numCamSpirals+1):
            
            xPos = 2.*float((spiral))*deltaX
            yPos = 0.
            
            # For the two outermost spirals, there is not a pixel in the y=0 row.
            if spiral < 12:
                pixVertices = self.getPolygon(numSides=4, radius=self.pixSideLength, xCenter=xPos, yCenter=yPos, rotAngle=math.pi/4.);
                polygon = Polygon(pixVertices, True)
                self.patches.append(polygon)

                pixNum += 1
                self.pixNumArr.append(pixNum)
                pixNumStr = "%d" % pixNum
                #plt.text(xPos+self.textOffsetX*(math.floor(math.log10(pixNum)+1.)), yPos+self.textOffsetY, pixNum, size=self.pixLabelFontSize)
                
                self.pos[pixNum-1,0]=xPos
                self.pos[pixNum-1,1]=yPos

            
            nextPixDir = np.zeros((spiral*6,2))
            skipPixel = np.zeros((spiral*6,1))
            
            for y in range(spiral*6-1):
                # print "%d" % (y/spiral)
                if (y/spiral < 1):
                    nextPixDir[y,:] = [-1,-1]
                elif (y/spiral >= 1 and y/spiral < 2):
                    nextPixDir[y,:] = [-2,0]
                elif (y/spiral >= 2 and y/spiral < 3):
                    nextPixDir[y,:] = [-1,1]
                elif (y/spiral >= 3 and y/spiral < 4):
                    nextPixDir[y,:] = [1,1]
                elif (y/spiral >= 4 and y/spiral < 5):
                    nextPixDir[y,:] = [2,0]
                elif (y/spiral >= 5 and y/spiral < 6):
                    nextPixDir[y,:] = [1,-1]
                
            
            # The two outer spirals are not fully populated with pixels.
            # The second outermost spiral is missing only six pixels (one was excluded above).
            if (spiral == 12):
                for i in range(1,6):
                    skipPixel[spiral*i-1] = 1
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
            
            for y in range(spiral*6-1)((numX, numX)):
                
                xPos += nextPixDir[y,0]*deltaX
                yPos += nextPixDir[y,1]*deltaY
                
                if skipPixel[y,0] == 0:
                    pixVertices = self.getPolygon(numSides=4, radius=self.pixSideLength, xCenter=xPos, yCenter=yPos, rotAngle=math.pi/4.);
                    polygon = Polygon(pixVertices, True)
                    self.patches.append(polygon)
                
                    pixNum += 1
                    self.pixNumArr.append(pixNum)
                    pixNumStr = "%d" % pixNum
                    #plt.text(xPos+self.textOffsetX*(math.floor(math.log10(pixNum)+1.)), yPos+self.textOffsetY, pixNum, size=self.pixLabelFontSize)

                    self.pos[pixNum-1,0]=xPos
                    self.pos[pixNum-1,1]=yPos
    
        self.patchCollec = PatchCollection(self.patches, cmap=matplotlib.cm.jet, alpha=1.)
        # self.patchCollec = PatchCollection(self.patches, cmap=matplotlib.cm.spectral, alpha=1.)
        # self.colors = 100*np.random.rand(len(self.patches))


    ############################################################################
    #                        Add title, annotations, etc.                      #
    ############################################################################
    def addTitle(self,title):
        if self.forDQM == True:
            self.fig.suptitle(title,fontsize=16,x=0.415)
        else:
            # self.fig.suptitle(title,fontsize=16,x=0.429)
            self.fig.suptitle(title,fontsize=16,x=0.38)
    
    def addTitleTopLeft(self,title):
        if self.forDQM == True:
            self.fig.suptitle(title,fontsize=12,x=0.147,y=0.925)
        else:
            # self.fig.suptitle(title,fontsize=12,x=0.147,y=0.925)
            self.fig.suptitle(title,fontsize=12,x=0.09,y=0.925)
    
    def addTitleTopRight(self,title):
        if self.forDQM == True:
            self.fig.suptitle(title,fontsize=12,x=0.663, y=0.925)
        else:
            # self.fig.suptitle(title,fontsize=12,x=0.71,y=0.925)
            self.fig.suptitle(title,fontsize=12,x=0.68,y=0.925)
    
    def addTitleBottomLeft(self,title):
        if self.forDQM == True:
            self.fig.suptitle(title,fontsize=12,x=0.147, y=0.1)
        else:
            # self.fig.suptitle(title,fontsize=12,x=0.147, y=0.1)
            self.fig.suptitle(title,fontsize=12,x=0.09, y=0.1)
        
    def addTitleBottomRight(self,title):
        if self.forDQM == True:
            self.fig.suptitle(title,fontsize=12,x=0.663, y=0.1)
        else:
            # self.fig.suptitle(title,fontsize=12,x=0.71, y=0.1)
            self.fig.suptitle(title,fontsize=12,x=0.68, y=0.1)
    
    def addColorbarTitle(self, title):
        self.cbar.set_label(title,fontsize=16)

    
    ############################################################################
    #                      Draw/clear/save the camera map.                     #
    ############################################################################
    def draw(self, drawColorbar=True, masked=False):
        # self.buildCamera()
        self.buildSquareCamera()

        self.pixValsMasked = np.ma.masked_where(self.pixVals == 0, self.pixVals)
        if masked == True:
            self.patchCollec.set_array(self.pixValsMasked)
        else:
            self.patchCollec.set_array(self.pixVals)
        self.ax.add_collection(self.patchCollec)
        if drawColorbar == True:
            if self.forDQM == True:
                self.cbar = plt.colorbar(self.patchCollec, shrink=0.95,pad=0.0)
            else:
                self.cbar = plt.colorbar(self.patchCollec, shrink=0.95)
            for label in self.cbar.ax.get_yticklabels():
                label.set_fontsize(16)
    
    def redraw(self, newPixVals, drawColorbar=True, masked=False):
        plt.clf()
        plt.cla()
        
        self.fig = plt.figure()
        self.ax = plt.axes([0.09,0.0095,0.85,0.9275])
        self.ax.set_xlim(-21,21)
        self.ax.set_ylim(-21,21)
        self.ax.set_frame_on(False)
        self.ax.set_xticklabels([])
        self.ax.set_xticks([])
        self.ax.set_yticklabels([])
        self.ax.set_yticks([])
        
        self.pixVals = newPixVals
        self.draw(drawColorbar=drawColorbar, masked=masked)
        
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
    def getPolygon(self,numSides=6, radius=1, xCenter=0., yCenter=0.,rotAngle=math.pi/2.):
        
        pixVertices = np.zeros((numSides,2))
        
        for i in range(1,numSides+1):
            pixVertices[i-1,:] = [xCenter + radius*math.cos(2*math.pi*float(i)/float(numSides)+rotAngle),
                                  yCenter + radius*math.sin(2*math.pi*float(i)/float(numSides)+rotAngle)]
            
        return pixVertices

    ############################################################################
    #         Make oversampled square, and use Gaussian kernels                #
    ############################################################################

    def oversample(self, rate=2, cam_radius=19.):
        #oversampling at rate 2 means 4 squares for a pixel
        assert isinstance( rate, int ), "can only oversample at integer rate"
        self.size=np.sqrt(2.)/rate
        self.square_coordinates=np.zeros((500, 2))
        #self.cam_radius = cam_radius
        #numX = int(cam_radius*2./size+1)
        numX = int((np.max(self.pos)-np.min(self.pos))/self.size+1*rate)
        print numX
        #self.z = np.random.rand(numX, numX)
        self.z = np.zeros((numX, numX))
        self.z = -np.ones((numX, numX))
        self.x = self.pos[:,0]/np.sqrt(2.)*rate
        self.y = self.pos[:,1]/np.sqrt(2.)*rate
        self.testdf = pd.DataFrame(index=range(500), columns=['xpos', 'xind', 'ypos', 'yind'])
        #self.x = np.arange(-cam_radius, cam_radius+size, size)
        #self.y = np.arange(-cam_radius, cam_radius+size, size)
        for i_ in range(len(self.pixVals)):
            print "Pixel ", i_, "x",  self.x[i_], (self.pos[i_,0]-np.min(self.pos))/np.sqrt(2.)*rate
            print "y", self.y[i_], (self.pos[i_,1]-np.min(self.pos))/np.sqrt(2.)*rate
            self.testdf.iloc[i_] = self.x[i_], (self.pos[i_,0]-np.min(self.pos))/np.sqrt(2.)*rate, self.y[i_], (self.pos[i_,1]-np.min(self.pos))/np.sqrt(2.)*rate
            #x_ = (self.pos[i_,0]-np.min(self.pos))/np.sqrt(2.)*2.
            #y_ = (self.pos[i_,1]-np.min(self.pos))/np.sqrt(2.)*2.
            y_ = self.x[i_] -np.min(self.x)
            x_ = self.y[i_] -np.min(self.y)
            self.square_coordinates[i_, 0] = y_
            self.square_coordinates[i_, 1] = x_
            self.z[x_:x_+rate, y_:y_+rate] = self.pixVals[i_]
            #self.z[x_, y_] = self.pixVals[i_]
            #for k in range(1, rate):
            #    self.z[x_, y_+k] = self.pixVals[i_]
            #    self.z[x_+k, y_] = self.pixVals[i_]
            #    self.z[x_+k, y_+k] = self.pixVals[i_]

    def drawOversampled(self, cm = plt.cm.CMRmap, vmin=100, vmax=600):
        plt.pcolor(self.z, cmap=cm, vmin=vmin, vmax=vmax)
        for pixNum in range(499):
            plt.text(self.square_coordinates[pixNum,0], self.square_coordinates[pixNum, 1], pixNum, size=self.pixLabelFontSize)
        plt.show()


    def GaussianKernel(self, x=0, y=0, val=1., size=2, stddev = 0.6):
        """
        self.z is the square grid (oversampled)
        add a Gaussian centered at x, y to z
        """
        pixel_scale = (cam_radius*2./(self.z.shape[0]-1))
        xlow=int(np.floor((x-size+cam_radius)/pixel_scale))
        xhi=int(np.ceil((x+size+cam_radius)/pixel_scale))
        ylow=int(np.floor((y-size+cam_radius)/pixel_scale))
        yhi=int(np.ceil((y+size+cam_radius)/pixel_scale))
        xlow=max(xlow, 0)
        ylow=max(ylow, 0)
        xhi=min(xhi, self.z.shape[0]-1)
        yhi=min(xhi, self.z.shape[0]-1)
        print xlow, xhi, ylow, yhi
        for i_ in range(xlow, xhi+1):
            for j_ in range(ylow, yhi+1):
                x_ = -cam_radius+i_*pixel_scale
                y_ = -cam_radius+j_*pixel_scale
                self.z[i_][j_] += val*np.exp(-0.5 * ((x_-x)**2 + (y_-y)**2) / (stddev/pixel_scale)**2)
        #return self.z

    def FillGaussian(self, size=2, stddev = 0.6):
        for i_, val_ in enumerate(self.pixVals):
            print i_, val_, self.pos[i_,0], self.pos[i_,1]
            self.GaussianKernel(x=self.pos[i_,0], y=self.pos[i_,1], val=val_, size=size, stddev=stddev)

    def plotZ(self, cm=plt.cm.inferno, interpolation='nearest', scale=None):
        img = plt.imshow(self.z, extent=(self.x.min(), self.x.max(), self.y.min(), self.y.max()),interpolation=interpolation, cmap=cm)
        plt.colorbar(img, cmap=cm, boundaries=scale)
        if scale:
            self.setScale(scale[0], scale[1])

    def plotGrid(self):
        ax = plt.gca()
        # grid "shades" (boxes)
        w, h = self.x[1] - self.x[0], self.y[1] - self.y[0]
        for i, x in enumerate(self.x[:-1]):
            for j, y in enumerate(self.y[:-1]):
                if i % 2 == j % 2: # racing flag style
                    ax.add_patch(Rectangle((x, y), w, h, fill=True, color='#008610', alpha=.1))
        # grid lines
        for x in self.x:
            plt.plot([x, x], [self.y[0], self.y[-1]], color='black', alpha=.33, linestyle=':')
        for y in self.y:
            plt.plot([self.x[0], self.x[-1]], [y, y], color='black', alpha=.33, linestyle=':')
        #plt.show()

