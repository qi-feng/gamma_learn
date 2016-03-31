
from PyVAPlotSquareCam import *
import pandas as pd

df = pd.read_csv("64081.txt", sep = r"\s+", header=None, nrows=200)

cols = []
for i in range(4):
    cols.extend(['T'+str(i+1)+'EvtNum', 'T'+str(i+1)+'TrigMask', 'T'+str(i+1)+'EvtType', 'T'+str(i+1)+'EvtFlag'])
    for j in range(500):
        cols.extend(['T'+str(i+1)+'Trig'+str(j)])
    for j in range(500):
        cols.extend(['T'+str(i+1)+'Charge'+str(j)])

def drawSquareEvent(pixelVals, title="Camera map oversampled with squares", colorbar_title=None, rate=2, cm = plt.cm.CMRmap, vmin=100, vmax=600):
        cam = PyVAPlotSquareCam(pixelVals)
        cam.buildSquareCamera();
        cam.oversample(rate=rate)
        cam.addTitle(title)
        x_over = np.arange(cam.x.min(), cam.x.max(), (cam.x.max() - cam.x.min())/cam.z.shape[0])
        y_over = np.arange(cam.y.min(), cam.y.max(), (cam.y.max() - cam.y.min())/cam.z.shape[1])
        #x_over = np.arange(-2, 2, 4./cam.z.shape[0])
        #y_over = np.arange(-2, 2, 4./cam.z.shape[1])
        plt.pcolor(x_over, y_over, cam.z, cmap=cm, vmin=vmin, vmax=vmax)
        if colorbar_title:
            cam.addColorbarTitle(colorbar_title)
        plt.colorbar()
        plt.show()

def drawOversampled(z, cm = plt.cm.CMRmap, vmin=100, vmax=600, xmin=0, xmax = 54, ax=None, fig=None):
    z_index = pd.read_csv("oversample_coordinates.csv")
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        doplot = True
    else:
        doplot = False
    im = ax.pcolor(z.T, cmap=cm, vmin=vmin, vmax=vmax)
    for pixNum in range(499):
        ax.text(z_index.at[pixNum,'x1'], z_index.at[pixNum, 'y1'], pixNum, size=7)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    fig.colorbar(im, ax=ax)
    if doplot:
        plt.show()
    else:
        return ax

def drawArrayOversampled(zs, cm = plt.cm.CMRmap, vmin=100, vmax=600, xmin=0, xmax = 54, suppress_axes=True):
    #z_index = pd.read_csv("oversample_coordinates.csv")
    fig, ax = plt.subplots(2, 2)
    for i in range(4):
        ax_ = drawOversampled(zs[i], ax=ax.flatten()[i], fig=fig, cm = cm, vmin=vmin, vmax=vmax, xmin=xmin, xmax=xmax)
        if suppress_axes:
            ax_.set_frame_on(False)
            ax_.set_xticklabels([])
            ax_.set_xticks([])
            ax_.set_yticklabels([])
            ax_.set_yticks([])

    plt.tight_layout()
    plt.show()


#evt32 = df.iloc[[33]].values[0][504:504+500]
#
##drawEvent(evt32, scale=(0,500), title="Raw Charge - T%d" % (1), colorbar_title = "Charge [DC]")
#
#for i in [32, 35, 40, 43, 71]:
#    evt32 = df.iloc[[i]].values[0][504:504+500]
#    drawEvent(evt32, scale=(0,500), title="Raw Charge - T%d Evt%d" % (1, i), colorbar_title = "Charge [DC]")
#

