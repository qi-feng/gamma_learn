
from PyVAPlotSquareCam import *

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

#evt32 = df.iloc[[33]].values[0][504:504+500]
#
##drawEvent(evt32, scale=(0,500), title="Raw Charge - T%d" % (1), colorbar_title = "Charge [DC]")
#
#for i in [32, 35, 40, 43, 71]:
#    evt32 = df.iloc[[i]].values[0][504:504+500]
#    drawEvent(evt32, scale=(0,500), title="Raw Charge - T%d Evt%d" % (1, i), colorbar_title = "Charge [DC]")
#

