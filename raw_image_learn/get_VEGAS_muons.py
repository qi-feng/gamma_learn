from VNN.utils.io import *
from VNN.utils.vegas_io import *
from VNN.utils.squarecam import *
from VNN.utils.image import *
#from PyVAPlotCam import PyVAPlotCam


#f = "/raid/biggams/qfeng/data/PG1553/81634UCORmedWinterMuonSt4.root"

def get_muon_charge(f, outfile_base="81635_muon", non_muon_outfile_base="81635_non_muon", num_of_non_muon=5000, dump=True,
                    outdir="muon_hunter_images", dpi=144):
    m_evtNums, m_tels, m_allCharges = read_muon_data(f, tels=[0, 1, 2, 3], read_charge=True, save_muon=True,
                                          outfile_base=outfile_base, save_non_muon=True, num_of_non_muon=num_of_non_muon,
                                          non_muon_outfile_base=non_muon_outfile_base)

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
            plt.savefig(outdir+"/"+str(outfile_base)+"_evt" + str(int(m_evtNums[i])) + "_tel"+str(m_tels[i])+".jpeg", dpi=dpi)
            #cam.show()


        for i in range(nm_allCharges.shape[-1]):
            cam = PyVAPlotCam(nm_allCharges[:, i], fig=plt.figure(figsize=(7, 7)), ax=plt.subplot(111))
            cam.draw(drawColorbar=False, draw_pixNum=False, cm=plt.cm.jet)
            plt.savefig(outdir+"/"+str(non_muon_outfile_base)+"_evt" + str(int(nm_evtNums[i])) + "_tel"+str(nm_tels[i])+".jpeg", dpi=dpi)

    return m_evtNums, m_tels, m_allCharges, nm_evtNums, nm_tels, nm_allCharges



