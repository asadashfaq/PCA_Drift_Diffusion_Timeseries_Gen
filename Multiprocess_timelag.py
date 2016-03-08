__author__ = 'Raunbak'
import numpy as np
import DriftDiffusionTool as DDtool
from multiprocessing import Process
# A SHOT AT MULTIPROCESSING WITH PYTHON .

if __name__ == '__main__':

    # What alpha values used
    alphas = '0.8' # '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    # What system is beng solved
    mode = 'Linear'
    title = 'Localized'

    weights_all = DDtool.calculatePCAweights('results/Europe_gHO1.0_aHO'+alphas+'_'+mode+'.npz')
    weights_monthly_all,weights_daily_all,weights_hourly_all,detrended_weight = DDtool.detrendData(weights_all)
    NumberOfComponents=1

    for deltat in [1,4]:

        timelag=deltat
        jobs = []
        for i in range(3):
            p = Process(target=DDtool.generateMismatchFromTimelagVersion3, args=('results/Europe_gHO1.0_aHO'+alphas+'_'+mode+'.npz',weights_monthly_all,weights_daily_all,weights_hourly_all,timelag,NumberOfComponents,))
            jobs.append(p)
            p.start()

        for j in jobs:
            j.join()
