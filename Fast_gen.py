import numpy as np
import DriftDiffusionTool as DDtool

# What alpha values used
alphas = '0.8' # '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
# What system is beng solved
mode = 'Linear'
title = 'Localized'

#DDtool.createCumulativeSumOfDaysInData(StartMonth='Feb')

# Load data
weights_all = DDtool.calculatePCAweights('results/Europe_gHO1.0_aHO'+alphas+'_'+mode+'.npz')
weights_monthly_all,weights_daily_all,weights_hourly_all,detrended_weight = DDtool.detrendData(weights_all)
#DDtool.generateTransitionKDEs(weights_monthly_all,weights_daily_all,weights_hourly_all,detrended_weight,NumberOfComponents=1)


#Start = weights_monthly_all[:,0]
#Start = np.vstack((Start,weights_daily_all[:,0]))
#Start = np.vstack((Start,weights_hourly_all[:,0]))

Start = DDtool.createStartValueArrayFromDetrendData(weights_monthly_all,weights_daily_all,weights_hourly_all,year=0,StartMonth='Jan')

DDtool.generate_new_data('results/Europe_gHO1.0_aHO'+alphas+'_'+mode+'.npz',StartValues=Start,StartMonth='Jan',N = 1,NumberOfComponents=2)

#DDtool.generate_new_monthly_data(weights_monthly_all[:,0],N=24,NumberOfComponents=1)

#DDtool.generate_new_daily_data(weights_daily_all[0,0],N = 24,NumberOfComponents=1)

#DDtool.generate_new_hourly_data(weights_hourly_all[0,0],N = 24,NumberOfComponents=1)

#DDtool.addTimeScalesTogethor('results/Europe_gHO1.0_aHO'+alphas+'_'+mode+'.npz',N = 24,NumberOfComponents=1)