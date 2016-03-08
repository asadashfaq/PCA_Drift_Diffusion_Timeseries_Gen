__author__ = 'Raunbak'
import numpy as np
import dataLoader as loader
import PCA_tools as PCA
import pickle
import KDE_tool as KDE
import EUgrid as Grid
import aurespf.solvers as au
from scipy.stats import norm
from matplotlib.pylab import *
from scipy import stats
from scipy.stats import norm
"""
 This Tool is constructed for use on the ISET data files -
 Startdate of this data is 1/01/2000 (unsure of the year - ?)
 It is an 8 year dataset, in one hour indcrements
 To use this on another dataset - changes needs to be made.
"""

# Globals used :
samplerate = 500
tau = 1

def calculatePCAweights(filename):
    """
    From a solved network - calculate the PCA weights
    :param filename:
    :return:
    """
    # Load the data from a solved system
    mismatch = loader.get_eu_mismatch_balancing_injection_meanload(filename)[0]

    # Total number of days in the data [hours/24]
    numberOfDays = mismatch.shape[1] / 24

    # Center and normalize data, for use with PCA
    mismatch_c, mean_mismatch = PCA.center(mismatch)
    h, Ntilde = PCA.normalize(mismatch_c)

    # Arrays to store data in.
    weights_all = np.zeros([30, len(mismatch[1])])
    weightsDaily_all = np.zeros([30, numberOfDays])
    weight_monthly_all = np.zeros([30, numberOfDays])

    # Calculating the amplitude of each PC
    for j in range(30):
        weights_all[j, :] = PCA.get_xi_weight(h, j)

    return weights_all


def detrendData(weights):
    """
    Detrends the PCA weights in my defined way - semi-monthly - daily - hourly
    :param weights:
    :return:
    """

    CumSumOfMonthsInDays,CumSumOfMonthsInHours = createCumulativeSumOfDaysInData()

    numberOfDays = weights.shape[1] / 24

    # Data structures for the detrended weights
    # Monthly : 30x192
    weights_monthly_all = np.zeros([30, len(CumSumOfMonthsInHours)])
    # Daily : 30x2092
    weights_daily_all = np.zeros([30, numberOfDays])
    # Hourly : 30x70128
    weights_hourly_all = weights.copy()

    # Detrending on the monthly scale ("15" day avgs)
    for i, day in enumerate(CumSumOfMonthsInHours, start=0):
        if i == 0:
            span = np.arange(0, day)
        else:
            span = np.arange(CumSumOfMonthsInHours[i-1], day)

        # Calculated the next point, as the hourly avg over "15" days.
        weights_monthly_all[:, i] = np.sum(weights[:, span],axis=1) / (len(span))



    # Detrending on the daily scale (24 hour avg)
    detrended_weight = weights.copy()

    # First we do detrend the monthly scale data from the original data
    month = 0
    for i in range(len(weights[1])):
        if i > CumSumOfMonthsInHours[month]:
            month += 1
        detrended_weight[:,i] -= weights_monthly_all[:,month]

    # Then calcuted the avg of a day.
    for i in range(numberOfDays):
        span = np.arange(i*24, ((i+1)*24)) # Get a 24 hour slice. 0:23 -> 24:47 -> 48:.....
        weights_daily_all[:,i] = np.sum(detrended_weight[:,span],axis=1) / len(span)

    # Detrending on the hourly scale
    # The every 24 hours the day index changes, and every "15" days in hours the month index changes.
        # Hourly : 30x70128
    month = 0
    day = -1
    for i in range(len(weights[1])):
        if i % 24 == 0:
            day += 1
        if i > CumSumOfMonthsInHours[month]:
            month += 1
        weights_hourly_all[:,i] -= weights_monthly_all[:,month]
        weights_hourly_all[:,i] -= weights_daily_all[:,day]

    # Just a check that everything is in order.
    assert weights[0,0] == (weights_monthly_all[0,0]+weights_daily_all[0,0]+weights_hourly_all[0,0])
    assert weights[0,0] == (weights_monthly_all[0,0]+(np.sum(detrended_weight[0,np.arange(0,24)]) / 24)+weights_hourly_all[0,0])

    return weights_monthly_all,weights_daily_all,weights_hourly_all,detrended_weight


def generateTransitionKDEs(weights_monthly_all,weights_daily_all,weights_hourly_all,detrended_weight,NumberOfComponents=1):
    """
    This takes a long time. (The hourly part does)
    Takes detrended data and creates transition KDEs to be used in time series generation
    A KDE is created for each month in a year
    Results are saved in pkl files.
    :param weights_monthly_all:
    :param weights_daily_all:
    :param weights_hourly_all:
    :param detrended_weight:
    :param NumberOfComponents:
    :return:
    """
    numberOfDays = weights_daily_all.shape[1]
    CumSumOfMonthsInDays,CumSumOfMonthsInHours = createCumulativeSumOfDaysInData()

    for component in range(NumberOfComponents):

        #######################################################################################
        ####################### Monthly KDE ###################################################
        #######################################################################################
        # Data strucktures
        kdes = {}
        max_values = []
        min_values = []
        value_intervals = np.zeros([12,samplerate]) #

        # Pick out the samples for the current component, reshape into a 8x24 array
        Monthly_samples = np.reshape(weights_monthly_all[component,:],(8, 24))

        # Begin the calculation of transition KDEs -
        x = 0
        for i in np.arange(0,24,2):
            if i == 22:
                weight_monthly = np.append(Monthly_samples[:,i],Monthly_samples[:,i+1])
                weight_monthly_trans = np.append(Monthly_samples[:,i+1],Monthly_samples[:,0])

            else:
                weight_monthly = np.append(Monthly_samples[:,i],Monthly_samples[:,i+1])
                weight_monthly_trans = np.append(Monthly_samples[:,i+1],Monthly_samples[:,i+2])

            xmin = weight_monthly.min()
            xmax = weight_monthly.max()
            ymin = weight_monthly_trans.min()
            ymax = weight_monthly_trans.max()

            values = np.vstack([weight_monthly, weight_monthly_trans])
            kernel = stats.gaussian_kde(values)

            kdes[x] = kernel


            # Defining the sample range, and number of samples
            max_value = np.max(weight_monthly) + np.max(weight_monthly)/10  # Max value plus 10%. THIS IS JUST AN ESTIMATE
            min_value = np.min(weight_monthly) + np.min(weight_monthly)/10  # Min value plus 10%. JUST AN ESTIMATE
            value_interval = np.linspace(min_value, max_value, samplerate)

            max_values = np.append(max_values,max_value)
            min_values = np.append(min_values,min_value)
            value_intervals[x,:] = value_interval
            x = x + 1

            X, Y = np.mgrid[-2:2:500j, -2:2:500j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kernel(positions).T, X.shape)
            fig, ax = plt.subplots()
            ax.imshow(np.rot90(Z),extent=[-2, 2, -2, 2])
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.plot(weight_monthly, weight_monthly_trans, 'k.')
            fig.savefig('results/figures/' + 'Semi-Month'+str(i)+'-k='+str(component)+'KDE' + '.pdf')
            plt.close(fig)



        Monthly_data = {'max': max_values,'min':min_values,'interval': value_intervals,'kde':kdes}
        with open('monthly_kde_full_k='+str(component+1)+'.pkl', 'wb') as f:
            pickle.dump(Monthly_data, f, pickle.HIGHEST_PROTOCOL)

        ########################################################################################
        ########################## Now to the daily KDE ########################################
        ########################################################################################

        Daily_Month_dic = createDictioariesOfMonthsInHours(np.reshape(detrended_weight[component,:],(numberOfDays, 24)),CumSumOfMonthsInHours)
        kdes = {}
        max_values = []
        min_values = []
        value_intervals = np.zeros([12,500])
        x = 0
        for key in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Okt','Nov','Dec']:
        #     print key
            months = Daily_Month_dic[key]
            daily = np.sum(months,axis=1)/24

            # Here the transition is from one day to the next, with the last day transition to the first in the arry.
            weight_daily = daily
            weight_daily_trans = np.roll(daily,-1)

            xmin = weight_daily.min()
            xmax = weight_daily.max()
            ymin = weight_daily_trans.min()
            ymax = weight_daily_trans.max()

            values = np.vstack([weight_daily, weight_daily_trans])
            kernel = stats.gaussian_kde(values)

            kdes[x] = kernel

            # Defining the sample range, and number of samples
            max_value = np.max(weight_daily) + np.max(weight_daily)/10  # Max value plus 10%. THIS IS JUST AN ESTIMATE
            min_value = np.min(weight_daily) + np.min(weight_daily)/10  # Min value plus 10%. JUST AN ESTIMATE
            value_interval = np.linspace(min_value, max_value, samplerate)

            max_values = np.append(max_values,max_value)
            min_values = np.append(min_values,min_value)
            value_intervals[x,:] = value_interval
            x = x + 1


            X, Y = np.mgrid[-2:2:500j, -2:2:500j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kernel(positions).T, X.shape)
            fig, ax = plt.subplots()
            ax.imshow(np.rot90(Z),extent=[-2, 2, -2, 2])
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])

            ax.plot(weight_daily, weight_daily_trans, 'k.')
            fig.savefig('results/figures/' + key+'-daily''-k='+str(component)+'KDE' + '.pdf')
            plt.close(fig)


        # Save to file - saving the max, min, value interval and the kde.
        daily_data = {'max': max_values,'min':min_values,'interval': value_intervals,'kde':kdes}
        with open('daily_kde_full_k='+str(component+1)+'.pkl', 'wb') as f:
            pickle.dump(daily_data, f, pickle.HIGHEST_PROTOCOL)

        ########################################################################################
        ################# Now to the hourly KDE (The slowest one) ##############################
        ########################################################################################

        Month_dic = createDictioariesOfMonthsInHours(np.reshape(weights_hourly_all[component,:],(numberOfDays, 24)),CumSumOfMonthsInHours)

        hourly_kdes = {}
        hourly_max_values = []
        hourly_min_values = []
        hourly_value_intervals = np.zeros([24*12,500])

        x = 0

        for key in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Okt','Nov','Dec']:
            months = Month_dic[key]
            for i in range(24):
                if i == 23:
                    m1 = months[:,i]
                    m2 = months[:,0]
                else:
                    m1 = months[:,i]
                    m2 = months[:,i+1]

                xmin = m1.min()
                xmax = m2.max()
                ymin = m2.min()
                ymax = m2.max()

                values = np.vstack([m1, m2])
                kernel = stats.gaussian_kde(values)

                hourly_kdes [x] = kernel


                # Defining the sample range, and number of samples
                max_value = np.max(months[:,i]) + np.max(months[:,i])/10  # Max value plus 10%. THIS IS JUST AN ESTIMATE
                min_value = np.min(months[:,i]) + np.min(months[:,i])/10  # Min value plus 10%. JUST AN ESTIMATE
                value_interval = np.linspace(min_value, max_value, samplerate)

                hourly_max_values = np.append(hourly_max_values ,max_value)
                hourly_min_values = np.append(hourly_min_values ,min_value)
                hourly_value_intervals[x,:] = value_interval
                x = x + 1


                X, Y = np.mgrid[-1:1:500j, -1:1:500j]
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.reshape(kernel(positions).T, X.shape)
                fig, ax = plt.subplots()
                ax.imshow(np.rot90(Z),extent=[-1, 1, -1, 1])
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.plot(m1, m2, 'k.')
                fig.savefig('results/figures/' + key+'-Hourly-'+str(i)+'-k='+str(component)+'KDE' + '.pdf')
                plt.close(fig)

                # X, Y = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]
                # positions = np.vstack([X.ravel(), Y.ravel()])
                # Z = np.reshape(kernel(positions).T, X.shape)
                #
                # fig, ax = plt.subplots()
                # ax.imshow(np.rot90(Z),extent=[xmin, xmax, ymin, ymax])
                # if i == 23:
                #     ax.plot(months[:,i],months[:,0], 'k.')
                # else:
                #     ax.plot(months[:,i],months[:,i+1], 'k.')
                # ax.set_ylabel(r'$a(t+\tau)$')
                # ax.set_xlabel(r'$a(t)$')
                # ax.set_xlim([-1,1])
                # ax.set_ylim([-1,1])
                # fig.savefig('results/figures/' + key+str(i)+'KDE' + '.pdf')
                #
                # print 'results/figures/' + key+str(i)+'KDE' + '.pdf'+'  SAVED'


        Hourly_data = {'max':  hourly_max_values,'min': hourly_min_values,'interval': hourly_value_intervals,'kde':hourly_kdes}
        with open('hourly_kde_full_k='+str(component+1)+'.pkl', 'wb') as f:
           pickle.dump(Hourly_data, f, pickle.HIGHEST_PROTOCOL)


def generate8YearMismatchData(filename,weights_monthly_all,weights_daily_all,weights_hourly_all,NumberOfComponents=1):
    # Data strucktues
    T = len(weights_hourly_all[0,:])
    a_h = np.zeros((NumberOfComponents,T))
    a_d = np.zeros((NumberOfComponents,T))
    a_m = np.zeros((NumberOfComponents,T))

    # Setting starting values
    a_h[:,0] = weights_hourly_all[0:NumberOfComponents,0]
    a_d[:,0] = weights_daily_all[0:NumberOfComponents,0]
    a_m[:,0] = weights_monthly_all[0:NumberOfComponents,0]

    hours_in_month = createCumulativeSumOfDaysInData(StartMonth='Jan')[1][1::2]
    hours_in_semi_month = createCumulativeSumOfDaysInData(StartMonth='Jan')[1]

    # Load the data from a solved system
    mismatch = loader.get_eu_mismatch_balancing_injection_meanload(filename)[0]

    # Center and normalize data, for use with PCA
    mismatch_c, mean_mismatch = PCA.center(mismatch)
    h, Ntilde = PCA.normalize(mismatch_c)

    # N is the  number of hours
    epsilon = np.zeros((NumberOfComponents,T))

    # We have a network of 30 nodes, so the new mismatch needs to be 30xN+1
    approx_mismatch = np.zeros((mismatch.shape[0],T))

    # Loop over number of components
    for component in range(NumberOfComponents):
        # Load relevant kdes and associated  values.
        with open('hourly_kde_full_k='+str(component+1)+'.pkl', 'rb') as f:
            Hourly_data = pickle.load(f)

        kdes_h = Hourly_data['kde']
        max_values_h = Hourly_data['max']
        min_values_h = Hourly_data['min']
        value_intervals_h = Hourly_data['interval']

        with open('daily_kde_full_k='+str(component+1)+'.pkl', 'rb') as f:
            Daily_data = pickle.load(f)

        kdes_d = Daily_data['kde']
        max_value_d = Daily_data['max']
        min_value_d = Daily_data['min']
        value_interval_d = Daily_data['interval']

        with open('monthly_kde_full_k='+str(component+1)+'.pkl', 'rb') as f:
                Monthly_data = pickle.load(f)

        kdes_m = Monthly_data['kde']
        max_values_m = Monthly_data['max']
        min_values_m = Monthly_data['min']
        value_intervals_m = Monthly_data['interval']


        # Begin to create generated values
        daily_amplitude = weights_daily_all[component,0]
        semi_monthly_amplitude = weights_monthly_all[component,0]
        day = 0
        semi_month = 0
        month = 0
        year = 0
        time_of_day = 0 # Since we use the real data values for hour 0, the generated data is
        for hour in range(1,T):

            #print time_of_day,'Time of day'
            #print hour,'Hour'
            # Keeping track on wat month and year we are in when generating data
            if hour == hours_in_month[12*year+month]:
                month += 1

            if month == 12:
                month = 0
                year += 1

            samples = np.vstack([np.repeat(a_h[component,hour-1],samplerate), np.linspace(min_values_h[(24*month)+time_of_day], max_values_h[(24*month)+time_of_day], samplerate)])
            pdf = kdes_h[(24*month)+time_of_day].evaluate(samples)

            drift = KDE.kramer_moyal_coeff(a_h[component,hour-1], value_intervals_h[(24*month)+time_of_day,:], pdf, n=1, tau=1)
            diffusion = KDE.kramer_moyal_coeff(a_h[component,hour-1], value_intervals_h[(24*month)+time_of_day,:], pdf, n=2, tau=1)

            p = norm.rvs(loc=0, scale=1)

            a_h[component,hour] = a_h[component,hour-1] + drift*tau + np.sqrt(diffusion*tau)*p

            time_of_day += 1
            if time_of_day == 24:
                time_of_day = 0

            # Check if the new value is "legal"
            samples = np.vstack([np.repeat(a_h[component,hour],samplerate), np.linspace(min_values_h[(24*month)+time_of_day], max_values_h[(24*month)+time_of_day], samplerate)])
            pdf = kdes_h[(24*month)+time_of_day].evaluate(samples)

            iteration = 0
            if np.sum(pdf) == 0.0:
                # print 'out', i
                while np.sum(pdf) == 0.0:

                    p = norm.rvs(loc=0, scale=1)

                    a_h[component,hour] = a_h[component,hour-1] + drift*tau + np.sqrt(diffusion*tau)*p

                    samples = np.vstack([np.repeat(a_h[component,hour],samplerate), np.linspace(min_values_h[(24*month)+time_of_day], max_values_h[(24*month)+time_of_day], samplerate)])
                    pdf = kdes_h[(24*month)+time_of_day].evaluate(samples)
                    iteration += 1
                    if iteration > 1000:
                        print('Too many iterations')
                        break

            ############################################################################
            ############################## DAILY PART ##################################
            ############################################################################
            # 24 hours has passed we need a new daily value -
            if hour % 24.0 == 0:
                samples = np.vstack([np.repeat(a_d[component,hour],samplerate), np.linspace(min_value_d[month], max_value_d[month], samplerate)])
                pdf = kdes_d[month].evaluate(samples)

                drift = KDE.kramer_moyal_coeff(a_d[component,hour], value_interval_d[month,:], pdf, n=1, tau=1)
                diffusion = KDE.kramer_moyal_coeff(a_d[component,hour], value_interval_d[month,:], pdf, n=2, tau=1)

                p = norm.rvs(loc=0, scale=1)

                daily_amplitude = a_d[component,hour] + drift*tau + np.sqrt(diffusion*tau)*p

                samples = np.vstack([np.repeat(daily_amplitude,samplerate), np.linspace(min_value_d[month], max_value_d[month], samplerate)])
                pdf = kdes_d[month].evaluate(samples)

                iteration = 0
                if np.sum(pdf) == 0.0:

                    while np.sum(pdf) == 0.0:

                        p = norm.rvs(loc=0, scale=1)

                        daily_amplitude = a_d[component,hour] + drift*tau + np.sqrt(diffusion*tau)*p

                        samples = np.vstack([np.repeat(daily_amplitude,samplerate), np.linspace(min_value_d[month], max_value_d[month], samplerate)])
                        pdf = kdes_d[month].evaluate(samples)

                        iteration += 1
                        if iteration > 1000:
                            print('Too many iterations')
                            break

            # Write the daily value to the daily amplitude array
            a_d[component,hour] = daily_amplitude

            ############################################################################
            ################## MONTHLY PART ############################################
            ############################################################################
            if hour == hours_in_semi_month[semi_month]:
                semi_month += 1

                samples = np.vstack([np.repeat(a_m[component,hour-1],samplerate), np.linspace(min_values_m[month], max_values_m[month], samplerate)])
                pdf = kdes_m[month].evaluate(samples)

                drift = KDE.kramer_moyal_coeff(a_m[component,hour-1], value_intervals_m[month,:], pdf, n=1, tau=1)
                diffusion = KDE.kramer_moyal_coeff(a_m[component,hour-1], value_intervals_m[month,:], pdf, n=2, tau=1)

                p = norm.rvs(loc=0, scale=1)

                semi_monthly_amplitude = a_m[component,hour-1] + drift*tau + np.sqrt(diffusion*tau)*p

                samples = np.vstack([np.repeat(semi_monthly_amplitude,samplerate), np.linspace(min_values_m[month], max_values_m[month], samplerate)])
                pdf = kdes_m[month].evaluate(samples)

                # Check if we go out of bounds, then find a new random value which would bring us back.
                iteration = 0
                if np.sum(pdf) == 0.0:
                    #print 'out', i
                    while np.sum(pdf) == 0.0:

                        p = norm.rvs(loc=0, scale=1)

                        semi_monthly_amplitude = a_m[component,hour-1] + drift*tau + np.sqrt(diffusion*tau)*p

                        samples = np.vstack([np.repeat(semi_monthly_amplitude,samplerate), np.linspace(min_values_m[month], max_values_m[month], samplerate)])
                        pdf = kdes_m[month].evaluate(samples)

                        iteration += 1
                        if iteration > 1000:
                            print('Too many iterations')
                            break

            a_m[component,hour] = semi_monthly_amplitude

            if hour%10000 == 0:
                print '10k hours generated'

        print 'Mean values'
        print np.mean(a_h[component,:])
        print np.mean(a_d[component,:])
        print np.mean(a_m[component,:])
        a_h[component,:] = a_h[component,:] - np.mean(a_h[component,:])
        a_d[component,:] = a_d[component,:] - np.mean(a_d[component,:])
        a_m[component,:] = a_m[component,:] - np.mean(a_m[component,:])

        for i in np.arange(0, T, tau):
            epsilon[component,i] = a_m[component,i] + a_d[component,i] + a_h[component,i] # The +1 is to skip the startvalue of epsilon_h
            if i == 0:
                print epsilon[component,i]
                print PCA.get_xi_weight(h, component)[0]

        lambd, princ_comp = PCA.get_principal_component(h, component)
        mismatch_PC = PCA.unnormalize_uncenter(princ_comp,Ntilde, mean_mismatch)

        approx_mismatch += np.outer(mismatch_PC, epsilon[component,:])

    filename = 'weights_generated.npy'
    np.save('approx_mismatch_generated'+'.npy',approx_mismatch)
    np.save(filename, epsilon)


def generateMismatchFromTimelag(filename,weights_monthly_all,weights_daily_all,weights_hourly_all,timelag=1,NumberOfComponents=1):

    # Data strucktues
    T = len(weights_hourly_all[0,:])
    a_h = np.zeros((NumberOfComponents,T))
    a_d = np.zeros((NumberOfComponents,T))
    a_m = np.zeros((NumberOfComponents,T))

    # Setting starting values
    a_h[:,0] = weights_hourly_all[0:NumberOfComponents,0]
    a_d[:,0] = weights_daily_all[0:NumberOfComponents,0]
    a_m[:,0] = weights_monthly_all[0:NumberOfComponents,0]

    hours_in_month = createCumulativeSumOfDaysInData(StartMonth='Jan')[1][1::2]
    hours_in_semi_month = createCumulativeSumOfDaysInData(StartMonth='Jan')[1]

    # Load the data from a solved system
    mismatch = loader.get_eu_mismatch_balancing_injection_meanload(filename)[0]

    # Center and normalize data, for use with PCA
    mismatch_c, mean_mismatch = PCA.center(mismatch)
    h, Ntilde = PCA.normalize(mismatch_c)

    # N is the  number of hours
    epsilon = np.zeros((NumberOfComponents,T))

    # We have a network of 30 nodes, so the new mismatch needs to be 30xN+1
    approx_mismatch = np.zeros((mismatch.shape[0],T))

    # Loop over number of components
    for component in range(NumberOfComponents):

        # Load relevant kdes and associated  values.
        with open('hourly_kde_full_k='+str(component+1)+'.pkl', 'rb') as f:
            Hourly_data = pickle.load(f)

        kdes_h = Hourly_data['kde']
        max_values_h = Hourly_data['max']
        min_values_h = Hourly_data['min']
        value_intervals_h = Hourly_data['interval']

        numberOfkdes = len(Hourly_data['kde'])

        with open('daily_kde_full_k='+str(component+1)+'.pkl', 'rb') as f:
            Daily_data = pickle.load(f)

        kdes_d = Daily_data['kde']
        max_value_d = Daily_data['max']
        min_value_d = Daily_data['min']
        value_interval_d = Daily_data['interval']

        with open('monthly_kde_full_k='+str(component+1)+'.pkl', 'rb') as f:
                Monthly_data = pickle.load(f)

        kdes_m = Monthly_data['kde']
        max_values_m = Monthly_data['max']
        min_values_m = Monthly_data['min']
        value_intervals_m = Monthly_data['interval']

        # Begin to create generated values
        daily_amplitude = weights_daily_all[component,0]
        semi_monthly_amplitude = weights_monthly_all[component,0]
        UpdateDaily_because_of_timelag = False
        UpdateSemiMonthly_because_of_timelag = False
        day = 0
        semi_month = 0
        month = 0
        year = 0
        time_of_day = 0 # Since we use the real data values for hour 0, the generated data is

        # Setting the first element in the array to be equal to the data, undtill we reach the first valid point for timelag.
        for hour in range(0,timelag):
            a_h[component,hour] = weights_hourly_all[component,hour]
            a_d[component,hour] = weights_daily_all[component,hour]
            a_m[component,hour] = weights_monthly_all[component,hour]

        for hour in range(timelag,T):
        # Keeping track on wat month and year we are in when generating data
            if hour > hours_in_month[12*year+month]:
                month += 1

            if month == 12:
                month = 0
                year += 1
            # Setting the a_h value to the value of the data where we wish to start generating from.
            a_h[component,hour] = weights_hourly_all[component,hour-timelag]

            for i in range(timelag):

                h = ((24*month)+time_of_day+i)
                if ((24*month)+time_of_day+i) >= numberOfkdes:
                    h = 0 + ((24*month)+time_of_day+i)%numberOfkdes

                samples = np.vstack([np.repeat(a_h[component,hour],samplerate), np.linspace(min_values_h[h], max_values_h[h], samplerate)])
                pdf = kdes_h[h].evaluate(samples)


                drift = KDE.kramer_moyal_coeff(a_h[component,hour], value_intervals_h[h,:], pdf, n=1, tau=1)
                diffusion = KDE.kramer_moyal_coeff(a_h[component,hour], value_intervals_h[h,:], pdf, n=2, tau=1)

                p = norm.rvs(loc=0, scale=1)

                hourly_amp = a_h[component,hour] + drift*tau + np.sqrt(diffusion*tau)*p
                next_h = h+1

                if next_h >= numberOfkdes:
                    next_h = 0 + ((24*month)+time_of_day+i+1)%numberOfkdes
                # Check if the new value is "legal" in the next hour
                samples = np.vstack([np.repeat(hourly_amp,samplerate), np.linspace(min_values_h[next_h], max_values_h[next_h], samplerate)])
                pdf = kdes_h[next_h].evaluate(samples)

                iteration = 0
                if np.sum(pdf) == 0.0:
                    # print 'out', i
                    while np.sum(pdf) == 0.0:

                        p = norm.rvs(loc=0, scale=1)

                        hourly_amp = a_h[component,hour] + drift*tau + np.sqrt(diffusion*tau)*p

                        samples = np.vstack([np.repeat(hourly_amp,samplerate), np.linspace(min_values_h[next_h], max_values_h[next_h], samplerate)])
                        pdf = kdes_h[next_h].evaluate(samples)

                        iteration += 1
                        if iteration > 1000:
                            print('Too many iterations')
                            break
                # Updates the values in a_h, this is done 'timelag' times.
                a_h[component,hour] = hourly_amp

            # The next hour is a new tme of day.
            time_of_day += 1
            if time_of_day == 24:
                time_of_day = 0

            ############################################################################
            ############################## DAILY PART ##################################
            ############################################################################
            if hour % 24.0 == 0 and hour != 0:
                # Calculate what day we are in.
                day = hour / 24.0

                # If the timelag fits with the 24 hours, we know everything about the last day - we then generate a new value from the real data.
                if hour % timelag == 0:
                    UpdateDaily_because_of_timelag = False
                    samples = np.vstack([np.repeat(weights_daily_all[component,day-1],samplerate), np.linspace(min_value_d[month], max_value_d[month], samplerate)])
                    pdf = kdes_d[month].evaluate(samples)

                    drift = KDE.kramer_moyal_coeff(weights_daily_all[component,day-1], value_interval_d[month,:], pdf, n=1, tau=1)
                    diffusion = KDE.kramer_moyal_coeff(weights_daily_all[component,day-1], value_interval_d[month,:], pdf, n=2, tau=1)

                    p = norm.rvs(loc=0, scale=1)

                    daily_amplitude = weights_daily_all[component,day-1] + drift*tau + np.sqrt(diffusion*tau)*p

                    # Check if the next hour is in a new month, needed to check if the next value is legal.
                    next_m = month
                    if (hour+1) > hours_in_month[12*year+month]:
                        next_m = month + 1
                        if next_m == 12:
                            next_m = 0
                    samples = np.vstack([np.repeat(daily_amplitude,samplerate), np.linspace(min_value_d[next_m], max_value_d[next_m], samplerate)])
                    pdf = kdes_d[next_m].evaluate(samples)

                    iteration = 0
                    if np.sum(pdf) == 0.0:

                        while np.sum(pdf) == 0.0:

                            p = norm.rvs(loc=0, scale=1)

                            daily_amplitude = weights_daily_all[component,day-1] + drift*tau + np.sqrt(diffusion*tau)*p

                            samples = np.vstack([np.repeat(daily_amplitude,samplerate), np.linspace(min_value_d[next_m], max_value_d[next_m], samplerate)])
                            pdf = kdes_d[next_m].evaluate(samples)

                            iteration += 1
                            if iteration > 1000:
                                print('Too many iterations')
                                break

                # If we are not on the timelag then generate the next daily value from the generated series.
                # Will be updated when we hit the timelag.
                else:
                    UpdateDaily_because_of_timelag = True
                    samples = np.vstack([np.repeat(a_d[component,hour],samplerate), np.linspace(min_value_d[month], max_value_d[month], samplerate)])
                    pdf = kdes_d[month].evaluate(samples)

                    drift = KDE.kramer_moyal_coeff(a_d[component,hour], value_interval_d[month,:], pdf, n=1, tau=1)
                    diffusion = KDE.kramer_moyal_coeff(a_d[component,hour], value_interval_d[month,:], pdf, n=2, tau=1)

                    p = norm.rvs(loc=0, scale=1)

                    daily_amplitude = a_d[component,hour] + drift*tau + np.sqrt(diffusion*tau)*p

                    # Check if next hour is in a new month
                    next_m = month
                    if (hour+1) > hours_in_month[12*year+month]:
                        next_m = month + 1
                        if next_m == 12:
                            next_m = 0

                    samples = np.vstack([np.repeat(daily_amplitude,samplerate), np.linspace(min_value_d[next_m], max_value_d[next_m], samplerate)])
                    pdf = kdes_d[next_m].evaluate(samples)

                    iteration = 0
                    if np.sum(pdf) == 0.0:

                        while np.sum(pdf) == 0.0:

                            p = norm.rvs(loc=0, scale=1)

                            daily_amplitude = a_d[component,hour] + drift*tau + np.sqrt(diffusion*tau)*p

                            samples = np.vstack([np.repeat(daily_amplitude,samplerate), np.linspace(min_value_d[next_m], max_value_d[next_m], samplerate)])
                            pdf = kdes_d[next_m].evaluate(samples)

                            iteration += 1
                            if iteration > 1000:
                                print('Too many iterations')
                                break

                 # Write the daily value to the daily amplitude array
                a_d[component,hour] = daily_amplitude

                        # Update the daily value
            if UpdateDaily_because_of_timelag and hour % timelag == 0:

                UpdateDaily_because_of_timelag = False

                samples = np.vstack([np.repeat(weights_daily_all[component,day-1],samplerate), np.linspace(min_value_d[month], max_value_d[month], samplerate)])
                pdf = kdes_d[month].evaluate(samples)

                drift = KDE.kramer_moyal_coeff(weights_daily_all[component,day-1], value_interval_d[month,:], pdf, n=1, tau=1)
                diffusion = KDE.kramer_moyal_coeff(weights_daily_all[component,day-1], value_interval_d[month,:], pdf, n=2, tau=1)

                p = norm.rvs(loc=0, scale=1)

                daily_amplitude = weights_daily_all[component,day-1] + drift*tau + np.sqrt(diffusion*tau)*p

                # Check if the next hour is in a new month, needed to check if the next value is legal.
                next_m = month
                if (hour+1) > hours_in_month[12*year+month]:
                    next_m = month + 1
                    if next_m == 12:
                        next_m = 0

                samples = np.vstack([np.repeat(daily_amplitude,samplerate), np.linspace(min_value_d[next_m], max_value_d[next_m], samplerate)])
                pdf = kdes_d[next_m].evaluate(samples)

                iteration = 0
                if np.sum(pdf) == 0.0:

                    while np.sum(pdf) == 0.0:

                        p = norm.rvs(loc=0, scale=1)

                        daily_amplitude = weights_daily_all[component,day-1] + drift*tau + np.sqrt(diffusion*tau)*p

                        samples = np.vstack([np.repeat(daily_amplitude,samplerate), np.linspace(min_value_d[next_m], max_value_d[next_m], samplerate)])
                        pdf = kdes_d[next_m].evaluate(samples)

                        iteration += 1
                        if iteration > 1000:
                            print('Too many iterations')
                            break

            # Write the daily value to the daily amplitude array
            a_d[component,hour] = daily_amplitude
            ############################################################################
            ################## MONTHLY PART ############################################
            ############################################################################

            if hour == hours_in_semi_month[semi_month]:
                semi_month+= 1

                if hour % timelag == 0:
                    UpdateSemiMonthly_because_of_timelag = False

                    samples = np.vstack([np.repeat(weights_monthly_all[component,semi_month-1],samplerate), np.linspace(min_values_m[month], max_values_m[month], samplerate)])
                    pdf = kdes_m[month].evaluate(samples)

                    drift = KDE.kramer_moyal_coeff(weights_monthly_all[component,semi_month-1], value_intervals_m[month,:], pdf, n=1, tau=1)
                    diffusion = KDE.kramer_moyal_coeff(weights_monthly_all[component,semi_month-1], value_intervals_m[month,:], pdf, n=2, tau=1)

                    p = norm.rvs(loc=0, scale=1)

                    semi_monthly_amplitude = weights_monthly_all[component,semi_month-1] + drift*tau + np.sqrt(diffusion*tau)*p

                    # Check if the next hour is in a new month, needed to check if the next value is legal.
                    next_m = month
                    if (hour+1) > hours_in_month[12*year+month]:
                        next_m = month + 1
                        if next_m == 12:
                            next_m = 0
                    samples = np.vstack([np.repeat(semi_monthly_amplitude,samplerate), np.linspace(min_values_m[next_m], max_values_m[next_m], samplerate)])
                    pdf = kdes_m[next_m].evaluate(samples)

                    # Check if we go out of bounds, then find a new random value which would bring us back.
                    iteration = 0
                    if np.sum(pdf) == 0.0:
                        while np.sum(pdf) == 0.0:

                            p = norm.rvs(loc=0, scale=1)

                            semi_monthly_amplitude = weights_monthly_all[component,semi_month-1] + drift*tau + np.sqrt(diffusion*tau)*p

                            samples = np.vstack([np.repeat(semi_monthly_amplitude,samplerate), np.linspace(min_values_m[next_m], max_values_m[next_m], samplerate)])
                            pdf = kdes_m[next_m].evaluate(samples)

                            iteration += 1
                            if iteration > 1000:
                                print('Too many iterations')
                                break
                else:
                    UpdateSemiMonthly_because_of_timelag = True
                    samples = np.vstack([np.repeat(a_m[component,hour-1],samplerate), np.linspace(min_values_m[month], max_values_m[month], samplerate)])
                    pdf = kdes_m[month].evaluate(samples)

                    drift = KDE.kramer_moyal_coeff(a_m[component,hour-1], value_intervals_m[month,:], pdf, n=1, tau=1)
                    diffusion = KDE.kramer_moyal_coeff(a_m[component,hour-1], value_intervals_m[month,:], pdf, n=2, tau=1)

                    p = norm.rvs(loc=0, scale=1)

                    semi_monthly_amplitude = a_m[component,hour-1] + drift*tau + np.sqrt(diffusion*tau)*p

                    # Check if the next hour is in a new month, needed to check if the next value is legal.
                    next_m = month
                    if (hour+1) > hours_in_month[12*year+month]:
                        next_m = month + 1
                        if next_m == 12:
                            next_m = 0

                    samples = np.vstack([np.repeat(semi_monthly_amplitude,samplerate), np.linspace(min_values_m[next_m], max_values_m[next_m], samplerate)])
                    pdf = kdes_m[next_m].evaluate(samples)

                    # Check if we go out of bounds, then find a new random value which would bring us back.
                    iteration = 0
                    if np.sum(pdf) == 0.0:
                        #print 'out', i
                        while np.sum(pdf) == 0.0:

                            p = norm.rvs(loc=0, scale=1)

                            semi_monthly_amplitude = a_m[component,hour-1] + drift*tau + np.sqrt(diffusion*tau)*p

                            samples = np.vstack([np.repeat(semi_monthly_amplitude,samplerate), np.linspace(min_values_m[next_m], max_values_m[next_m], samplerate)])
                            pdf = kdes_m[next_m].evaluate(samples)

                            iteration += 1
                            if iteration > 1000:
                                print('Too many iterations')
                                break

            if UpdateSemiMonthly_because_of_timelag and hour % timelag == 0:
                UpdateSemiMonthly_because_of_timelag = False

                samples = np.vstack([np.repeat(weights_monthly_all[component,semi_month-1],samplerate), np.linspace(min_values_m[month], max_values_m[month], samplerate)])
                pdf = kdes_m[month].evaluate(samples)

                drift = KDE.kramer_moyal_coeff(weights_monthly_all[component,semi_month-1], value_intervals_m[month,:], pdf, n=1, tau=1)
                diffusion = KDE.kramer_moyal_coeff(weights_monthly_all[component,semi_month-1], value_intervals_m[month,:], pdf, n=2, tau=1)

                p = norm.rvs(loc=0, scale=1)

                semi_monthly_amplitude = weights_monthly_all[component,semi_month-1] + drift*tau + np.sqrt(diffusion*tau)*p
                # Check if the next hour is in a new month, needed to check if the next value is legal.
                next_m = month
                if (hour+1) > hours_in_month[12*year+month]:
                    next_m = month + 1
                    if next_m == 12:
                        next_m = 0

                samples = np.vstack([np.repeat(semi_monthly_amplitude,samplerate), np.linspace(min_values_m[next_m], max_values_m[next_m], samplerate)])
                pdf = kdes_m[next_m].evaluate(samples)

                # Check if we go out of bounds, then find a new random value which would bring us back.
                iteration = 0
                if np.sum(pdf) == 0.0:
                    #print 'out', i
                    while np.sum(pdf) == 0.0:

                        p = norm.rvs(loc=0, scale=1)

                        semi_monthly_amplitude = weights_monthly_all[component,semi_month-1] + drift*tau + np.sqrt(diffusion*tau)*p

                        samples = np.vstack([np.repeat(semi_monthly_amplitude,samplerate), np.linspace(min_values_m[next_m], max_values_m[next_m], samplerate)])
                        pdf = kdes_m[next_m].evaluate(samples)
                        iteration += 1

                        if iteration > 1000:
                            print('Too many iterations')
                            break

            a_m[component,hour] = semi_monthly_amplitude

            if hour%10000 == 0:
                print str(hour/1000)+'k hours generated'

        for i in np.arange(0, T, tau):
            epsilon[component,i] = a_m[component,i] + a_d[component,i] + a_h[component,i] # The +1 is to skip the startvalue of epsilon_h

        lambd, princ_comp = PCA.get_principal_component(h, component)
        mismatch_PC = PCA.unnormalize_uncenter(princ_comp,Ntilde, mean_mismatch)

        approx_mismatch += np.outer(mismatch_PC, epsilon[component,:])
        print 'One Component done'

    filename = 'weights_generated.npy'
    np.save('approx_mismatch_generated'+'.npy',approx_mismatch)
    np.save(filename, epsilon)


def solveNetworkWithGeneratedSeries(filename='approx_mismatch_generated'+'.npy',mode='copper linear',savefilename='Solved_Network'):
    N = Grid.EU_Nodes_Gamma_Alpha_input(None, g=1.0, a=0.8, full_load=True)
    new_mismatch = np.load(filename)

    i = 0
    for i, n in enumerate(N, start=0):

        n.mismatch = new_mismatch[i,:]


    M, F = au.solve(N, mode=mode)

    M.save_nodes(savefilename+'.npz')
    np.save(savefilename+'_Flow.npy', F)



def createCumulativeSumOfDaysInData(StartMonth='Jan'):

    IndexStartMonth = getIndexOfStartMonth(StartMonth=StartMonth)
    # This is purely for the ISET DATA!

    # Data strucking of days in a month during a year
    DaysInAMonth = np.asarray([15, 31-15, 15, 28-15,15, 31-15,15, 30-15,15, 31-15,15, 30-15,15, 31-15,15, 31-15,15, 30-15,15, 31-15,15, 30-15,15, 31-15])
    DaysInAMonthLeapYear = np.asarray([15, 31-15, 15, 29-15,15, 31-15, 15, 30-15,15, 31-15,15, 30-15,15, 31-15,15, 31-15,15, 30-15,15, 31-15,15, 30-15,15, 31-15])

    DaysInAMonth = np.roll(DaysInAMonth,IndexStartMonth*-2)
    DaysInAMonthLeapYear= np.roll(DaysInAMonthLeapYear,IndexStartMonth*-2)

    # Builing a list for use with cumsum, adding up to the total number of days in the data.
    CumSumOfMonths = np.append(DaysInAMonthLeapYear, DaysInAMonth)
    CumSumOfMonths = np.append(CumSumOfMonths, DaysInAMonth)
    CumSumOfMonths = np.append(CumSumOfMonths, DaysInAMonth)
    CumSumOfMonths = np.append(CumSumOfMonths, DaysInAMonth)
    CumSumOfMonths = np.append(CumSumOfMonths, DaysInAMonthLeapYear)
    CumSumOfMonths = np.append(CumSumOfMonths, DaysInAMonth)
    CumSumOfMonths = np.append(CumSumOfMonths, DaysInAMonth)

    # Creating the cumsum list.
    CumSumOfMonthsInDays = np.cumsum(CumSumOfMonths)
    CumSumOfMonthsInHours = np.cumsum(CumSumOfMonths)*24

    return CumSumOfMonthsInDays,CumSumOfMonthsInHours


def createStartValueArrayFromDetrendData(weights_monthly_all,weights_daily_all,weights_hourly_all,year=0,StartMonth='Jan'):
    """
    An ease of use method to be used with the method generate_new_data
    :param weights_monthly_all:
    :param weights_daily_all:
    :param weights_hourly_all:
    :return Start: An array of start values each rows: semi-monthly,daily,hourly column: component
    """
    if StartMonth == 'Jan' and year == 0:
        month = 0
        day = 0
        hour = 0
    else:
        month = (getIndexOfStartMonth(StartMonth=StartMonth)*2)-1
        days_in_month = np.insert(createCumulativeSumOfDaysInData()[0][1::2],0,0)
        day = days_in_month[month+(year*24)]-1
        hour = (day*24)-1

    # Stack the array as 3x30
    Start = weights_monthly_all[:,month+(year*24)]
    Start = np.vstack((Start,weights_daily_all[:,day]))
    Start = np.vstack((Start,weights_hourly_all[:,hour]))

    return Start


def createDictioariesOfMonthsInHours(Days,CumSumOfMonths):
    # THIS IS A VERY UGLY METHOD, BUT IT WORKS
    # Creates a dicstionaries of all calaender months days in their hours
    # BASED ON CUMSUM FOR THE ISET DATA

    # First we count the total number of days in a month from the data.
    Jan = 0
    Feb = 0
    Mar = 0
    Apr = 0
    May = 0
    Jun = 0
    Jul = 0
    Aug = 0
    Sep = 0
    Okt = 0
    Nov = 0
    Dec = 0

    daybefore = 0
    month = 0
    for i, day in enumerate(CumSumOfMonths[1::2]/24, start=0):

        if i == 0:
            span = np.arange(0, day)
        else:
            span = np.arange(daybefore, day)

        if i % 12 == 0:
            month = 0

        if month == 0:
            Jan += len(span)
        if month == 1:
            Feb += len(span)
        if month == 2:
            Mar += len(span)
        if month == 3:
            Apr += len(span)
        if month == 4:
            May += len(span)
        if month == 5:
            Jun += len(span)
        if month == 6:
            Jul += len(span)
        if month == 7:
            Aug += len(span)
        if month == 8:
            Sep += len(span)
        if month == 9:
            Okt += len(span)
        if month == 10:
            Nov += len(span)
        if month == 11:
            Dec += len(span)

        month += 1
        daybefore = day

    Jan_a = np.zeros((Jan,24))
    Feb_a = np.zeros((Feb,24))
    Mar_a = np.zeros((Mar,24))
    Apr_a = np.zeros((Apr,24))
    May_a = np.zeros((May,24))
    Jun_a = np.zeros((Jun,24))
    Jul_a = np.zeros((Jul,24))
    Aug_a = np.zeros((Aug,24))
    Sep_a = np.zeros((Sep,24))
    Okt_a = np.zeros((Okt,24))
    Nov_a = np.zeros((Nov,24))
    Dec_a = np.zeros((Dec,24))

    bj = 0
    bf = 0
    bm = 0
    ba = 0
    bmay = 0
    bjun = 0
    bjul = 0
    baug = 0
    bs = 0
    bo = 0
    bn = 0
    bd = 0

    daybefore = 0
    month = 0

    for i, day in enumerate(CumSumOfMonths[1::2]/24, start=0):
        if i == 0:
            span = np.arange(0, day)
        else:
            span = np.arange(daybefore, day)

        if i % 12 == 0:
            month = 0

        if month == 0:
            Jan_a[np.arange(bj,bj+len(span)),:] = Days[span,:]
            bj += len(span)

        if month == 1:
            Feb_a[np.arange(bf,bf+len(span)),:] = Days[span,:]
            bf += len(span)

        if month == 2:
            Mar_a[np.arange(bm,bm+len(span)),:] = Days[span,:]
            bm += len(span)

        if month == 3:
            Apr_a[np.arange(ba,ba+len(span)),:] = Days[span,:]
            ba += len(span)

        if month == 4:
            May_a[np.arange(bmay,bmay+len(span)),:] = Days[span,:]
            bmay += len(span)

        if month == 5:
            Jun_a[np.arange(bjun,bjun+len(span)),:] = Days[span,:]
            bjun += len(span)

        if month == 6:

            Jul_a[np.arange(bjul,bjul+len(span)),:] = Days[span,:]
            bjul += len(span)

        if month == 7:

            Aug_a[np.arange(baug,baug+len(span)),:] = Days[span,:]
            baug += len(span)

        if month == 8:
            Sep_a[np.arange(bs,bs+len(span)),:] = Days[span,:]
            bs += len(span)

        if month == 9:

            Okt_a[np.arange(bo,bo+len(span)),:] = Days[span,:]
            bo += len(span)

        if month == 10:
            Nov_a[np.arange(bn,bn+len(span)),:] = Days[span,:]
            bn += len(span)

        if month == 11:
            Dec_a[np.arange(bd,bd+len(span)),:] = Days[span,:]
            bd += len(span)


        month += 1
        daybefore = day

    #################################################################################
    Month_dic = {'Jan':Jan_a,'Feb' : Feb_a,'Mar': Mar_a,'Apr':Apr_a,'May':May_a,'Jun':Jun_a,'Jul':Jul_a,'Aug':Aug_a,'Sep':Sep_a,'Okt':Okt_a,'Nov':Nov_a,'Dec':Dec_a}

    return Month_dic

def getIndexOfStartMonth(StartMonth='Jan'):

    MonthStrings = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Okt','Nov','Dec']
    IndexStartMonth = MonthStrings.index(StartMonth)

    return IndexStartMonth

