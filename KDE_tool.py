__author__ = 'Raunbak'
import numpy as np
from matplotlib.pylab import *
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from scipy import stats


def create_new_series(series, start_point=None, kernel_type='epanechnikov'):

    #
    grid = GridSearchCV(KernelDensity(kernel=kernel_type), {'bandwidth': np.logspace(-1, 1, 20)}) # 20-fold cross-validation
    # vstack(x,y) -> y,x liste

    # EXAMPLE TEXT USING MISMATCH
    # Stacking the mismatches, as such:
    # | mismatch[0], mismatch[1},..... |
    # | mismatch[1], mismatch[2],..... |
    # Therefore we delete the last element.
    # This is done because we want to look at the transition from one value to the next one.
    stack = np.vstack((series[:-1], np.roll(series, -1)[:-1])).T

    # Fitting the GridSearch to the stack. So we can finde the best bandwidth for the KDE.
    grid.fit(stack)
    # Printing the bandwidth number.
    print grid.best_params_
    # use the best estimator to compute the kernel density estimate
    kde = grid.best_estimator_

    # Create array to store data
    new_mismatch_series = np.zeros_like(series)

    # Choosing start point for series.
    current_mismatch = series[0]
    if start_point is not None:
        current_mismatch = start_point


    # Finding max and min values to draw from.
    max_value = np.max(series) + np.max(series)/10  # Max value plus 10%. THIS IS JUST AN ESTIMATE
    min_value = np.min(series) + np.min(series)/10  # Min value plus 10%. JUST AN ESTIMATE
    # Draw a new data point from the KDE.
    for i in range(len(series)):

        # Store the current mismatch to the series.
        new_mismatch_series[i] = current_mismatch

        # Sample from the KDE
        # Limits are from max and min values. Right now hardcoded we find the propability
        # of a 100 different transition of the current mismatch, from min value to max value.
        samples2 = np.vstack((np.repeat(current_mismatch, 100), np.linspace(min_value, max_value, 100))).T
        pdf2 = np.exp(kde.score_samples(samples2))

        # Normalize and find the cumsum
        pdf2 = pdf2/np.sum(pdf2)
        cdf = np.cumsum(pdf2)

        # Choose random number to select the next value, from the cumsum
        randomNumber = np.random.rand(1)
        index = np.digitize(randomNumber, cdf)

        # Finding the next mismatch.
        current_mismatch = np.linspace(min_value, max_value, 100)[index]

    return new_mismatch_series


def create_contour_of_kde(series, kernel_type='epanechnikov', filename='contour_KDE'):

    grid = GridSearchCV(KernelDensity(kernel=kernel_type), {'bandwidth': np.logspace(-10, 10, 1000)}) # 20-fold cross-validation
    series_roll = np.roll(series,-1)
    stack = np.vstack((series[:-1], series_roll[:-1])).T

    grid.fit(stack)
    print grid.best_params_

    # use the best estimator to compute the kernel density estimate
    kde = grid.best_estimator_
    # Finding max and min values to draw from.
    max_value = np.max(series) + np.max(series)/10  # Max value plus 10%. THIS IS JUST AN ESTIMATE
    min_value = np.min(series) + np.min(series)/10  # Min value plus 10%. JUST AN ESTIMATE

    # sample points from the data
    samples = np.vstack((np.repeat(np.linspace(min_value, max_value, 100), 100), np.tile(np.linspace(min_value, max_value, 100), 100))).T

    pdf = np.exp(kde.score_samples(samples))
    pdf = np.reshape(pdf, (100,100))

    fig, ax = plt.subplots()
    ax.plot(series, np.roll(series, -1), 'k.',alpha=0.5)
    ax.contour(np.linspace(min_value, max_value, 100), np.linspace(min_value, max_value, 100), pdf,10)
    ax.grid('on')
    ax.set_ylabel(r'$a(t+\tau)$')
    ax.set_xlabel(r'$a(t)$')

    fig.savefig('results/figures/' + filename + '.pdf')
    print 'results/figures/' + filename + '.pdf'+'  SAVED'


def create_kde(series, kernel_type='epanechnikov'):

    #
    grid = GridSearchCV(KernelDensity(kernel=kernel_type), {'bandwidth': np.logspace(-10, 10, 1000)}) # 20-fold cross-validation
    # vstack(x,y) -> y,x liste

    # EXAMPLE TEXT USING MISMATCH
    # Stacking the mismatches, as such:
    # | mismatch[0], mismatch[1},..... |
    # | mismatch[1], mismatch[2],..... |
    # Therefore we delete the last element.
    # This is done because we want to look at the transition from one value to the next one.
    stack = np.vstack((series[:-1], np.roll(series, -1)[:-1])).T

    # Fitting the GridSearch to the stack. So we can finde the best bandwidth for the KDE.
    grid.fit(stack)
    # Printing the bandwidth number.
    # use the best estimator to compute the kernel density estimate
    kde = grid.best_estimator_

    return kde, grid.best_params_


def kramer_moyal_coeff(current_value,transition_values, pdf, n, tau):

    # From eq. 3.85 in The Fokker planch eq.
    assert (len(transition_values) == len(pdf))
    # Make sure that the transitions probability are normalized
    norm_pdf = pdf/np.sum(pdf)
    # Difference between x and X(t+tau)

    difference = (transition_values - current_value)**n

    return 1.0/tau * 1.0/np.math.factorial(n) * np.sum((difference*norm_pdf)) #1.0/tau * np.trapz(difference/np.math.factorial(n)*norm_pdf) ## Older way

    #if n == 1:
    #    return np.sum(np.multiply(np.power(difference, n), norm_pdf)) #np.multiply(1/tau,np.sum(np.multiply(np.power(difference, n), norm_pdf)))
    #if n == 2:
    #    return np.multiply(0.5 * 1/tau, np.sum(np.multiply(np.power(difference, n), norm_pdf)))
    #if n == 3:
    #    return np.multiply(0.16666666666 * 1/tau, np.sum(np.multiply(np.power(difference, n), norm_pdf)))
    #if n == 4:
    #    return np.multiply(0.04166666666 * 1/tau, np.sum(np.multiply(np.power(difference, n), norm_pdf)))
