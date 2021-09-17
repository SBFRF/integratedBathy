"""
Created on 5/3/2017
this is a plotting library for CSHORE inputs data and results...
@author: David Young, but if it doesn't work, Spicer Bak

"""
import datetime as DT
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.image as image
import os
from pip.compat import total_seconds
from getdatatestbed.getDataFRF import getObs
from sblib.sblib import timeMatch, statsBryant
from sblib.anglesLib import vectorRotation

# THIS ENTIRE .PY FILE IS DEPRECATED.
# ALL THESE FUNCTIONS HAVE BEEN MOVED TO nonoperationalPlots.py OR operationalPlots.py!!!!!!!!

def obs_V_mod_TS(ofname, p_dict, logo_path='../ArchiveFolder/CHL_logo.png'):
    """
    This script basically just compares two time series, under 
        the assmption that one is from the model and one a set of observations

    :param  file_path: this is the full file-path (string) to the location where the plot will be saved
    :param p_dict: has 6 keys to it.
        (1) a vector of datetimes ('time')
        (2) vector of observations ('obs')
        (3) vector of model data ('model')
        (4) variable name (string) ('var_name')
        (5) variable units (string!!) ('units') -> this will be put inside a tex math environment!!!!
        (6) plot title (string) ('p_title')
    :return: a model vs. observation time-series plot'
        the dictionary of the statistics calculated

    """
    # this function plots observed data vs. model data for time-series data and computes stats for it.

    assert len(p_dict['time']) == len(p_dict['obs']) == len(p_dict['model']), "Your time, model, and observation arrays are not all the same length!"
    assert sum([isinstance(p_dict['time'][ss], DT.datetime) for ss in range(0, len(p_dict['time']))]) == len(p_dict['time']), 'Your times input must be an array of datetimes!'
    # calculate total duration of data to pick ticks for Xaxis on time series plot
    totalDuration = p_dict['time'][-1] - p_dict['time'][0]
    if totalDuration.days > 365:  # this is a year +  of data
        # mark 7 day increments with monthly major lables
        majorTickLocator = mdates.MonthLocator(interval=3) # every 3 months
        minorTickLocator = mdates.AutoDateLocator() # DayLocator(7)
        xfmt = mdates.DateFormatter('%Y-%m')
    elif totalDuration.days > 30: # thie is months of data that is not a year
        # mark 12 hour with daily major labels
        majorTickLocator = mdates.DayLocator(1)
        minorTickLocator = mdates.HourLocator(12)
        xfmt = mdates.DateFormatter('%Y-%m-%d')
    elif totalDuration.days > 5:
        # mark 6 hours with daily major labels
        majorTickLocator = mdates.DayLocator(1)
        minorTickLocator = mdates.HourLocator(6)
        xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
    else:
        # mark hourly with 6 hour labels major intervals
        tickInterval = 6  # hours?
        majorTickLocator = mdates.HourLocator(interval=tickInterval)
        minorTickLocator = mdates.HourLocator(1)
        xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M')

    ####################################################################################################################
    # Begin Plot
    ####################################################################################################################
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(p_dict['p_title'], fontsize=18, fontweight='bold', verticalalignment='top')

    # time series
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    min_val = np.min([np.min(p_dict['obs']), np.nanmin(p_dict['model'])])
    max_val = np.max([np.max(p_dict['obs']), np.nanmax(p_dict['model'])])
    if min_val < 0 and max_val > 0:
        base_date = min(p_dict['time']) - DT.timedelta(seconds=0.5 * (p_dict['time'][1] - p_dict['time'][0]).total_seconds())
        base_times = np.array([base_date + DT.timedelta(seconds=n * (p_dict['time'][1] - p_dict['time'][0]).total_seconds()) for n in range(0, len(p_dict['time']) + 1)])
        ax1.plot(base_times, np.zeros(len(base_times)), 'k--')
    ax1.plot(p_dict['time'], p_dict['obs'], 'r.', label='Observed')
    ax1.plot(p_dict['time'], p_dict['model'], 'b.', label='Model')
    ax1.set_ylabel('%s [$%s$]' % (p_dict['var_name'], p_dict['units']), fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax1.set_ylim([sf1 * min_val, sf2 * max_val])
    ax1.set_xlim([min(p_dict['time']) - DT.timedelta(seconds=0.5 * (p_dict['time'][1] - p_dict['time'][0]).total_seconds()),
                  max(p_dict['time']) + DT.timedelta(seconds=0.5 * (p_dict['time'][1] - p_dict['time'][0]).total_seconds())])

    # this is what you change for time-series x-axis ticks!!!!!

    ax1.xaxis.set_major_locator(majorTickLocator)
    ax1.xaxis.set_minor_locator(minorTickLocator)
    ax1.xaxis.set_major_formatter(xfmt)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    ax1.tick_params(labelsize=14)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, borderaxespad=0., fontsize=14)

    # Now working on the 1-1 comparison subplot
    one_one = np.linspace(min_val - 0.05 * (max_val - min_val), max_val + 0.05 * (max_val - min_val), 100)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax2.plot(one_one, one_one, 'k-', label='unity-line')
    if min_val < 0 and max_val > 0:
        ax2.plot(one_one, np.zeros(len(one_one)), 'k--')
        ax2.plot(np.zeros(len(one_one)), one_one, 'k--')
    ax2.plot(p_dict['obs'], p_dict['model'], 'r*')
    ax2.set_xlabel('Observed %s [$%s$]' % (p_dict['var_name'], p_dict['units']), fontsize=16)
    ax2.set_ylabel('Model %s [$%s$]' % (p_dict['var_name'], p_dict['units']), fontsize=16)
    ax2.set_xlim([min_val - 0.025 * (max_val - min_val), max_val + 0.025 * (max_val - min_val)])
    ax2.set_ylim([min_val - 0.025 * (max_val - min_val), max_val + 0.025 * (max_val - min_val)])
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax2.tick_params(labelsize=14)
    plt.legend(loc=0, ncol=1, borderaxespad=0.5, fontsize=14)

    # stats and stats text
    stats_dict = {}
    stats_dict = statsBryant(p_dict['obs'], p_dict['model'])
    stats_dict['m_mean'] = np.nanmean(p_dict['model'])
    stats_dict['o_mean'] = np.mean(p_dict['obs'])
    # the below are calculated in statsBryant... this keeps all statistics calcs in the same place
    # stats_dict['bias'] = np.mean(p_dict['obs'] - p_dict['model'])
    # stats_dict['RMSE'] = np.sqrt((1 / (float(len(p_dict['obs'])) - 1)) * np.sum(np.power(p_dict['obs'] - p_dict['model'] - stats_dict['bias'], 2)))
    # stats_dict['SI'] = stats_dict['RMSE'] / float(stats_dict['m_mean'])
    # stats_dict['sym_slp'] = np.sqrt(np.sum(np.power(p_dict['obs'], 2)) / float(np.sum(np.power(p_dict['model'], 2))))
    # dum = np.zeros([2, len(p_dict['model'])])
    # dum[0] = p_dict['model'].flatten()
    # dum[1] = p_dict['obs'].flatten()
    # stats_dict['corr_coef'] = np.corrcoef(dum)[0, 1]

    header_str = '%s Comparison \nModel to Observations:' % (p_dict['var_name'])
    m_mean_str = '\n Model Mean $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['m_mean']), p_dict['units'])
    o_mean_str = '\n Observation Mean $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['o_mean']), p_dict['units'])
    bias_str = '\n Bias $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['bias']), p_dict['units'])
    RMSE_str = '\n RMSE $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['RMSE']), p_dict['units'])
    SI_str = '\n Similarity Index $=%s$' % ("{0:.2f}".format(stats_dict['scatterIndex']))
    sym_slp_str = '\n Symmetric Slope $=%s$' % ("{0:.2f}".format(stats_dict['symSlope']))
    corr_coef_str = '\n Correlation Coefficient $=%s$' % ("{0:.2f}".format(stats_dict['corr']))
    RMSE_Norm_str = u'\n %%RMSE $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['RMSEnorm']), p_dict['units'])
    num_String = '\n Number of samples $= %s$' %len(stats_dict['residuals'])
    plot_str = m_mean_str + o_mean_str + bias_str + RMSE_str + SI_str + sym_slp_str + corr_coef_str + RMSE_Norm_str + num_String
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax3.axis('off')
    ax4 = ax3.twinx()
    ax3.axis('off')

    try:
        ax4.axis('off')
        dir_name = os.path.dirname(__file__).split('\\plotting')[0]
        CHL_logo = image.imread(os.path.join(dir_name, logo_path))
        ax4 = fig.add_axes([0.78, 0.02, 0.20, 0.20], anchor='SE', zorder=-1)
        ax4.imshow(CHL_logo)
        ax4.axis('off')
    except:
        print 'Plot generated sans CHL Logo!'

    ax3.axis('off')
    ax3.text(0.01, 0.99, header_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=18,
             fontweight='bold')
    ax3.text(0.01, 0.90, plot_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=16)

    fig.subplots_adjust(wspace=0.4, hspace=0.1)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
    fig.savefig(ofname, dpi=300)
    plt.close()
    return stats_dict

def bc_plot(ofname, p_dict):
    """
    This is the script to plot some information about the boundary conditions that were put into the CSHORE infile..
    :param file_path: this is the full file-path (string) to the location where the plot will be saved i.e., C://users... 
        DONT include the final '/' or the actual NAME of the plot!!!!!! 
    :param p_dict:
        (1) a vector of datetimes ('time')
        (2) vector of bathymetry x-positions ('x')
        (3) vector of bathymetry bed elevations ('zb')
        (4) datetime that the bathy survey was DONE ('init_bathy_stime')
        (5) vector of water levels ('time-series') at the offshore boundary ('WL')
        (6) vector of significant wave heights (time-series) at the offshore boundary ('Hs')
        (7) vector of wave angles (time-series) at the offshore boundary ('angle')
        (8) vector of wave periods (time-series) at the offshore boundary ('Tp')
        (9) plot title ('string') ('p_title')
    :return: a plot of the boundary conditions for the simulation 
    """

    # get rid of this and include them in the handed dictionary if you want to include vegetation in the plots
    p_dict['veg_ind'] = []
    p_dict['non_veg_ind'] = []

    # assert some stuff to throw errors if need be!
    assert len(p_dict['time']) == len(p_dict['Hs']) == len(p_dict['WL']) == len(p_dict['angle']), "Your time, Hs, wave angle, and WL arrays are not all the same length!"
    assert len(p_dict['x']) == len(p_dict['zb']), "Your x and zb arrays are not the same length!"
    assert sum([isinstance(p_dict['time'][ss], DT.datetime) for ss in range(0, len(p_dict['time']))]) == len(p_dict['time']), 'Your times input must be an array of datetimes!'

    xfmt = mdates.DateFormatter('%m/%d/%y %H:%M')
    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(p_dict['p_title'], fontsize=14, fontweight='bold', verticalalignment='top')

    # Plotting Hs and WL
    ax1 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    a, = ax1.plot(p_dict['time'], p_dict['Hs'], 'r-', label='$H_{s}$')
    ax1.set_ylabel('$H_s$ [$m$]')
    ax1.tick_params('y', colors='r')
    ax1.set_xlim([np.min(p_dict['time']), np.max(p_dict['time'])])
    ax1.set_ylim([0.9 * np.min(p_dict['Hs']), 1.1 * np.max(p_dict['Hs'])])
    ax1.yaxis.label.set_color('red')
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax1.xaxis.set_major_formatter(xfmt)

    # determine axis scale factor
    if np.min(p_dict['WL']) >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if np.max(p_dict['WL']) >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9

    ax2 = ax1.twinx()
    ax2.plot(p_dict['time'], np.zeros(len(p_dict['WL'])), 'b--')
    b, = ax2.plot(p_dict['time'], p_dict['WL'], 'b-', label='WL')
    ax2.set_ylabel('$WL$ [$m$]')
    ax2.tick_params('y', colors='b')
    ax2.set_ylim([sf1 * np.min(p_dict['WL']), sf2 * np.max(p_dict['WL'])])
    ax2.set_xlim([np.min(p_dict['time']), np.max(p_dict['time'])])
    ax2.yaxis.label.set_color('blue')
    p = [a, b]
    ax1.legend(p, [p_.get_label() for p_ in p], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0., fontsize=14)

    # determine axis scale factor
    if np.min(p_dict['zb']) >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if np.max(p_dict['zb']) >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9

    if len(p_dict['veg_ind']) > 0:
        ax3 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
        ax3.plot(p_dict['x'][p_dict['veg_ind']], p_dict['zb'][p_dict['veg_ind']], 'g-', label='Vegetated')
        ax3.plot(p_dict['x'][p_dict['non_veg_ind']], p_dict['zb'][p_dict['non_veg_ind']], 'y-', label='Non-vegetated')
        ax3.plot(p_dict['x'], np.mean(p_dict['WL']) * np.ones(len(p_dict['x'])), 'b-', label='Mean WL')
        ax3.set_ylabel('$Elevation$ [$m$]')
        ax3.set_xlabel('x [$m$]')
        ax3.set_xlim([np.min(p_dict['x']), np.max(p_dict['x'])])
        ax3.set_ylim([sf1 * np.min(p_dict['zb']), sf2 * np.max(p_dict['zb'])])
        Bathy_date = p_dict['init_bathy_stime'].strftime('%Y-%m-%dT%H:%M:%SZ')
        ax3.text(0.05, 0.85, 'Bathymetry Survey Time:\n' + Bathy_date, transform=ax3.transAxes, color='black', fontsize=12)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, borderaxespad=0., fontsize=14)


    else:
        ax3 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
        ax3.plot(p_dict['x'], p_dict['zb'], 'y-', label='Non-vegetated')
        ax3.plot(p_dict['x'], np.mean(p_dict['WL']) * np.ones(len(p_dict['x'])), 'b-', label='Mean WL')
        ax3.set_ylabel('$Elevation$ [$m$]')
        ax3.set_xlabel('x [$m$]')
        ax3.set_xlim([np.min(p_dict['x']), np.max(p_dict['x'])])
        ax3.set_ylim([sf1 * np.min(p_dict['zb']), sf2 * np.max(p_dict['zb'])])
        Bathy_date = p_dict['init_bathy_stime'].strftime('%Y-%m-%dT%H:%M:%SZ')
        ax3.text(0.05, 0.85, 'Bathymetry Survey Time:\n' + Bathy_date, transform=ax3.transAxes, color='black', fontsize=12)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0., fontsize=14)

    # plotting Tp and angle
    ax4 = plt.subplot2grid((2, 3), (0, 0), colspan=1)
    a, = ax4.plot(p_dict['time'], p_dict['Tp'], 'b-', label='$T_{p}$')
    ax4.set_ylabel('$T_{p}$ [$m$]')
    ax4.tick_params('y', colors='b')
    ax4.set_xlim([np.min(p_dict['time']), np.max(p_dict['time'])])
    ax4.set_ylim([0.9 * np.min(p_dict['Tp']), 1.1 * np.max(p_dict['Tp'])])
    ax4.yaxis.label.set_color('blue')
    ax4.xaxis.set_major_formatter(xfmt)
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=24))

    ax5 = ax4.twinx()
    b, = ax5.plot(p_dict['time'], p_dict['angle'], 'r-', label='Wave Angle')
    ax5.plot(p_dict['time'], np.zeros(len(p_dict['angle'])), 'r--')
    ax5.set_ylabel('$decimal$ $^{0}$')
    ax5.tick_params('y', colors='r')
    ax5.set_ylim([-180, 180])
    ax5.set_xlim([np.min(p_dict['time']), np.max(p_dict['time'])])
    ax5.yaxis.label.set_color('red')
    p = [a, b]
    ax4.legend(p, [p_.get_label() for p_ in p], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0., fontsize=14)

    fig.subplots_adjust(wspace=0.4, hspace=0.1)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[.05, 0.05, 0.95, 0.925])
    fig.savefig(ofname, dpi=300)
    plt.close()

def obs_V_mod_bathy(ofname, p_dict, obs_dict, logo_path='ArchiveFolder/CHL_logo.png', contour_s=3, contour_d=8):
    """
    This is a plot to compare observed and model bathymetry to each other 
    :param file_path: this is the full file-path (string) to the location where the plot will be saved i.e., C://users... 
        DONT include the final '/' or the actual NAME of the plot!!!!!! 
    :param p_dict:
        (1) a vector of x-positions for the bathymetry ('x')
            MAKE SURE THIS IS IN FRF COORDS!!!!
        (2) vector of OBSERVED bathymetry bed elevations ('obs')
        (3) datetime of the OBSERVED bathymetry survey ('obs_time')
        (4) vector of MODEL bathymetry bed elevations ('model')
        (5) datetime of the MODEL bathymetry ('model_time')
        (6) vector of model Hs at EACH model NODE at the TIME of the MODEL BATHYMETRY ('Hs')
        (7) vector of the standard deviation of model Hs at EACH model NODE ('sigma_Hs')
        (8) time series of water level at the offshore boundary ('WL')
        (12) array of datetimes for the water level data ('time').
            AS A HEADS UP, THIS IS THE RANGE OF TIMES THAT WILL GO INTO getObs for the comparisons!!!
        (9) variable name ('var_name')
        (10) variable units ('units') (string) -> this will be put inside a tex math environment!!!!
        (11) plot title (string) ('p_title')
        
    :param logo_path: this is the path to get the CHL logo to display it on the plot!!!!
    :param contour_s: this is the INSIDE THE SANDBAR contour line (shallow contour line)
        we are going out to for the volume calculations (depth in m!!)
    :param contour_d: this is the OUTSIDE THE SANDBAR contour line (deep contour line)
        we are going out to for the volume calculations (depth in m!!)
    :return: 
        model to observation comparison for spatial data - right now all it does is bathymetry?
        may need modifications to be more general, but I'm not sure what other
        kind of data would need to be plotted in a similar manner? 
    """

    # Altimeter data!!!!!!!!
    Alt05 = obs_dict['Alt05']
    Alt04 = obs_dict['Alt04']
    Alt03 = obs_dict['Alt03']

    # wave data
    Adopp_35 = obs_dict['Adopp_35']
    AWAC6m = obs_dict['AWAC6m']
    AWAC8m = obs_dict['AWAC8m']

    assert len(p_dict['sigma_Hs']) == len(p_dict['Hs']) == len(p_dict['x']) == len(p_dict['obs']) == len(p_dict['model']), "Your x, Hs, model, and observation arrays are not the same length!"

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(p_dict['p_title'], fontsize=18, fontweight='bold', verticalalignment='top')

    # transects
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    min_val = np.min([np.min(p_dict['obs']), np.min(p_dict['model'])])
    max_val = np.max([np.max(p_dict['obs']), np.max(p_dict['model'])])
    min_x = np.min(p_dict['x'])
    max_x = np.max(p_dict['x'])
    dum_x = np.linspace(min_x - 0.05 * (max_x - min_x), max_x + 0.05 * (max_x - min_x), 100)
    a, = ax1.plot(dum_x, np.mean(p_dict['WL']) * np.ones(len(dum_x)), 'b-', label='Mean WL')
    # get the time strings!!
    obs_date = p_dict['obs_time'].strftime('%Y-%m-%d %H:%M')
    model_date = p_dict['model_time'].strftime('%Y-%m-%d %H:%M')
    b, = ax1.plot(p_dict['x'], p_dict['obs'], 'r-', label='Observed \n' + obs_date)
    c, = ax1.plot(p_dict['x'], p_dict['model'], 'y-', label='Model \n' + model_date)

    # add altimeter data!!
    temp05 = Alt05['zb'][Alt05['plot_ind'] == 1]
    temp04 = Alt04['zb'][Alt04['plot_ind'] == 1]
    temp03 = Alt03['zb'][Alt03['plot_ind'] == 1]
    # Alt05
    f, = ax1.plot(Alt05['xFRF']*np.ones(2), [temp05 - np.std(Alt05['zb']), temp05 + np.std(Alt05['zb'])], 'k-', label='Gage Data')
    g, = ax1.plot(Alt05['xFRF'] * np.ones(1), temp05, 'k_', label='Gage Data')
    # Alt04
    h, = ax1.plot(Alt04['xFRF']*np.ones(2), [temp04 - np.std(Alt04['zb']), temp04 + np.std(Alt04['zb'])], 'k-', label='Gage Data')
    i, = ax1.plot(Alt04['xFRF'] * np.ones(1), temp04, 'k_', label='Gage Data')
    # Alt03
    j, = ax1.plot(Alt03['xFRF']*np.ones(2), [temp03 - np.std(Alt03['zb']), temp03 + np.std(Alt03['zb'])], 'k-', label='Gage Data')
    k, = ax1.plot(Alt03['xFRF'] * np.ones(1), temp03, 'k_', label='Gage Data')

    ax5 = ax1.twinx()
    d, = ax5.plot(p_dict['x'], p_dict['Hs'], 'g-', label='Model $H_{s}$')
    e, = ax5.plot(p_dict['x'], p_dict['Hs'] + p_dict['sigma_Hs'], 'g--', label='$H_{s} \pm \sigma_{H_{s}}$')
    ax5.plot(p_dict['x'], p_dict['Hs'] - p_dict['sigma_Hs'], 'g--')

    # add wave data!!
    temp35 = Adopp_35['Hs'][Adopp_35['plot_ind'] == 1]
    temp6m = AWAC6m['Hs'][AWAC6m['plot_ind'] == 1]
    temp8m = AWAC8m['Hs'][AWAC8m['plot_ind'] == 1]
    # Adopp_35
    l, = ax5.plot(Adopp_35['xFRF']*np.ones(2), [temp35 - np.std(Adopp_35['Hs']), temp35 + np.std(Adopp_35['Hs'])], 'k-', label='Gage Data')
    m, = ax5.plot(Adopp_35['xFRF']*np.ones(1), temp35, 'k_', label='Gage Data')
    # AWAC6m
    n, = ax5.plot(AWAC6m['xFRF']*np.ones(2), [temp6m - np.std(AWAC6m['Hs']), temp6m + np.std(AWAC6m['Hs'])], 'k-', label='Gage Data')
    o, = ax5.plot(AWAC6m['xFRF']*np.ones(1), temp6m, 'k_', label='Gage Data')
    # AWAC8m
    p, = ax5.plot(AWAC8m['xFRF']*np.ones(2), [temp8m - np.std(AWAC8m['Hs']), temp8m + np.std(AWAC8m['Hs'])], 'k-', label='Gage Data')
    q, = ax5.plot(AWAC8m['xFRF']*np.ones(1), temp8m, 'k_', label='Gage Data')

    ax1.set_ylabel('Elevation (NAVD88) [$%s$]' % (p_dict['units']), fontsize=16)
    ax1.set_xlabel('Cross-shore Position [$%s$]' % (p_dict['units']), fontsize=16)
    ax5.set_ylabel('$H_{s}$ [$%s$]' % (p_dict['units']), fontsize=16)
    ax5.set_xlabel('Cross-shore Position [$%s$]' % (p_dict['units']), fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax1.set_ylim([sf1 * min_val, sf2 * max_val])
    ax1.set_xlim([min(dum_x), max(dum_x)])
    ax1.tick_params('y', colors='r')
    ax1.yaxis.label.set_color('red')

    ax5.tick_params('y', colors='g')
    ax5.set_ylim([-1.05 * max(p_dict['Hs'] + p_dict['sigma_Hs']), 1.05 * max(p_dict['Hs'] + p_dict['sigma_Hs'])])
    ax5.set_xlim([min(dum_x), max(dum_x)])
    ax5.yaxis.label.set_color('green')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax5.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax5.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax1.tick_params(labelsize=14)
    ax5.tick_params(labelsize=14)
    p = [a, b, c, f, d, e]
    ax1.legend(p, [p_.get_label() for p_ in p], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(p), borderaxespad=0., fontsize=12, handletextpad=0.05)

    # 1 to 1
    one_one = np.linspace(min_val - 0.05 * (max_val - min_val), max_val + 0.05 * (max_val - min_val), 100)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax2.plot(one_one, one_one, 'k-', label='$45^{0}$-line')
    if min_val < 0 and max_val > 0:
        ax2.plot(one_one, np.zeros(len(one_one)), 'k--')
        ax2.plot(np.zeros(len(one_one)), one_one, 'k--')
    ax2.plot(p_dict['obs'], p_dict['model'], 'r*')
    ax2.set_xlabel('Observed %s [$%s$]' % (p_dict['var_name'], p_dict['units']), fontsize=16)
    ax2.set_ylabel('Model %s [$%s$]' % (p_dict['var_name'], p_dict['units']), fontsize=16)
    ax2.set_xlim([min_val - 0.025 * (max_val - min_val), max_val + 0.025 * (max_val - min_val)])
    ax2.set_ylim([min_val - 0.025 * (max_val - min_val), max_val + 0.025 * (max_val - min_val)])
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax2.tick_params(labelsize=14)
    plt.legend(loc=0, ncol=1, borderaxespad=0.5, fontsize=14)

    # stats and stats text
    stats_dict = {}
    stats_dict['bias'] = np.mean(p_dict['obs'] - p_dict['model'])
    stats_dict['RMSE'] = np.sqrt((1 / (float(len(p_dict['obs'])) - 1)) * np.sum(np.power(p_dict['obs'] - p_dict['model'] - stats_dict['bias'], 2)))
    stats_dict['sym_slp'] = np.sqrt(np.sum(np.power(p_dict['obs'], 2)) / float(np.sum(np.power(p_dict['model'], 2))))
    # correlation coef
    dum = np.zeros([2, len(p_dict['model'])])
    dum[0] = p_dict['model'].flatten()
    dum[1] = p_dict['obs'].flatten()
    stats_dict['corr_coef'] = np.corrcoef(dum)[0, 1]

    # volume change
    # shallow
    index_XXm = np.min(np.argwhere(p_dict['obs'] >= -1 * contour_s).flatten())  # ok, the indices currently count from offshore to onshore, so we want the SMALLEST index!
    vol_obs = np.trapz(p_dict['obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][1] - p_dict['x'][0])
    vol_model = np.trapz(p_dict['model'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][1] - p_dict['x'][0])
    stats_dict['vol_change_%sm' % (contour_s)] = vol_model - vol_obs
    #deep
    index_XXm = np.min(np.argwhere(p_dict['obs'] >= -1 * contour_d).flatten())  # ok, the indices currently count from offshore to onshore, so we want the SMALLEST index!
    vol_obs = np.trapz(p_dict['obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][1] - p_dict['x'][0])
    vol_model = np.trapz(p_dict['model'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][1] - p_dict['x'][0])
    stats_dict['vol_change_%sm' % (contour_d)] = vol_model - vol_obs

    header_str = '%s Comparison \nModel to Observations:' % (p_dict['var_name'])
    bias_str = '\n Bias $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['bias']), p_dict['units'])
    RMSE_str = '\n RMSE $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['RMSE']), p_dict['units'])
    sym_slp_str = '\n Symmetric Slope $=%s$' % ("{0:.2f}".format(stats_dict['sym_slp']))
    corr_coef_str = '\n Correlation Coefficient $=%s$' % ("{0:.2f}".format(stats_dict['corr_coef']))
    shall_vol_str = '\n $%s$ $%s$ Volume Change $=%s$ $(%s^{3}/%s)$' % (contour_s, p_dict['units'], "{0:.2f}".format(stats_dict['vol_change_%sm' % (contour_s)]), p_dict['units'], p_dict['units'])
    deep_vol_str = '\n $%s$ $%s$ Volume Change $=%s$ $(%s^{3}/%s)$' % (contour_d, p_dict['units'], "{0:.2f}".format(stats_dict['vol_change_%sm' % (contour_d)]), p_dict['units'], p_dict['units'])
    vol_expl_str = '*Note: volume change is defined as the \n $model$ volume minus the $observed$ volume'

    plot_str = bias_str + RMSE_str + sym_slp_str + corr_coef_str + shall_vol_str + deep_vol_str
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax3.axis('off')
    ax4 = ax3.twinx()
    ax3.axis('off')
    try:
        ax4.axis('off')
        dir_name = os.path.dirname(__file__).split('\\plotting')[0]
        CHL_logo = image.imread(os.path.join(dir_name, logo_path))
        ax4 = fig.add_axes([0.78, 0.02, 0.20, 0.20], anchor='SE', zorder=-1)
        ax4.imshow(CHL_logo)
        ax4.axis('off')
    except:
        print 'Plot generated sans CHL logo!'
    ax3.axis('off')
    ax3.text(0.01, 0.99, header_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=18, fontweight='bold')
    ax3.text(0.00, 0.90, plot_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=16)
    ax3.text(0.02, 0.43, vol_expl_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=14)

    fig.subplots_adjust(wspace=0.4, hspace=0.1)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(ofname, dpi=300)
    plt.close()

def mod_results(ofname, p_dict, obs_dict):

    """
    This script just lets you visualize the model outputs at a particular time-step
    :param file_path: this is the full file-path (string) to the location where the plot will be saved i.e., C://users... 
        DONT include the final '/' or the actual NAME of the plot!!!!!! 
    :param p_dict: 
        (1) a vector of x-positions for the bathymetry ('x')
            MAKE SURE THIS IS IN FRF COORDS!!!!
        (2) vector of MODEL bathymetry bed elevations ('zb_m')
        (3) vector of the standard deviation of the MODEL bathymetry bed elevations at each node! ('sigma_zbm')
        (4) datetime of the MODEL bathymetry ('model_time')
        (5) vector of model Hs at EACH model NODE at the TIME of the MODEL BATHYMETRY ('Hs')
        (6) vector of the standard deviation of model Hs at EACH model NODE ('sigma_Hs')
        (7) vector of the setup at EACH model NODE at the TIME of the MODEL BATHYMETRY ('setup_m')
            NOTE: the "setup" value output by the model is actually just the water surface elevation!!!!
            So if you want the actual "setup" you need to subtract off some reference water level!
            I used the water level at the offshore boundary at the same time-step,
                but we will need to check this once we resolve the model comparison issue with Brad!!
        (8) vector of the standard deviation of model setup at EACH model NODE ('sigma_setup')
            DONT have to subtract anything for standard deviation, it wont change....
        (9) plot title (string) ('p_title')
        (10) array of datetimes for the water level data ('time').
            AS A HEADS UP, THIS IS THE RANGE OF TIMES THAT WILL GO INTO getObs for the comparisons!!!
    :return: plot of a bunch of model results
    """

    # Altimeter data!!!!!!!!
    Alt05 = obs_dict['Alt05']
    Alt04 = obs_dict['Alt04']
    Alt03 = obs_dict['Alt03']

    # wave data
    Adopp_35 = obs_dict['Adopp_35']
    AWAC6m = obs_dict['AWAC6m']
    AWAC8m = obs_dict['AWAC8m']

    # get rid of this and include them in the handed dictionary if you want to include vegetation in the plots
    p_dict['veg_ind'] = []
    p_dict['non_veg_ind'] = []

    assert len(p_dict['zb_m']) == len(p_dict['sigma_zbm']) == len(p_dict['x']) == len(p_dict['Hs_m']) == len(p_dict['sigma_Hs']) == len(p_dict['setup_m']) == len(p_dict['sigma_setup']), "Your x, Hs, zb, and setup arrays are not the same length!"

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(p_dict['p_title'], fontsize=18, fontweight='bold', verticalalignment='top')

    # Hs
    ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
    min_val = np.nanmin(p_dict['Hs_m'] - p_dict['sigma_Hs'])
    max_val = np.nanmax(p_dict['Hs_m'] + p_dict['sigma_Hs'])
    min_x = np.min(p_dict['x'])
    max_x = np.max(p_dict['x'])
    dum_x = np.linspace(min_x - 0.05 * (max_x - min_x), max_x + 0.05 * (max_x - min_x), 100)

    if min_val < 0 and max_val > 0:
        ax1.plot(dum_x, np.zeros(len(dum_x)), 'k--')

    ax1.plot(p_dict['x'], p_dict['Hs_m'] - p_dict['sigma_Hs'], 'r--', label='$H_{s} \pm \sigma_{H_{s}}$')
    ax1.plot(p_dict['x'], p_dict['Hs_m'] + p_dict['sigma_Hs'], 'r--')
    ax1.plot(p_dict['x'], p_dict['Hs_m'], 'b-', label='Model $H_{s}$')

    # observation plots HOOOOOOO!
    temp35 = Adopp_35['Hs'][Adopp_35['plot_ind'] == 1]
    temp6m = AWAC6m['Hs'][AWAC6m['plot_ind'] == 1]
    temp8m = AWAC8m['Hs'][AWAC8m['plot_ind'] == 1]
    # Adopp_35
    ax1.plot(Adopp_35['xFRF']*np.ones(2), [temp35 - np.std(Adopp_35['Hs']), temp35 + np.std(Adopp_35['Hs'])], 'k-', label='Gage Data')
    ax1.plot(Adopp_35['xFRF']*np.ones(1), [temp35], 'k_')
    # AWAC6m
    ax1.plot(AWAC6m['xFRF']*np.ones(2), [temp6m - np.std(AWAC6m['Hs']), temp6m + np.std(AWAC6m['Hs'])], 'k-')
    ax1.plot(AWAC6m['xFRF']*np.ones(1), [temp6m], 'k_')
    # AWAC8m
    ax1.plot(AWAC8m['xFRF']*np.ones(2), [temp8m - np.std(AWAC8m['Hs']), temp8m + np.std(AWAC8m['Hs'])], 'k-')
    ax1.plot(AWAC8m['xFRF']*np.ones(1), [temp8m], 'k_')


    ax1.set_ylabel('$H_{s}$ [$m$]', fontsize=16)
    ax1.set_xlabel('Cross-shore Position [$m$]', fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax1.set_ylim([sf1 * min_val, sf2 * max_val])
    ax1.set_xlim([min(dum_x), max(dum_x)])

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    ax1.tick_params(labelsize=14)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, borderaxespad=0., fontsize=14)

    # Setup
    ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)
    min_val = np.nanmin(p_dict['setup_m'] - p_dict['sigma_setup'])
    max_val = np.nanmax(p_dict['setup_m'] + p_dict['sigma_setup'])

    if min_val < 0 and max_val > 0:
        ax2.plot(dum_x, np.zeros(len(dum_x)), 'k--')

    ax2.plot(p_dict['x'], p_dict['setup_m'] - p_dict['sigma_setup'], 'r--', label='$W_{setup} \pm \sigma_{W_{setup}}$')
    ax2.plot(p_dict['x'], p_dict['setup_m'] + p_dict['sigma_setup'], 'r--')
    ax2.plot(p_dict['x'], p_dict['setup_m'], 'b-', label='Model $W_{setup}$')

    ax2.set_ylabel('$W_{setup}$ [$m$]', fontsize=16)
    ax2.set_xlabel('Cross-shore Position [$m$]', fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax2.set_ylim([sf1 * min_val, sf2 * max_val])
    ax2.set_xlim([min(dum_x), max(dum_x)])

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    ax2.tick_params(labelsize=14)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0., fontsize=14)

    # Zb
    ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1)
    min_val = np.nanmin(p_dict['zb_m'] - p_dict['sigma_zbm'])
    max_val = np.nanmax(p_dict['zb_m'] + p_dict['sigma_zbm'])

    if len(p_dict['veg_ind']) > 0:
        ax3.plot(p_dict['x'], p_dict['zb_m'] - p_dict['sigma_zbm'], 'r--', label='$z_{b} \pm \sigma_{z_{b}}$')
        ax3.plot(p_dict['x'], p_dict['zb_m'] + p_dict['sigma_zbm'], 'r--')
        ax3.plot(p_dict['x'][p_dict['veg_ind']], p_dict['zb_m'][p_dict['veg_ind']], 'g-', label='Vegetated $z_{b}$')
        ax3.plot(p_dict['x'][p_dict['non_veg_ind']], p_dict['zb_m'][p_dict['non_veg_ind']], 'y-', label='Non-vegetated $z_{b}$')
        # get the bathy date
        zb_date = p_dict['model_time'].strftime('%Y-%m-%dT%H:%M:%SZ')
        ax3.text(0.75, 0.75, 'Bathymetry Date:\n' + zb_date, transform=ax3.transAxes, color='black', fontsize=14)
        col_num = 4
    else:
        ax3.plot(p_dict['x'], p_dict['zb_m'] - p_dict['sigma_zbm'], 'r--', label='$z_{b} \pm \sigma_{z_{b}}$')
        ax3.plot(p_dict['x'], p_dict['zb_m'] + p_dict['sigma_zbm'], 'r--')
        ax3.plot(p_dict['x'], p_dict['zb_m'], 'y-', label='Model $z_{b}$')
        zb_date = p_dict['model_time'].strftime('%Y-%m-%dT%H:%M:%SZ')
        ax3.text(0.75, 0.75, 'Bathymetry Date:\n' + zb_date, transform=ax3.transAxes, color='black', fontsize=14)
        col_num = 3

    # add altimeter data!!
    temp05 = Alt05['zb'][Alt05['plot_ind'] == 1]
    temp04 = Alt04['zb'][Alt04['plot_ind'] == 1]
    temp03 = Alt03['zb'][Alt03['plot_ind'] == 1]
    # Alt05
    ax3.plot(Alt05['xFRF']*np.ones(2), [temp05 - np.std(Alt05['zb']), temp05 + np.std(Alt05['zb'])], 'k-', label='Gage Data')
    ax3.plot(Alt05['xFRF'] * np.ones(1), [temp05], 'k_')
    # Alt04
    ax3.plot(Alt04['xFRF']*np.ones(2), [temp04 - np.std(Alt04['zb']), temp04 + np.std(Alt04['zb'])], 'k-')
    ax3.plot(Alt04['xFRF'] * np.ones(1), [temp04], 'k_')
    # Alt03
    ax3.plot(Alt03['xFRF']*np.ones(2), [temp03 - np.std(Alt03['zb']), temp03 + np.std(Alt03['zb'])], 'k-')
    ax3.plot(Alt03['xFRF'] * np.ones(1), [temp03], 'k_')

    ax3.set_ylabel('Elevation (NAVD88) [$m$]', fontsize=16)
    ax3.set_xlabel('Cross-shore Position [$m$]', fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax3.set_ylim([sf1 * min_val, sf2 * max_val])
    ax3.set_xlim([min(dum_x), max(dum_x)])

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    ax3.tick_params(labelsize=14)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=col_num, borderaxespad=0., fontsize=14)

    fig.subplots_adjust(wspace=0.4, hspace=0.6)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
    fig.savefig(ofname, dpi=300)
    plt.close()

def als_results(ofname, p_dict, obs_dict):
    """
    This is just some script to visualize the alongshore current results from the model output at a particular time step
    :param file_path: this is the full file-path (string) to the location where the plot will be saved i.e., C://users... 
        DONT include the final '/' or the actual NAME of the plot!!!!!! 
    :param p_dict: 
        (1) a vector of x-positions for the bathymetry ('x')
            MAKE SURE THIS IS IN FRF COORDS!!!!
        (2) vector of MODEL bathymetry bed elevations ('zb_m')
        (3) datetime of the MODEL bathymetry ('model_time')
        (4) vector of model alongshore velocity at EACH model NODE at the TIME of the MODEL BATHYMETRY ('vmean_m')
        (5) vector of the standard deviation of model alongshore velocity at EACH model NODE ('sigma_vm')
        (6) vector of model Hs at EACH model NODE at the TIME of the MODEL BATHYMETRY ('Hs')
        (7) vector of the standard deviation of model Hs at EACH model NODE ('sigma_Hs')
        (8) plot title (string) ('p_title')
        (9) array of datetimes for the water level data ('time').
            AS A HEADS UP, THIS IS THE RANGE OF TIMES THAT WILL GO INTO getObs for the comparisons!!!
    :return: plot of some alongshore current stuff
    """

    # Altimeter data!!!!!!!!
    Alt05 = obs_dict['Alt05']
    Alt04 = obs_dict['Alt04']
    Alt03 = obs_dict['Alt03']

    # wave data
    Adopp_35 = obs_dict['Adopp_35']
    AWAC6m = obs_dict['AWAC6m']
    AWAC8m = obs_dict['AWAC8m']

    # get rid of this and include them in the handed dictionary if you want to include vegetation in the plots
    p_dict['veg_ind'] = []
    p_dict['non_veg_ind'] = []

    # assert stuff here....
    assert len(p_dict['zb_m']) == len(p_dict['x']) == len(p_dict['vmean_m']) == len(p_dict['sigma_vm']), "Your x, zb, and y-vel arrays are not the same length!"

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(p_dict['p_title'], fontsize=18, fontweight='bold', verticalalignment='top')

    ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=1)

    # zb
    min_val = np.nanmin(p_dict['zb_m'])
    max_val = np.nanmax(p_dict['zb_m'])
    min_x = np.min(p_dict['x'])
    max_x = np.max(p_dict['x'])
    dum_x = np.linspace(min_x - 0.05 * (max_x - min_x), max_x + 0.05 * (max_x - min_x), 100)
    if len(p_dict['veg_ind']) > 0:
        zb_date = p_dict['model_time'].strftime('%Y-%m-%d %H:%M')
        a, = ax1.plot(p_dict['x'][p_dict['veg_ind']], p_dict['zb_m'][p_dict['veg_ind']], 'g-', label='Vegetated $z_{b}$ ' + '(' + zb_date + ')')
        b, = ax1.plot(p_dict['x'][p_dict['non_veg_ind']], p_dict['zb_m'][p_dict['non_veg_ind']], 'y-', label='Non-vegetated $z_{b}$ ' + '(' + zb_date + ')')
        col_num = 5
    else:
        zb_date = p_dict['model_time'].strftime('%Y-%m-%d %H:%M')
        a, = ax1.plot(p_dict['x'], p_dict['zb_m'], 'y-', label='Model $z_{b}$ ' + '(' + zb_date + ')')
        col_num = 4

    # add altimeter data!!
    temp05 = Alt05['zb'][Alt05['plot_ind'] == 1]
    temp04 = Alt04['zb'][Alt04['plot_ind'] == 1]
    temp03 = Alt03['zb'][Alt03['plot_ind'] == 1]
    # Alt05
    c, = ax1.plot(Alt05['xFRF']*np.ones(2), [temp05 - np.std(Alt05['zb']), temp05 + np.std(Alt05['zb'])], 'y-', label='Altimeter')
    d, = ax1.plot(Alt05['xFRF'] * np.ones(1), [temp05], 'y_')
    # Alt04
    e, = ax1.plot(Alt04['xFRF']*np.ones(2), [temp04 - np.std(Alt04['zb']), temp04 + np.std(Alt04['zb'])], 'y-')
    f, = ax1.plot(Alt04['xFRF'] * np.ones(1), [temp04], 'y_')
    # Alt03
    g, = ax1.plot(Alt03['xFRF']*np.ones(2), [temp03 - np.std(Alt03['zb']), temp03 + np.std(Alt03['zb'])], 'y-')
    h, = ax1.plot(Alt03['xFRF'] * np.ones(1), [temp03], 'y_')

    ax1.set_ylabel('Elevation (NAVD88) [$m$]', fontsize=16)
    # ax1.set_xlabel('Cross-shore Position [$m$]', fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax1.set_ylim([sf1 * min_val, sf2 * max_val])
    ax1.set_xlim([min(dum_x), max(dum_x)])
    ax1.tick_params('y', colors='g')
    ax1.yaxis.label.set_color('green')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax1.tick_params(labelsize=14)

    # y-vel
    min_val = np.nanmin(p_dict['vmean_m'] - p_dict['sigma_vm'])
    max_val = np.nanmax(p_dict['vmean_m'] + p_dict['sigma_vm'])
    ax2 = ax1.twinx()
    if min_val < 0 and max_val > 0:
        ax2.plot(dum_x, np.zeros(len(dum_x)), 'b--')

    i, = ax2.plot(p_dict['x'], p_dict['vmean_m'], 'b-', label='Model $V$')

    # velocity data HOOOOOOOOO!
    temp35V = Adopp_35['V'][Adopp_35['plot_ind_V'] == 1]
    temp6mV = AWAC6m['V'][AWAC6m['plot_ind_V'] == 1]
    temp8mV = AWAC8m['V'][AWAC8m['plot_ind_V'] == 1]
    # Adopp_35
    j, = ax2.plot(Adopp_35['xFRF']*np.ones(2), [temp35V - np.std(Adopp_35['V']), temp35V + np.std(Adopp_35['V'])], 'b:', label='Current Data')
    k, = ax2.plot(Adopp_35['xFRF']*np.ones(1), [temp35V], 'b_')
    # AWAC6m
    l, = ax2.plot(AWAC6m['xFRF']*np.ones(2), [temp6mV - np.std(AWAC6m['V']), temp6mV + np.std(AWAC6m['V'])], 'b:')
    m, = ax2.plot(AWAC6m['xFRF']*np.ones(1), [temp6mV], 'b_')
    try:
        # AWAC8m
        n, = ax2.plot(AWAC8m['xFRF']*np.ones(2), [temp8mV - np.std(AWAC8m['V']), temp8mV + np.std(AWAC8m['V'])], 'b:')
        o, = ax2.plot(AWAC8m['xFRF']*np.ones(1), [temp8mV], 'b_')
    except:
        pass

    ax2.set_ylabel('Along-shore Current [$m/s$]', fontsize=16)
    # ax2.set_xlabel('Cross-shore Position [$m$]', fontsize=16)

    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax2.set_ylim([sf1 * min_val, sf2 * max_val])
    ax2.set_xlim([min(dum_x), max(dum_x)])
    ax2.tick_params('y', colors='b')
    ax2.yaxis.label.set_color('blue')

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax2.tick_params(labelsize=14)

    if col_num == 5:
        p = [a, b, c, i, j]
    else:
        p = [a, c, i, j]

    ax1.legend(p, [p_.get_label() for p_ in p], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=col_num,
               borderaxespad=0., fontsize=14)

    ax3 = plt.subplot2grid((2, 1), (1, 0), colspan=1)

    # zb
    min_val = np.nanmin(p_dict['zb_m'])
    max_val = np.nanmax(p_dict['zb_m'])
    min_x = np.min(p_dict['x'])
    max_x = np.max(p_dict['x'])
    dum_x = np.linspace(min_x - 0.05 * (max_x - min_x), max_x + 0.05 * (max_x - min_x), 100)
    if len(p_dict['veg_ind']) > 0:
        zb_date = p_dict['model_time'].strftime('%Y-%m-%d %H:%M')
        a, = ax3.plot(p_dict['x'][p_dict['veg_ind']], p_dict['zb_m'][p_dict['veg_ind']], 'g-', label='Vegetated $z_{b}$ ' + '(' + zb_date + ')')
        b, = ax3.plot(p_dict['x'][p_dict['non_veg_ind']], p_dict['zb_m'][p_dict['non_veg_ind']], 'y-', label='Non-vegetated $z_{b}$ ' + '(' + zb_date + ')')
        col_num = 5
    else:
        zb_date = p_dict['model_time'].strftime('%Y-%m-%d %H:%M')
        a, = ax3.plot(p_dict['x'], p_dict['zb_m'], 'y-', label='Model $z_{b}$ ' + '(' + zb_date + ')')
        col_num = 4


    # add altimeter data!!
    temp05 = Alt05['zb'][Alt05['plot_ind'] == 1]
    temp04 = Alt04['zb'][Alt04['plot_ind'] == 1]
    temp03 = Alt03['zb'][Alt03['plot_ind'] == 1]
    # Alt05
    c, = ax3.plot(Alt05['xFRF']*np.ones(2), [temp05 - np.std(Alt05['zb']), temp05 + np.std(Alt05['zb'])], 'y-', label='Altimeter')
    d, = ax3.plot(Alt05['xFRF'] * np.ones(1), [temp05], 'y_')
    # Alt04
    e, = ax3.plot(Alt04['xFRF']*np.ones(2), [temp04 - np.std(Alt04['zb']), temp04 + np.std(Alt04['zb'])], 'y-')
    f, = ax1.plot(Alt04['xFRF'] * np.ones(1), [temp04], 'y_')
    # Alt03
    g, = ax3.plot(Alt03['xFRF']*np.ones(2), [temp03 - np.std(Alt03['zb']), temp03 + np.std(Alt03['zb'])], 'y-')
    h, = ax3.plot(Alt03['xFRF'] * np.ones(1), [temp03], 'y_')


    ax3.set_ylabel('Elevation (NAVD88) [$m$]', fontsize=16)
    ax3.set_xlabel('Cross-shore Position [$m$]', fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax3.set_ylim([sf1 * min_val, sf2 * max_val])
    ax3.set_xlim([min(dum_x), max(dum_x)])
    ax3.tick_params('y', colors='g')
    ax3.yaxis.label.set_color('green')

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax3.tick_params(labelsize=14)

    # Hs
    min_val = np.nanmin(p_dict['Hs_m'] - p_dict['sigma_Hs'])
    max_val = np.nanmax(p_dict['Hs_m'] + p_dict['sigma_Hs'])
    ax4 = ax3.twinx()
    if min_val < 0 and max_val > 0:
        ax4.plot(dum_x, np.zeros(len(dum_x)), 'b--')

    i, = ax4.plot(p_dict['x'], p_dict['Hs_m'], 'b-', label='Model $H_{s}$')

    # observation plots HOOOOOOO!
    temp35 = Adopp_35['Hs'][Adopp_35['plot_ind'] == 1]
    temp6m = AWAC6m['Hs'][AWAC6m['plot_ind'] == 1]
    temp8m = AWAC8m['Hs'][AWAC8m['plot_ind'] == 1]
    # Adopp_35
    j, = ax4.plot(Adopp_35['xFRF'] * np.ones(2), [temp35 - np.std(Adopp_35['Hs']), temp35 + np.std(Adopp_35['Hs'])], 'b:', label='Wave Data')
    k, = ax4.plot(Adopp_35['xFRF'] * np.ones(1), [temp35], 'b_')
    # AWAC6m
    l, = ax4.plot(AWAC6m['xFRF'] * np.ones(2), [temp6m - np.std(AWAC6m['Hs']), temp6m + np.std(AWAC6m['Hs'])], 'b:')
    m, = ax4.plot(AWAC6m['xFRF'] * np.ones(1), [temp6m], 'b_')
    try:
        # AWAC8m
        n, = ax4.plot(AWAC8m['xFRF'] * np.ones(2), [temp8m - np.std(AWAC8m['Hs']), temp8m + np.std(AWAC8m['Hs'])], 'b:')
        o, = ax4.plot(AWAC8m['xFRF'] * np.ones(1), [temp8m], 'b_')
    except:
        pass

    ax4.set_ylabel('$H_{s}$ [$m$]', fontsize=16)
    ax4.set_xlabel('Cross-shore Position [$m$]', fontsize=16)

    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax4.set_ylim([sf1 * min_val, sf2 * max_val])
    ax4.set_xlim([min(dum_x), max(dum_x)])
    ax4.tick_params('y', colors='b')
    ax4.yaxis.label.set_color('blue')

    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax4.tick_params(labelsize=14)

    if col_num == 5:
        p = [a, b, c, i, j]
    else:
        p = [a, c, i, j]

    ax3.legend(p, [p_.get_label() for p_ in p], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=col_num,
               borderaxespad=0., fontsize=14)

    fig.subplots_adjust(wspace=0.4, hspace=1.0)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
    fig.savefig(ofname, dpi=300)
    plt.close()

def alt_PlotData(name, mod_time, mod_times, THREDDS='FRF'):

    """
    This function is just to remove clutter in my plot functions
    all it does is pull out altimeter data and put it into the appropriate dictionary keys.
    If None, it will return masked arrays.

    :param name: name of the altimeter you want (Alt03, Alt04, Alt05)
    :param mod_time: start time of the model
    :param time: array of model datetimes
    :param mod: array of model observations at that instrument location corresponding to variable "comp_time"

    :return: Altimeter data dictionary with keys:
            'zb' - elevation
            'name' - gage name
            'time' - timestamps of the data
            'xFRF' - position of the gage
            'plot_ind' - this just tells it which data point it should plot for the snapshots
    """
    t1 = mod_times[0] - DT.timedelta(days=0, hours=0, minutes=3)
    t2 = mod_times[-1] + DT.timedelta(days=0, hours=0, minutes=3)
    frf_Data = getObs(t1, t2, THREDDS)

    try:
        dict = {}
        alt_data = frf_Data.getALT(name)
        dict['zb'] = alt_data['bottomElev']
        dict['time'] = alt_data['time']
        dict['name'] = alt_data['gageName']
        dict['xFRF'] = round(alt_data['xFRF'])
        plot_ind = np.where(abs(dict['time'] - mod_time) == min(abs(dict['time'] - mod_time)), 1, 0)
        dict['plot_ind'] = plot_ind
        dict['TS_toggle'] = True


    except:
        # just make it a masked array
        dict = {}
        comp_time = [mod_times[ii] + (mod_times[ii + 1] - mod_times[ii]) / 2 for ii in range(0, len(mod_times) - 1)]
        dict['time'] = comp_time
        fill_x = np.ones(np.shape(dict['time']))
        dict['zb'] = np.ma.array(fill_x, mask=np.ones(np.shape(dict['time'])))
        dict['name'] = name
        dict['xFRF'] = np.ma.array(np.ones(1), mask=np.ones(1))
        fill_ind = np.zeros(np.shape(dict['time']))
        fill_ind[0] = 1
        dict['plot_ind'] = fill_ind
        dict['TS_toggle'] = False


    return dict

def wave_PlotData(name, mod_time, time, THREDDS='FRF'):
    """
    This function is just to remove clutter in my plotting scripts
    all it does is pull out altimeter data and put it into the appropriate dictionary keys.
    If None, it will return masked arrays.

    :param t1: start time you want to pull (a datetime, not a string)
    :param t2: end time you want to pull (a datetime, not a string)
    :param name: name of the wave gage you want
    :param mod_time: start time of the model
    :param time: array of model datetimes

    :return: Altimeter data dictionary with keys:
            'Hs' - significant wave height
            'name' - gage name
            'wave_time' - timestamps of the data
            'xFRF' - position of the gage
            'plot_ind' - this just tells it which data point it should plot for the snapshots
    """

    t1 = time[0] - DT.timedelta(days=0, hours=0, minutes=3)
    t2 = time[-1] + DT.timedelta(days=0, hours=0, minutes=3)

    frf_Data = getObs(t1, t2, THREDDS)

    try:

        dict = {}
        wave_data = frf_Data.getWaveSpec(gaugenumber=name)
        cur_data = frf_Data.getCurrents(name)

        dict['name'] = name
        dict['wave_time'] = wave_data['time']
        dict['cur_time'] = cur_data['time']
        dict['Hs'] = wave_data['Hs']
        dict['xFRF'] = wave_data['xFRF']
        dict['plot_ind'] = np.where(abs(dict['wave_time'] - mod_time) == min(abs(dict['wave_time'] - mod_time)), 1, 0)
        dict['plot_ind_V'] = np.where(abs(dict['cur_time'] - mod_time) == min(abs(dict['cur_time'] - mod_time)), 1, 0)
        # rotate my velocities!!!
        test_fun = lambda x: vectorRotation(x, theta=360 - (71.8 + (90 - 71.8) + 71.8))
        newV = [test_fun(x) for x in zip(cur_data['aveU'], cur_data['aveV'])]
        dict['U'] = zip(*newV)[0]
        dict['V'] = zip(*newV)[1]
        dict['TS_toggle'] = True

    except:
        print('No data at %s!  Will return masked array.') %name
        # just make it a masked array
        dict = {}
        dict['wave_time'] = time
        dict['cur_time'] = time
        fill_x = np.ones(np.shape(dict['wave_time']))
        dict['Hs'] = np.ma.array(fill_x, mask=np.ones(np.shape(dict['wave_time'])))
        dict['name'] = 'AWAC8m'
        dict['xFRF'] = np.ma.array(np.ones(1), mask=np.ones(1))
        fill_ind = np.zeros(np.shape(dict['wave_time']))
        fill_ind[0] = 1
        dict['plot_ind'] = fill_ind
        dict['plot_ind_V'] = fill_ind
        dict['U'] = np.ma.array(fill_x, mask=np.ones(np.shape(dict['wave_time'])))
        dict['V'] = np.ma.array(fill_x, mask=np.ones(np.shape(dict['wave_time'])))
        dict['TS_toggle'] = False

    return dict

def lidar_PlotData(time, THREDDS='FRF'):

    t1 = time[0] - DT.timedelta(days=0, hours=0, minutes=3)
    t2 = time[-1] + DT.timedelta(days=0, hours=0, minutes=3)

    frf_Data = getObs(t1, t2, THREDDS)

    try:
        dict = {}
        lidar_data_RU = frf_Data.getLidarRunup()
        dict['runup2perc'] = lidar_data_RU['totalWaterLevel']
        dict['runupTime'] = lidar_data_RU['time']
        dict['runupMean'] = np.nanmean(lidar_data_RU['elevation'], axis=1)

        lidar_data_WP = frf_Data.getLidarWaveProf()
        dict['waveTime'] = lidar_data_WP['time']
        dict['xFRF'] = lidar_data_WP['xFRF']
        dict['yFRF'] = lidar_data_WP['yFRF']
        dict['Hs'] = lidar_data_WP['waveHsTotal']
        dict['WL'] = lidar_data_WP['waterLevel']

    except:
        # just make it a masked array
        dict['runupTime'] = np.zeros(20)
        fill_x = np.ones(np.shape(dict['runupTime']))
        dict['runup2perc'] = np.ma.array(fill_x, mask=np.ones(np.shape(dict['runupTime'])))
        dict['runupMean'] = np.ma.array(fill_x, mask=np.ones(np.shape(dict['runupTime'])))
        dict['waveTime'] = np.zeros(20)
        dict['xFRF'] = np.ones(np.shape(dict['waveTime']))
        dict['yFRF'] = np.ones(np.shape(dict['waveTime']))
        fill_x = np.ones((np.shape(dict['waveTime'])[0], np.shape(dict['xFRF'])[0]))
        dict['Hs'] = np.ma.array(fill_x, mask=np.ones(np.shape(fill_x)))
        dict['WL'] = np.ma.array(fill_x, mask=np.ones(np.shape(fill_x)))
        dict['TS_toggle'] = False

    return dict

def obs_V_mod_bathy_TN(ofname, p_dict, obs_dict, logo_path='ArchiveFolder/CHL_logo.png', contour_s=3, contour_d=8):
    """
    This is a plot to compare observed and model bathymetry to each other
    :param file_path: this is the full file-path (string) to the location where the plot will be saved i.e., C://users...
        DONT include the final '/' or the actual NAME of the plot!!!!!!
    :param p_dict:
        (1) a vector of x-positions for the bathymetry ('x')
            MAKE SURE THIS IS IN FRF COORDS!!!!

        (2) vector of initial OBSERVED bathymetry bed elevations ('i_obs')
        (3) datetime of the initial OBSERVED bathymetry survey ('i_obs_time')

        (3) vector of final OBSERVED bathymetry bed elevations ('f_obs')
        (4) datetime of the final OBSERVED bathymetry survey ('f_obs_time')

        (5) vector of MODEL bathymetry bed elevations ('model')
        (6) datetime of the MODEL bathymetry ('model_time')
        (7) vector of model Hs at EACH model NODE at the TIME of the MODEL BATHYMETRY ('Hs')
        (8) vector of the standard deviation of model Hs at EACH model NODE ('sigma_Hs')
        (9) time series of water level at the offshore boundary ('WL')
        (13) array of datetimes for the water level data ('time').
            AS A HEADS UP, THIS IS THE RANGE OF TIMES THAT WILL GO INTO getObs for the comparisons!!!
        (10) variable name ('var_name')
        (11) variable units ('units') (string) -> this will be put inside a tex math environment!!!!
        (12) plot title (string) ('p_title')

    :param logo_path: this is the path to get the CHL logo to display it on the plot!!!!
    :param contour_s: this is the INSIDE THE SANDBAR contour line (shallow contour line)
        we are going out to for the volume calculations (depth in m!!)
    :param contour_d: this is the OUTSIDE THE SANDBAR contour line (deep contour line)
        we are going out to for the volume calculations (depth in m!!)
    :return:
        model to observation comparison for spatial data - right now all it does is bathymetry?
        may need modifications to be more general, but I'm not sure what other
        kind of data would need to be plotted in a similar manner?
    """

    # Altimeter data!!!!!!!!
    Alt05 = obs_dict['Alt05']
    Alt04 = obs_dict['Alt04']
    Alt03 = obs_dict['Alt03']

    # wave data
    Adopp_35 = obs_dict['Adopp_35']
    AWAC6m = obs_dict['AWAC6m']
    AWAC8m = obs_dict['AWAC8m']

    assert len(p_dict['sigma_Hs']) == len(p_dict['Hs']) == len(p_dict['x']) == len(p_dict['i_obs']) == len(p_dict['f_obs']) == len(p_dict['model']), "Your x, Hs, model, and observation arrays are not the same length!"

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(p_dict['p_title'], fontsize=18, fontweight='bold', verticalalignment='top')

    # transects
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    min_val = np.min([np.min(p_dict['i_obs']), np.min(p_dict['model']), np.min(p_dict['f_obs'])])
    max_val = np.max([np.max(p_dict['i_obs']), np.max(p_dict['model']), np.max(p_dict['f_obs'])])
    min_x = np.min(p_dict['x'])
    max_x = np.max(p_dict['x'])
    dum_x = np.linspace(min_x - 0.05 * (max_x - min_x), max_x + 0.05 * (max_x - min_x), 100)
    a, = ax1.plot(dum_x, np.mean(p_dict['WL']) * np.ones(len(dum_x)), 'b-', label='Mean WL')
    # get the time strings!!
    i_obs_date = p_dict['i_obs_time'].strftime('%Y-%m-%d %H:%M')
    model_date = p_dict['model_time'].strftime('%Y-%m-%d %H:%M')
    f_obs_date = p_dict['f_obs_time'].strftime('%Y-%m-%d %H:%M')
    b, = ax1.plot(p_dict['x'], p_dict['i_obs'], 'r--', label='Initial Observed \n' + i_obs_date)
    c, = ax1.plot(p_dict['x'], p_dict['model'], 'y-', label='Model \n' + model_date)
    r, = ax1.plot(p_dict['x'], p_dict['f_obs'], 'r-', label='Final Observed \n' + f_obs_date)

    # add altimeter data!!
    # Alt05
    f, = ax1.plot(Alt05['xFRF'] * np.ones(2), [min(Alt05['zb']), max(Alt05['zb'])], 'k-', label='Gage Data')
    g, = ax1.plot(Alt05['xFRF'] * np.ones(1), Alt05['zb'][Alt05['plot_ind'] == 1], 'k_', label='Gage Data')
    # Alt04
    h, = ax1.plot(Alt04['xFRF'] * np.ones(2), [min(Alt04['zb']), max(Alt04['zb'])], 'k-', label='Gage Data')
    i, = ax1.plot(Alt04['xFRF'] * np.ones(1), Alt04['zb'][Alt04['plot_ind'] == 1], 'k_', label='Gage Data')
    # Alt03
    j, = ax1.plot(Alt03['xFRF'] * np.ones(2), [min(Alt03['zb']), max(Alt03['zb'])], 'k-', label='Gage Data')
    k, = ax1.plot(Alt03['xFRF'] * np.ones(1), Alt03['zb'][Alt03['plot_ind'] == 1], 'k_', label='Gage Data')

    ax5 = ax1.twinx()
    d, = ax5.plot(p_dict['x'], p_dict['Hs'], 'g-', label='Model $H_{s}$')
    e, = ax5.plot(p_dict['x'], p_dict['Hs'] + p_dict['sigma_Hs'], 'g--', label='$H_{s} \pm \sigma_{H_{s}}$')
    ax5.plot(p_dict['x'], p_dict['Hs'] - p_dict['sigma_Hs'], 'g--')

    # add wave data!!
    # Adopp_35
    l, = ax5.plot(Adopp_35['xFRF'] * np.ones(2), [Adopp_35['Hs'][Adopp_35['plot_ind'] == 1] - np.std(Adopp_35['Hs']), Adopp_35['Hs'][Adopp_35['plot_ind'] == 1] + np.std(Adopp_35['Hs'])], 'k-', label='Gage Data')
    m, = ax5.plot(Adopp_35['xFRF'] * np.ones(1), Adopp_35['Hs'][Adopp_35['plot_ind'] == 1], 'k_', label='Gage Data')
    # AWAC6m
    n, = ax5.plot(AWAC6m['xFRF'] * np.ones(2), [AWAC6m['Hs'][AWAC6m['plot_ind'] == 1] - np.std(AWAC6m['Hs']), AWAC6m['Hs'][AWAC6m['plot_ind'] == 1] + np.std(AWAC6m['Hs'])], 'k-', label='Gage Data')
    o, = ax5.plot(AWAC6m['xFRF'] * np.ones(1), AWAC6m['Hs'][AWAC6m['plot_ind'] == 1], 'k_', label='Gage Data')
    # AWAC8m
    p, = ax5.plot(AWAC8m['xFRF'] * np.ones(2), [AWAC8m['Hs'][AWAC8m['plot_ind'] == 1] - np.std(AWAC8m['Hs']), AWAC8m['Hs'][AWAC8m['plot_ind'] == 1] + np.std(AWAC8m['Hs'])], 'k-', label='Gage Data')
    q, = ax5.plot(AWAC8m['xFRF'] * np.ones(1), AWAC8m['Hs'][AWAC8m['plot_ind'] == 1], 'k_', label='Gage Data')

    ax1.set_ylabel('Elevation (NAVD88) [$%s$]' % (p_dict['units']), fontsize=16)
    ax1.set_xlabel('Cross-shore Position [$%s$]' % (p_dict['units']), fontsize=16)
    ax5.set_ylabel('$H_{s}$ [$%s$]' % (p_dict['units']), fontsize=16)
    ax5.set_xlabel('Cross-shore Position [$%s$]' % (p_dict['units']), fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax1.set_ylim([sf1 * min_val, sf2 * max_val])
    ax1.set_xlim([min(dum_x), max(dum_x)])
    ax1.tick_params('y', colors='r')
    ax1.yaxis.label.set_color('red')

    ax5.tick_params('y', colors='g')
    ax5.set_ylim([-1.05 * max(p_dict['Hs'] + p_dict['sigma_Hs']), 1.05 * max(p_dict['Hs'] + p_dict['sigma_Hs'])])
    ax5.set_xlim([min(dum_x), max(dum_x)])
    ax5.yaxis.label.set_color('green')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax5.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax5.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax1.tick_params(labelsize=14)
    ax5.tick_params(labelsize=14)
    p = [a, b, c, r, f, d, e]
    ax1.legend(p, [p_.get_label() for p_ in p], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(p), borderaxespad=0., fontsize=10, handletextpad=0.05)

    #go ahead and make sure they all cover the same space.
    o_mask = np.ma.getmask(p_dict['f_obs'])
    o_mask2 = o_mask.copy()
    p_dict['f_obs'] = p_dict['f_obs'][~o_mask2]
    p_dict['i_obs'] = p_dict['i_obs'][~o_mask2]
    p_dict['model'] = p_dict['model'][~o_mask2]
    p_dict['x'] = p_dict['x'][~o_mask2]




    # 1 to 1
    one_one = np.linspace(min_val - 0.05 * (max_val - min_val), max_val + 0.05 * (max_val - min_val), 100)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax2.plot(one_one, one_one, 'k-', label='$45^{0}$-line')
    if min_val < 0 and max_val > 0:
        ax2.plot(one_one, np.zeros(len(one_one)), 'k--')
        ax2.plot(np.zeros(len(one_one)), one_one, 'k--')
    ax2.plot(p_dict['f_obs'], p_dict['model'], 'r*')
    ax2.set_xlabel('Final Observed %s [$%s$]' % (p_dict['var_name'], p_dict['units']), fontsize=14)
    ax2.set_ylabel('Model %s [$%s$]' % (p_dict['var_name'], p_dict['units']), fontsize=14)
    ax2.set_xlim([min_val - 0.025 * (max_val - min_val), max_val + 0.025 * (max_val - min_val)])
    ax2.set_ylim([min_val - 0.025 * (max_val - min_val), max_val + 0.025 * (max_val - min_val)])
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax2.tick_params(labelsize=14)
    plt.legend(loc=0, ncol=1, borderaxespad=0.5, fontsize=14)

    # stats and stats text
    stats_dict = {}
    stats_dict['bias'] = np.mean(p_dict['f_obs'] - p_dict['model'])
    stats_dict['RMSE'] = np.sqrt((1 / (float(len(p_dict['f_obs'])) - 1)) * np.sum(np.power(p_dict['f_obs'] - p_dict['model'] - stats_dict['bias'], 2)))
    stats_dict['sym_slp'] = np.sqrt(np.sum(np.power(p_dict['f_obs'], 2)) / float(np.sum(np.power(p_dict['model'], 2))))
    # correlation coef
    dum = np.zeros([2, len(p_dict['model'])])
    dum[0] = p_dict['model'].flatten()
    dum[1] = p_dict['f_obs'].flatten()
    stats_dict['corr_coef'] = np.corrcoef(dum)[0, 1]

    # volume change
    # shallow - predicted
    index_XXm = np.min(np.argwhere(p_dict['i_obs'] >= -1 * contour_s).flatten())  # ok, the indices currently count from offshore to onshore, so we want the SMALLEST index!
    vol_obs = np.trapz(p_dict['i_obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    vol_model = np.trapz(p_dict['model'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    stats_dict['pred_vol_change_%sm' % (contour_s)] = vol_model - vol_obs
    # shallow - actual
    index_XXm = np.min(np.argwhere(p_dict['i_obs'] >= -1 * contour_s).flatten())  # ok, the indices currently count from offshore to onshore, so we want the SMALLEST index!
    vol_obs_i = np.trapz(p_dict['i_obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    vol_obs_f = np.trapz(p_dict['f_obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    stats_dict['actual_vol_change_%sm' % (contour_s)] = vol_obs_f - vol_obs_i

    # deep - predicted
    index_XXm = np.min(np.argwhere(p_dict['i_obs'] >= -1 * contour_d).flatten())  # ok, the indices currently count from offshore to onshore, so we want the SMALLEST index!
    vol_obs = np.trapz(p_dict['i_obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    vol_model = np.trapz(p_dict['model'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    stats_dict['pred_vol_change_%sm' % (contour_d)] = vol_model - vol_obs

    # deep - actual
    index_XXm = np.min(np.argwhere(p_dict['i_obs'] >= -1 * contour_d).flatten())  # ok, the indices currently count from offshore to onshore, so we want the SMALLEST index!
    vol_obs_i = np.trapz(p_dict['i_obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    vol_obs_f = np.trapz(p_dict['f_obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    stats_dict['actual_vol_change_%sm' % (contour_d)] = vol_obs_i - vol_obs_f

    header_str = '%s Comparison \nModel to Observations:' % (p_dict['var_name'])
    bias_str = '\n Bias $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['bias']), p_dict['units'])
    RMSE_str = '\n RMSE $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['RMSE']), p_dict['units'])
    sym_slp_str = '\n Symmetric Slope $=%s$' % ("{0:.2f}".format(stats_dict['sym_slp']))
    corr_coef_str = '\n Correlation Coefficient $=%s$' % ("{0:.2f}".format(stats_dict['corr_coef']))

    shall_vol_str_p = '\n Modeled $%s$ $%s$ Volume Change $=%s$ $(%s^{3}/%s)$' % (contour_s, p_dict['units'], "{0:.2f}".format(stats_dict['pred_vol_change_%sm' % (contour_s)]), p_dict['units'], p_dict['units'])
    shall_vol_str_a = '\n Observed $%s$ $%s$ Volume Change $=%s$ $(%s^{3}/%s)$' % (contour_s, p_dict['units'], "{0:.2f}".format(stats_dict['actual_vol_change_%sm' % (contour_s)]), p_dict['units'], p_dict['units'])

    deep_vol_str_p = '\n Modeled $%s$ $%s$ Volume Change $=%s$ $(%s^{3}/%s)$' % (contour_d, p_dict['units'], "{0:.2f}".format(stats_dict['pred_vol_change_%sm' % (contour_d)]), p_dict['units'], p_dict['units'])
    deep_vol_str_a = '\n Observed $%s$ $%s$ Volume Change $=%s$ $(%s^{3}/%s)$' % (contour_d, p_dict['units'], "{0:.2f}".format(stats_dict['actual_vol_change_%sm' % (contour_d)]), p_dict['units'], p_dict['units'])

    vol_expl_str = '*Note: volume change is defined as the \n $final$ volume minus the $initial$ volume'

    plot_str = bias_str + RMSE_str + sym_slp_str + corr_coef_str + shall_vol_str_p + shall_vol_str_a + deep_vol_str_p + deep_vol_str_a
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax3.axis('off')
    ax4 = ax3.twinx()
    ax3.axis('off')
    try:
        ax4.axis('off')
        dir_name = os.path.dirname(__file__).split('\\plotting')[0]
        CHL_logo = image.imread(os.path.join(dir_name, logo_path))
        ax4 = fig.add_axes([0.78, 0.02, 0.20, 0.20], anchor='SE', zorder=-1)
        ax4.imshow(CHL_logo)
        ax4.axis('off')
    except:
        print 'Plot generated sans CHL logo!'
    ax3.axis('off')
    ax3.text(0.01, 0.99, header_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=16, fontweight='bold')
    ax3.text(0.00, 0.90, plot_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=14)
    ax3.text(0.02, 0.41, vol_expl_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=12)

    fig.subplots_adjust(wspace=0.4, hspace=0.1)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(ofname, dpi=300)
    plt.close()











