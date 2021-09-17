import os
import matplotlib.pyplot as plt
import  numpy as np

def bathyQAQCplots(fig_loc, d1, updatedBathy):
    """
    This function will QA QC the integrated bathy scripts
    :param fig_loc:  location where to save plots to
    :param d1: a date time object to save/label figures
    :param updatedBathy: a dictionary output by the wrapper function
        :key 'elevation': 3d bathymetry [t, y, x]
        :key 'xFRF': x coordinates of FRF
        :key 'yFRF': y coordinates of FRF
        :key
    :return:
        a bunch of plots
    """
    xFRF = updatedBathy['xFRF']
    yFRF = updatedBathy['yFRF']

    t = -1  # plot the last one of the day (if multiple) if not, should be 3D anyway
##################################################################

    print('Making Figures {}'.format(d1.strftime('%Y%m%d')))
    if not os.path.exists(fig_loc):
        os.makedirs(fig_loc)

    # zoomed in pcolor plot on AOI
    fig_name = 'DEM_' + d1.strftime('%Y%m%dT%H%M%SZ') + '.png'
    plt.figure()
    plt.pcolormesh(xFRF, yFRF, updatedBathy['elevation'][t, :, :], cmap=plt.cm.jet, vmin=-13, vmax=5)
    cbar = plt.colorbar()
    cbar.set_label('(m)')
    axes = plt.gca()
    axes.set_xlim([-50, 550])
    axes.set_ylim([-50, 1050])
    plt.xlabel('xFRF (m)')
    plt.ylabel('yFRF (m)')
    plt.savefig(os.path.join(fig_loc, fig_name))
    plt.close()
########################################################################################################
    fname2 = 'AlongShore_smoothing_{}.png'.format(d1.strftime('%Y%m%dT%H%M%SZ'))
    plt.figure(figsize=(12, 12))
    plt.suptitle('AlongShore Smoothing Check')
    for val in [100, 200, 300, 400, 500, 600, 700]:
        idxYshore_smo = np.argmin(np.abs(xFRF - val))
        plt.plot(yFRF, updatedBathy['elevation'][t, :, idxYshore_smo], '.', label='x={}'.format(val))

    plt.xlim([-10, 1200])
    plt.legend()
    plt.savefig(os.path.join(fig_loc, fname2))
    plt.close()
    # alongshore transect plots
    x_loc_check1 = int(100)
    x_loc_check2 = int(200)
    x_loc_check3 = int(350)
    x_check1 = np.where(xFRF == x_loc_check1)[0][0]
    x_check2 = np.where(xFRF == x_loc_check2)[0][0]
    x_check3 = np.where(xFRF == x_loc_check3)[0][0]

    # plot X and Y transects from newZdiff to see if it looks correct?
    fig_name = 'DEM_Ytrans_X' + str(x_loc_check1) + '_X' + \
               str(x_loc_check2) + '_X' + str(x_loc_check3) + \
               d1.strftime('%Y%m%d') + '.png'
##############################################################################
    fig = plt.figure(figsize=(8, 9))
    ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
    ax1.plot(yFRF, updatedBathy['elevation'][-1, :, x_check1], 'r')
    ax1.set_xlabel('Alongshore - $y$ ($m$)', fontsize=16)
    ax1.set_ylabel('Elevation ($m$)', fontsize=16)
    ax1.set_title('$X=%s$' % (str(x_loc_check1)), fontsize=16)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax1.tick_params(labelsize=14)
    ax1.text(0.10, 0.95, '(a)', horizontalalignment='left', verticalalignment='top',
             transform=ax1.transAxes, fontsize=16)
    ax1.set_xlim([-50, 1050])

    ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)
    ax2.plot(yFRF, updatedBathy['elevation'][-1, :, x_check2], 'r')
    ax2.set_xlabel('Alongshore - $y$ ($m$)', fontsize=16)
    ax2.set_ylabel('Elevation ($m$)', fontsize=16)
    ax2.set_title('$X=%s$' % (str(x_loc_check2)), fontsize=16)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax2.tick_params(labelsize=14)
    ax2.text(0.10, 0.95, '(b)', horizontalalignment='left', verticalalignment='top',
             transform=ax2.transAxes, fontsize=16)
    ax2.set_xlim([-50, 1050])

    ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1)
    ax3.plot(yFRF, updatedBathy['elevation'][-1, :, x_check3], 'r', label='Original')
    ax3.set_xlabel('Alongshore - $y$ ($m$)', fontsize=16)
    ax3.set_ylabel('Elevation ($m$)', fontsize=16)
    ax3.set_title('$X=%s$' % (str(x_loc_check3)), fontsize=16)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax3.tick_params(labelsize=14)
    ax3.text(0.10, 0.95, '(c)', horizontalalignment='left', verticalalignment='top',
             transform=ax3.transAxes, fontsize=16)
    ax3.set_xlim([-50, 1050])

    fig.subplots_adjust(wspace=0.4, hspace=0.1)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
    fig.savefig(os.path.join(fig_loc, fig_name), dpi=300)
    plt.close()
    ################################################################################
    # cross-shore transect plots
    y_loc_check1 = int(250)
    y_loc_check2 = int(500)
    y_loc_check3 = int(750)
    y_check1 = np.where(yFRF == y_loc_check1)[0][0]
    y_check2 = np.where(yFRF == y_loc_check2)[0][0]
    y_check3 = np.where(yFRF == y_loc_check3)[0][0]
    # plot a transect going in the cross-shore just to check it
    fig_name = 'DEM_' + d1.strftime('%Y%m%d') + '_Xtrans_Y' + str(
        y_loc_check1) + '_Y' + str(y_loc_check2) + '_Y' + str(
        y_loc_check3) + '.png'

    fig = plt.figure(figsize=(8, 9))
    ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
    ax1.plot(xFRF, updatedBathy['elevation'][-1, y_check1, :], 'b')
    ax1.set_xlabel('Cross-shore - $x$ ($m$)', fontsize=16)
    ax1.set_ylabel('Elevation ($m$)', fontsize=16)
    ax1.set_title('$Y=%s$' % (str(y_loc_check1)), fontsize=16)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax1.tick_params(labelsize=14)
    ax1.text(0.10, 0.95, '(a)', horizontalalignment='left', verticalalignment='top',
             transform=ax1.transAxes, fontsize=16)
    ax1.set_xlim([-50, 550])

    ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)
    ax2.plot(xFRF, updatedBathy['elevation'][-1, y_check2, :], 'b')
    ax2.set_xlabel('Cross-shore - $x$ ($m$)', fontsize=16)
    ax2.set_ylabel('Elevation ($m$)', fontsize=16)
    ax2.set_title('$Y=%s$' % (str(y_loc_check2)), fontsize=16)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax2.tick_params(labelsize=14)
    ax2.text(0.10, 0.95, '(b)', horizontalalignment='left', verticalalignment='top',
             transform=ax2.transAxes, fontsize=16)
    ax2.set_xlim([-50, 550])

    ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1)
    ax3.plot(xFRF, updatedBathy['elevation'][-1, y_check3, :], 'b')
    ax3.set_xlabel('Cross-shore - $x$ ($m$)', fontsize=16)
    ax3.set_ylabel('Elevation ($m$)', fontsize=16)
    ax3.set_title('$Y=%s$' % (str(y_loc_check3)), fontsize=16)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax3.tick_params(labelsize=14)
    ax3.text(0.10, 0.95, '(c)', horizontalalignment='left', verticalalignment='top',
             transform=ax3.transAxes, fontsize=16)
    ax3.set_xlim([-50, 550])

    fig.subplots_adjust(wspace=0.4, hspace=0.1)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
    fig.savefig(os.path.join(fig_loc, fig_name), dpi=300)
    plt.close()
