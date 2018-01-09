import os
import numpy as np
from getdatatestbed.getDataFRF import getObs
import datetime as DT
import MakeUpdatedBathyDEM as mBATHY
from collections import OrderedDict
import netCDF4 as nc
from sblib import geoprocess as gp
import makenc
from matplotlib import pyplot as plt
from sblib import sblib as sb


def makeBathyCBATHY(dSTR_s, dSTR_e, dir_loc, scalecDict=None, splineDict=None, ncStep='daily', plot=None):

    """
    :param dSTR_s: string that determines the start date of the times of the surveys you want to use to update the DEM
                    format is  dSTR_s = '2013-01-04T00:00:00Z'
                    no matter what you put here, it will always round it down to the beginning of the month
    :param dSTR_e: string that determines the end date of the times of the surveys you want to use to update the DEM
                    format is dSTR_e = '2014-12-22T23:59:59Z'
                    no matter what you put here, it will always round it up to the end of the month
    :param dir_loc: place where you want to save the .nc files that get written
                    the function will make the year directories inside of this location on its own.
    :param scalecDict: keys are:
                        x_smooth - x direction smoothing length for scalecInterp
                        y_smooth - y direction smoothing length for scalecInterp

                        if not specified it will default to:
                        x_smooth = 100
                        y_smooth = 200
    :param splineDict: keys are:
                        splinebctype
                            options are....
                            2 - second derivative goes to zero at boundary
                            1 - first derivative goes to zero at boundary
                            0 - value is zero at boundary
                            10 - force value and derivative(first?!?) to zero at boundary
                        lc - spline smoothing constraint value (integer <= 1)
                        dxm -  coarsening of the grid for spline (e.g., 2 means calculate with a dx that is 2x input dx)
                                can be tuple if you want to do dx and dy separately (dxm, dym), otherwise dxm is used for both
                        dxi - fining of the grid for spline (e.g., 0.1 means return spline on a grid that is 10x input dx)
                                as with dxm, can be a tuple if you want separate values for dxi and dyi
                        targetvar - this is the target variance used in the spline function.
                        wbysmooth - y-edge smoothing length scale
                        wbxsmooth - x-edge smoothing length scale

                        if not specified it will default to:
                        splinebctype = 10
                        lc = 4
                        dxm = 2
                        dxi = 1
                        targetvar = 0.45
                        wbysmooth = 300
                        wbxsmooth = 100
    :param ncStep: do you want to make monthly or daily nc files?
                    options are 'monthly' or 'daily'.  'monthly' is NOT recommended for CBATHY!!!!!
    :return:
            writes ncfiles for the new DEM's to the location dir_loc
    """
    # we are starting from the most up-to-date integrated bathymetry product, so this is the ncml from that!
    nc_b_url = 'http://134.164.129.55/thredds/dodsC/cmtb/projects/integratedBathyForCbathy/newSurvey.ncml'
    # this is the ncml for the cBathy ingegrated bathymetries
    nc_url = 'FILL IN'

    # Yaml files for my .nc files!!!!!
    global_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFti_global.yml'
    var_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFti_grid_var.yml'

    # check the ncStep input
    d_s = None
    if 'monthly' in ncStep:
        # force the survey to start at the first of the month and end at the last of the month!!!!
        dSTR_s = dSTR_s[0:7] + '-01T00:00:00Z'
        if dSTR_e[5:7] == '12':
            dSTR_e = str(int(dSTR_e[0:4]) + 1) + '-01' + '-01T00:00:00Z'
        else:
            dSTR_e = dSTR_e[0:5] + str(int(dSTR_e[5:7]) + 1).zfill(2) + '-01T00:00:00Z'
        d_e = DT.datetime.strptime(dSTR_e, '%Y-%m-%dT%H:%M:%SZ')
        d_s = DT.datetime.strptime(dSTR_s, '%Y-%m-%dT%H:%M:%SZ')
        dList = OrderedDict(((d_s + DT.timedelta(_)).strftime(r"%b-%y"), None) for _ in xrange((d_e - d_s).days+31)).keys()
        dList = [DT.datetime.strptime(date, r"%b-%y") for date in dList]


    elif 'daily' in ncStep:
        # force the survey to start at the beginning of the day and end at the beginning of the next day
        dSTR_s = dSTR_s[0:11] + '00:00:00Z'
        dSTR_e = dSTR_e[0:11] + '00:00:00Z'
        d_e = DT.datetime.strptime(dSTR_e, '%Y-%m-%dT%H:%M:%SZ')
        d_e = d_e + DT.timedelta(days=1)
        d_s = DT.datetime.strptime(dSTR_s, '%Y-%m-%dT%H:%M:%SZ')
        dList = OrderedDict(((d_s + DT.timedelta(_)), None) for _ in xrange((d_e - d_s).days+1)).keys()

    else:
        assert d_s is not None, 'ncStep input not recognized.  Acceptable inputs include daily or monthly.'


    # loop time
    nsteps = np.size(dList)-1
    for tt in range(0, nsteps):

        # pull out the dates I need for this step!
        d1 = dList[tt]
        d2 = dList[tt+1]

        # get the data that I need
        frf_Data = getObs(d1, d2)
        xminC = 0
        xmaxC = 500
        yminC = 0
        ymaxC = 1000
        cBathy = frf_Data.getBathyGridcBathy(xbounds=[xminC, xmaxC], ybounds=[yminC, ymaxC])

        # prep the data
        # new data
        newDict = {}
        newDict['elevation'] = -1*cBathy['depthKF']
        newDict['xFRF'] = cBathy['xm']
        newDict['yFRF'] = cBathy['ym']

        # background data
        backgroundDict = {}

        try:
            # look for the .nc file that I just wrote!!!
            old_bathy = nc.Dataset(os.path.join(prev_nc_loc, prev_nc_name))
            ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units, old_bathy.variables['time'].calendar)
            # find newest time prior to this
            t_mask = (ob_times <= d_s)  # boolean true/false of time
            t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one
            backgroundDict['elevation'] = old_bathy.variables['elevation'][t_idx, :]
            backgroundDict['xFRF'] = old_bathy.variables['xFRF'][:]
            backgroundDict['yFRF'] = old_bathy.variables['yFRF'][:]
        except:
            try:
                # look for the most up to date bathy in the ncml file....
                old_bathy = nc.Dataset(nc_url)
                ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units, old_bathy.variables['time'].calendar)
                # find newest time prior to this
                t_mask = (ob_times <= d_s)  # boolean true/false of time
                t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one
                backgroundDict['elevation'] = old_bathy.variables['elevation'][t_idx, :]
                backgroundDict['xFRF'] = old_bathy.variables['xFRF'][:]
                backgroundDict['yFRF'] = old_bathy.variables['yFRF'][:]
            except:

                # this code is for if you decide you want to go back to the time-mean bathy!!!!
                """
                # pull the most up-to-date integrated bathymetry product (prior to the cbathy start date)
                old_bathy = nc.Dataset(nc_b_url)
                backgroundDict['elevation'] = old_bathy.variables['elevation'][:]
                backgroundDict['xFRF'] = old_bathy.variables['xFRF'][:]
                backgroundDict['yFRF'] = old_bathy.variables['yFRF'][:]
                """

                #
                old_bathy = nc.Dataset(nc_b_url)
                ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units, old_bathy.variables['time'].calendar)
                # find newest time prior to this
                t_mask = (ob_times <= d_s)  # boolean true/false of time
                t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one
                backgroundDict['elevation'] = old_bathy.variables['elevation'][t_idx, :]
                backgroundDict['xFRF'] = old_bathy.variables['xFRF'][:]
                backgroundDict['yFRF'] = old_bathy.variables['yFRF'][:]

                """
                # how we looking?
                fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestFigures_CBATHY'
                # background I made from the immediately preceding survey?
                tempD = d1.strftime(r"%Y-%m-%d")
                fig_name = 'backgroundDEM_' + tempD[0:4] + tempD[5:7] + '.png'
                plt.figure()
                plt.pcolor(backgroundDict['xFRF'], backgroundDict['yFRF'], backgroundDict['elevation'], cmap=plt.cm.jet, vmin=-13, vmax=5)
                cbar = plt.colorbar()
                cbar.set_label('(m)')
                axes = plt.gca()
                axes.set_xlim([-50, 550])
                axes.set_ylim([-50, 1050])
                plt.xlabel('xFRF (m)')
                plt.ylabel('yFRF (m)')
                plt.savefig(os.path.join(fig_loc, fig_name))
                plt.close()
                """
               
                t = 1






        # go time!  this is scaleC + spline!
        out = mBATHY.makeUpdatedBATHY(backgroundDict, newDict, scalecDict=scalecDict, splineDict=splineDict)


        # write this to the appropriate nc_file!
        # get position stuff that will be constant for all surveys!!!
        xFRFi, yFRFi = np.meshgrid(out['xFRF'], out['yFRF'])
        xFRFi_vecN = xFRFi.reshape((1, xFRFi.shape[0] * xFRFi.shape[1]))[0]
        yFRFi_vecN = yFRFi.reshape((1, yFRFi.shape[0] * yFRFi.shape[1]))[0]
        # convert FRF coords to lat/lon
        test = gp.FRF2ncsp(xFRFi_vecN, yFRFi_vecN)
        # go through stateplane to avoid FRFcoords trying to guess the input coordinate systems
        temp = gp.ncsp2LatLon(test['StateplaneE'], test['StateplaneN'])
        lat_vec = temp['lat']
        lon_vec = temp['lon']
        E_vec = test['StateplaneE']
        N_vec = test['StateplaneN']

        lat = lat_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
        lon = lon_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
        E = E_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
        N = N_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])

        xFRF = xFRFi[0, :]
        yFRF = yFRFi[:, 1]
        latitude = lat
        longitude = lon
        easting = E
        northing = N

        # write the nc_file for this month, like a boss, with greatness
        nc_dict = {}
        nc_dict['elevation'] = out['elevation']
        nc_dict['xFRF'] = xFRF
        nc_dict['yFRF'] = yFRF
        nc_dict['latitude'] = latitude
        nc_dict['longitude'] = longitude
        nc_dict['northing'] = northing
        nc_dict['easting'] = easting
        # also want cbathy time....
        timeunits = 'seconds since 1970-01-01 00:00:00'
        surveyTime = [nc.date2num(T, timeunits) for T in cBathy['time']]
        nc_dict['time'] = np.array(np.array(surveyTime).squeeze())

        # what does the name need to be?
        tempD = d1.strftime(r"%Y-%m-%d")
        if 'monthly' in ncStep:
            nc_name = 'FRF-updated_cBathy_dem_grids_' + tempD[0:4] + tempD[5:7] + '.nc'
        elif 'daily' in ncStep:
            nc_name = 'FRF-updated_cBathy_dem_grids_' + tempD[0:4] + tempD[5:7] + tempD[8:10] + '.nc'

        # check to see if year directory exists
        if os.path.isdir(os.path.join(dir_loc, tempD[0:4])):
            pass
        else:
            # if not, make year directory
            os.makedirs(os.path.join(dir_loc, tempD[0:4]))

        # where am I saving these nc's as I make them
        nc_loc = os.path.join(dir_loc, tempD[0:4])
        # save em like a boss?
        makenc.makenc_tiBATHY(os.path.join(nc_loc, nc_name), nc_dict, globalYaml=global_yaml, varYaml=var_yaml)

        t = 1

        # save this location for next time through the loop
        prev_nc_name = nc_name
        prev_nc_loc = nc_loc

        # plot some stuff
        plot = 1
        if plot is None:
            pass
        else:

            fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestFigures_CBATHY'

            # zoomed in pcolor plot on AOI
            fig_name = 'cBathyDEM_' + tempD[0:4] + tempD[5:7] + tempD[8:10] + '.png'
            plt.figure()
            plt.pcolor(xFRF, yFRF, out['elevation'][-1, :, :], cmap=plt.cm.jet, vmin=-13, vmax=5)
            cbar = plt.colorbar()
            cbar.set_label('(m)')
            axes = plt.gca()
            axes.set_xlim([-50, 550])
            axes.set_ylim([-50, 1050])
            plt.xlabel('xFRF (m)')
            plt.ylabel('yFRF (m)')
            plt.savefig(os.path.join(fig_loc, fig_name))
            plt.close()

            #alongshore transect plots
            x_loc_check1 = int(100)
            x_loc_check2 = int(200)
            x_loc_check3 = int(350)
            x_check1 = np.where(xFRF == x_loc_check1)[0][0]
            x_check2 = np.where(xFRF == x_loc_check2)[0][0]
            x_check3 = np.where(xFRF == x_loc_check3)[0][0]

            # plot X and Y transects from newZdiff to see if it looks correct?
            fig_name = 'cBathyDEM_' + tempD[0:4] + tempD[5:7] + tempD[8:10] + '_Ytrans_X' + str(
                x_loc_check1) + '_X' + str(x_loc_check2) + '_X' + str(x_loc_check3) + '.png'

            fig = plt.figure(figsize=(8, 9))
            ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
            ax1.plot(yFRF, out['elevation'][-1, :, x_check1], 'r')
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
            ax2.plot(yFRF, out['elevation'][-1, :, x_check2], 'r')
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
            ax3.plot(yFRF, out['elevation'][-1, :, x_check3], 'r', label='Original')
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

            # cross-shore transect plots
            y_loc_check1 = int(250)
            y_loc_check2 = int(500)
            y_loc_check3 = int(750)
            y_check1 = np.where(yFRF == y_loc_check1)[0][0]
            y_check2 = np.where(yFRF == y_loc_check2)[0][0]
            y_check3 = np.where(yFRF == y_loc_check3)[0][0]
            # plot a transect going in the cross-shore just to check it
            fig_name = 'cBathyDEM_' + tempD[0:4] + tempD[5:7] + tempD[8:10] + '_Xtrans_Y' + str(y_loc_check1) + '_Y' + str(y_loc_check2) + '_Y' + str(
                y_loc_check3) + '.png'

            fig = plt.figure(figsize=(8, 9))
            ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
            ax1.plot(xFRF, out['elevation'][-1, y_check1, :], 'b')
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
            ax2.plot(xFRF, out['elevation'][-1, y_check2, :], 'b')
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
            ax3.plot(xFRF, out['elevation'][-1, y_check3, :], 'b')
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


def makeBathySurvey(dSTR_s, dSTR_e, dir_loc, scalecDict=None, splineDict=None, ncStep='monthly', plot=None):
    """
    :param dSTR_s: string that determines the start date of the times of the surveys you want to use to update the DEM
                    format is  dSTR_s = '2013-01-04T00:00:00Z'
                    no matter what you put here, it will always round it down to the beginning of the month
    :param dSTR_e: string that determines the end date of the times of the surveys you want to use to update the DEM
                    format is dSTR_e = '2014-12-22T23:59:59Z'
                    no matter what you put here, it will always round it up to the end of the month
    :param dir_loc: place where you want to save the .nc files that get written
                    the function will make the year directories inside of this location on its own.
    :param scalecDict: keys are:
                        x_smooth - x direction smoothing length for scalecInterp
                        y_smooth - y direction smoothing length for scalecInterp

                        if not specified it will default to:
                        x_smooth = 100
                        y_smooth = 200
    :param splineDict: keys are:
                        splinebctype
                            options are....
                            2 - second derivative goes to zero at boundary
                            1 - first derivative goes to zero at boundary
                            0 - value is zero at boundary
                            10 - force value and derivative(first?!?) to zero at boundary
                        lc - spline smoothing constraint value (integer <= 1)
                        dxm -  coarsening of the grid for spline (e.g., 2 means calculate with a dx that is 2x input dx)
                                can be tuple if you want to do dx and dy separately (dxm, dym), otherwise dxm is used for both
                        dxi - fining of the grid for spline (e.g., 0.1 means return spline on a grid that is 10x input dx)
                                as with dxm, can be a tuple if you want separate values for dxi and dyi
                        targetvar - this is the target variance used in the spline function.
                        wbysmooth - y-edge smoothing length scale
                        wbxsmooth - x-edge smoothing length scale

                        if not specified it will default to:
                        splinebctype = 10
                        lc = 4
                        dxm = 2
                        dxi = 1
                        targetvar = 0.45
                        wbysmooth = 300
                        wbxsmooth = 100
    :param ncStep: do you want to make monthly or daily nc files?
                    options are 'monthly' or 'daily'.  'monthly' is NOT recommended for CBATHY!!!!!
    :return:
            writes ncfiles for the new DEM's to the location dir_loc
    """
    # this is the standard background bathymetry that we started from.
    nc_b_url = 'http://134.164.129.55/thredds/dodsC/cmtb/grids/TimeMeanBackgroundDEM/backgroundDEMt0_TimeMean.nc'
    # this is the ncml for the cBathy ingegrated bathymetries
    nc_url = 'FILL IN'


    # Yaml files for my .nc files!!!!!
    global_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFti_global.yml'
    var_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFti_var.yml'

    # check the ncStep input
    d_s = None
    if 'monthly' in ncStep:
        # force the survey to start at the first of the month and end at the last of the month!!!!
        dSTR_s = dSTR_s[0:7] + '-01T00:00:00Z'
        if dSTR_e[5:7] == '12':
            dSTR_e = str(int(dSTR_e[0:4]) + 1) + '-01' + '-01T00:00:00Z'
        else:
            dSTR_e = dSTR_e[0:5] + str(int(dSTR_e[5:7]) + 1).zfill(2) + '-01T00:00:00Z'
        d_e = DT.datetime.strptime(dSTR_e, '%Y-%m-%dT%H:%M:%SZ')
        d_s = DT.datetime.strptime(dSTR_s, '%Y-%m-%dT%H:%M:%SZ')
        dList = OrderedDict(
            ((d_s + DT.timedelta(_)).strftime(r"%b-%y"), None) for _ in xrange((d_e - d_s).days + 31)).keys()
        dList = [DT.datetime.strptime(date, r"%b-%y") for date in dList]


    elif 'daily' in ncStep:
        # force the survey to start at the beginning of the day and end at the beginning of the next day
        dSTR_s = dSTR_s[0:11] + '00:00:00Z'
        dSTR_e = dSTR_e[0:11] + '00:00:00Z'
        d_e = DT.datetime.strptime(dSTR_e, '%Y-%m-%dT%H:%M:%SZ')
        d_e = d_e + DT.timedelta(days=1)
        d_s = DT.datetime.strptime(dSTR_s, '%Y-%m-%dT%H:%M:%SZ')
        dList = OrderedDict(((d_s + DT.timedelta(_)), None) for _ in xrange((d_e - d_s).days + 1)).keys()

    else:
        assert d_s is not None, 'ncStep input not recognized.  Acceptable inputs include daily or monthly.'

    # loop time
    nsteps = np.size(dList) - 1
    for tt in range(0, nsteps):

        # pull out the dates I need for this step!
        d1 = dList[tt]
        d2 = dList[tt + 1]

        # prep the data
        # get the survey data that I need
        newDict = mBATHY.getSurveyData(d1, d2)

        if newDict['elevation'] is None:
            continue
        else:
            # create my list of times for each survey?
            surveys = np.unique(newDict['surveyNumber'])
            surveyTime = np.zeros(np.shape(surveys))
            for ss in range(0, len(surveys)):
                # get the times of each survey
                idt = (newDict['surveyNumber'] == surveys[ss])
                # convert the times to a single time for each survey
                stimesS = newDict['surveyTime'][idt]
                # pull out the mean time
                stimeMS = min(stimesS) + (max(stimesS) - min(stimesS)) / 2
                # round it to nearest 12 hours.
                stimeM = sb.roundtime(stimeMS, roundTo=1 * 12 * 3600)
                timeunits = 'seconds since 1970-01-01 00:00:00'
                surveyTime[ss] = nc.date2num(stimeM, timeunits)

            # background data
            backgroundDict = {}

            try:
                # look for the .nc file that I just wrote!!!
                old_bathy = nc.Dataset(os.path.join(prev_nc_loc, prev_nc_name))
                ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units,
                                       old_bathy.variables['time'].calendar)
                # find newest time prior to this
                t_mask = (ob_times <= d_s)  # boolean true/false of time
                t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one
                backgroundDict['elevation'] = old_bathy.variables['elevation'][t_idx, :]
                backgroundDict['xFRF'] = old_bathy.variables['xFRF'][:]
                backgroundDict['yFRF'] = old_bathy.variables['yFRF'][:]
            except:
                try:
                    # look for the most up to date bathy in the ncml file....
                    old_bathy = nc.Dataset(nc_url)
                    ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units,
                                           old_bathy.variables['time'].calendar)
                    # find newest time prior to this
                    t_mask = (ob_times <= d_s)  # boolean true/false of time
                    t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one
                    backgroundDict['elevation'] = old_bathy.variables['elevation'][t_idx, :]
                    backgroundDict['xFRF'] = old_bathy.variables['xFRF'][:]
                    backgroundDict['yFRF'] = old_bathy.variables['yFRF'][:]
                except:

                    # pull the most up-to-date integrated bathymetry product (prior to the cbathy start date)
                    old_bathy = nc.Dataset(nc_b_url)
                    backgroundDict['elevation'] = old_bathy.variables['elevation'][:]
                    backgroundDict['xFRF'] = old_bathy.variables['xFRF'][:]
                    backgroundDict['yFRF'] = old_bathy.variables['yFRF'][:]
                    backgroundDict['tmBackTog'] = 1

                    """
                    # how we looking?
                    fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestFigures_CBATHY'
                    # background I made from the immediately preceding survey?
                    tempD = d1.strftime(r"%Y-%m-%d")
                    fig_name = 'backgroundDEM_' + tempD[0:4] + tempD[5:7] + '.png'
                    plt.figure()
                    plt.pcolor(backgroundDict['xFRF'], backgroundDict['yFRF'], backgroundDict['elevation'], cmap=plt.cm.jet, vmin=-13, vmax=5)
                    cbar = plt.colorbar()
                    cbar.set_label('(m)')
                    axes = plt.gca()
                    axes.set_xlim([-50, 550])
                    axes.set_ylim([-50, 1050])
                    plt.xlabel('xFRF (m)')
                    plt.ylabel('yFRF (m)')
                    plt.savefig(os.path.join(fig_loc, fig_name))
                    plt.close()
    
                   t = 1
                    """

            # go time!  this is scaleC + spline!
            out = mBATHY.makeUpdatedBATHY(backgroundDict, newDict, scalecDict=scalecDict, splineDict=splineDict)
            elevation = out['elevation']
            smoothAL = out['smoothAL']


            # check to see if any of the data has nan's.
            # If so that means there were not enough profile lines to create a survey.  Check for and remove them
            idN = np.isnan(smoothAL)
            if np.sum(idN) > 0:
                # drop all NaN stuff
                elevation = elevation[~idN, :, :]
                smoothAL = smoothAL[~idN]
                surveyTime = surveyTime[~idN]
                surveys = surveys[~idN]

            # is there anything left?
            test = np.shape(elevation)
            if test[0] < 1:
                # you got rid of everything in this file!
                continue
            else:
                # write this to the appropriate nc_file!
                # get position stuff that will be constant for all surveys!!!
                xFRFi, yFRFi = np.meshgrid(out['xFRF'], out['yFRF'])
                xFRFi_vecN = xFRFi.reshape((1, xFRFi.shape[0] * xFRFi.shape[1]))[0]
                yFRFi_vecN = yFRFi.reshape((1, yFRFi.shape[0] * yFRFi.shape[1]))[0]
                # convert FRF coords to lat/lon
                test = gp.FRF2ncsp(xFRFi_vecN, yFRFi_vecN)
                # go through stateplane to avoid FRFcoords trying to guess the input coordinate systems
                temp = gp.ncsp2LatLon(test['StateplaneE'], test['StateplaneN'])
                lat_vec = temp['lat']
                lon_vec = temp['lon']
                E_vec = test['StateplaneE']
                N_vec = test['StateplaneN']

                lat = lat_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
                lon = lon_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
                E = E_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
                N = N_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])

                xFRF = xFRFi[0, :]
                yFRF = yFRFi[:, 1]
                latitude = lat
                longitude = lon
                easting = E
                northing = N

                # write the nc_file for this month, like a boss, with greatness
                nc_dict = {}
                nc_dict['elevation'] = elevation
                nc_dict['xFRF'] = xFRF
                nc_dict['yFRF'] = yFRF
                nc_dict['latitude'] = latitude
                nc_dict['longitude'] = longitude
                nc_dict['northing'] = northing
                nc_dict['easting'] = easting
                nc_dict['time'] = np.array(surveyTime)
                nc_dict['surveyNumber'] = surveys
                nc_dict['y_smooth'] = smoothAL


                # what does the name need to be?
                tempD = d1.strftime(r"%Y-%m-%d")
                if 'monthly' in ncStep:
                    nc_name = 'CMTB-integratedBathyProduct_survey_' + tempD[0:4] + tempD[5:7] + '.nc'
                elif 'daily' in ncStep:
                    nc_name = 'CMTB-integratedBathyProduct_grid_' + tempD[0:4] + tempD[5:7] + tempD[8:10] + '.nc'

                # check to see if year directory exists
                if os.path.isdir(os.path.join(dir_loc, tempD[0:4])):
                    pass
                else:
                    # if not, make year directory
                    os.makedirs(os.path.join(dir_loc, tempD[0:4]))

                # where am I saving these nc's as I make them
                nc_loc = os.path.join(dir_loc, tempD[0:4])
                # save em like a boss?
                makenc.makenc_tiBATHY(os.path.join(nc_loc, nc_name), nc_dict, globalYaml=global_yaml, varYaml=var_yaml)

                # save this location for next time through the loop
                prev_nc_name = nc_name
                prev_nc_loc = nc_loc

                # plot some stuff
                if plot is None:
                    pass
                else:

                    fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestFigures_Transects'

                    # zoomed in pcolor plot on AOI
                    fig_name = 'transectDEM_' + tempD[0:4] + tempD[5:7] + '.png'
                    plt.figure()
                    plt.pcolor(xFRF, yFRF, out['elevation'][-1, :, :], cmap=plt.cm.jet, vmin=-13, vmax=5)
                    cbar = plt.colorbar()
                    cbar.set_label('(m)')
                    axes = plt.gca()
                    plt.xlabel('xFRF (m)')
                    plt.ylabel('yFRF (m)')
                    plt.savefig(os.path.join(fig_loc, fig_name))
                    plt.close()

                    # alongshore transect plots
                    x_loc_check1 = int(100)
                    x_loc_check2 = int(200)
                    x_loc_check3 = int(350)
                    x_check1 = np.where(xFRF == x_loc_check1)[0][0]
                    x_check2 = np.where(xFRF == x_loc_check2)[0][0]
                    x_check3 = np.where(xFRF == x_loc_check3)[0][0]

                    # plot X and Y transects from newZdiff to see if it looks correct?
                    fig_name = 'transectDEM_' + tempD[0:4] + tempD[5:7] + '_Ytrans_X' + str(
                        x_loc_check1) + '_X' + str(x_loc_check2) + '_X' + str(x_loc_check3) + '.png'

                    fig = plt.figure(figsize=(8, 9))
                    ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
                    ax1.plot(yFRF, out['elevation'][-1, :, x_check1], 'r')
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

                    ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)
                    ax2.plot(yFRF, out['elevation'][-1, :, x_check2], 'r')
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

                    ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1)
                    ax3.plot(yFRF, out['elevation'][-1, :, x_check3], 'r', label='Original')
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

                    fig.subplots_adjust(wspace=0.4, hspace=0.1)
                    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
                    fig.savefig(os.path.join(fig_loc, fig_name), dpi=300)
                    plt.close()

                    # cross-shore transect plots
                    y_loc_check1 = int(250)
                    y_loc_check2 = int(500)
                    y_loc_check3 = int(750)
                    y_check1 = np.where(yFRF == y_loc_check1)[0][0]
                    y_check2 = np.where(yFRF == y_loc_check2)[0][0]
                    y_check3 = np.where(yFRF == y_loc_check3)[0][0]
                    # plot a transect going in the cross-shore just to check it
                    fig_name = 'transectDEM_' + tempD[0:4] + tempD[5:7] + '_Xtrans_Y' + str(y_loc_check1) + '_Y' + str(
                        y_loc_check2) + '_Y' + str(
                        y_loc_check3) + '.png'

                    fig = plt.figure(figsize=(8, 9))
                    ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
                    ax1.plot(xFRF, out['elevation'][-1, y_check1, :], 'b')
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

                    ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)
                    ax2.plot(xFRF, out['elevation'][-1, y_check2, :], 'b')
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

                    ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1)
                    ax3.plot(xFRF, out['elevation'][-1, y_check3, :], 'b')
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

                    fig.subplots_adjust(wspace=0.4, hspace=0.1)
                    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
                    fig.savefig(os.path.join(fig_loc, fig_name), dpi=300)
                    plt.close()




