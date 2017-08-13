import os
import netCDF4 as nc
import numpy as np
from sblib import geoprocess as gp
from sblib import sblib as sb
import makenc
from matplotlib import pyplot as plt
from bsplineFunctions import bspline_pertgrid, DLY_bspline
import datetime as DT
from scalecInterp_python.DEM_generator import DEM_generator
import pandas as pd
from getdatatestbed import getDataFRF



def makeUpdatedBATHY_transects(dSTR_s, dSTR_e, dir_loc, scalecDict=None, splineDict=None, plot=None):
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

                        if not specified it will default to:
                        splinebctype = 10
                        lc = 4
                        dxm = 2
                        dxi = 1
                        targetvar = 0.45

    :param plot: toggle for turning plot on or off.  Anything besides None will cause it to plot

    :return: writes out the .nc files for the new DEMs in the appropriate year directories
                also creates and saves plots of the updated DEM at the end of each month if desired

    # basic steps:
    1. figures out how many years and months you have and loops over them
    2. pulls all surveys out for each month - if the average of the first and last time in the survey
    (rounded to the nearest 12 hours) is not in the month, then it will throw that survey out - it
    will be picked up in the preceeding or trailing month
    3. pulls most recent bathy that occurs right before the first survey
        -checks the previously written .nc file first, then the .ncml, then goes back to the original background DEM
    4. Loops over the surveys
        pulls them out, converts them to a subgrid using splinecInterp,
        then splines that subgrid back into the background DEM using the bsplineFunctions
    5. stacks all surveys for the month into one .nc file and writes it,
        also creates QA/QC plots of the last survey in the month if desired
    """

    #HARD CODED VARIABLES!!!
    filelist = ['http://134.164.129.55/thredds/dodsC/FRF/geomorphology/elevationTransects/survey/surveyTransects.ncml']
    # this is just the location of the ncml for the transects!!!!!

    nc_url = 'http://134.164.129.62:8080/thredds/dodsC/CMTB/grids/UpdatedBackgroundDEM/UpdatedBackgroundDEM.ncml'
    # this is just the location of the ncml for the already created UpdatedDEM

    nc_b_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestNCfiles'
    nc_b_name = 'backgroundDEMt0_TimeMean.nc'
    # these together are the location of the standard background bathymetry that we started from.

    # Yaml files for my .nc files!!!!!
    global_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFti_global.yml'
    var_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFti_var.yml'

    # CS-array url - I just use this to get the position, not for any data
    cs_array_url = 'http://134.164.129.55/thredds/dodsC/FRF/oceanography/waves/8m-array/2017/FRF-ocean_waves_8m-array_201707.nc'
    # where do I want to save any QA/QC figures
    fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures\QAQCfigs_transects'


    #check scalecDict and splineDict
    if scalecDict is None:
        x_smooth = 100  # scale c interp x-direction smoothing
        y_smooth = 200  # scale c interp y-direction smoothing
    else:
        x_smooth = scalecDict['x_smooth']  # scale c interp x-direction smoothing
        y_smooth = scalecDict['y_smooth']  # scale c interp y-direction smoothing

    if splineDict is None:
        splinebctype = 10
        lc = 4
        dxm = 2
        dxi = 1
        targetvar = 0.45
    else:
        splinebctype = splineDict['splinebctype']
        lc = splineDict['lc']
        dxm = splineDict['dxm']
        dxi = splineDict['dxi']
        targetvar = splineDict['targetvar']


    # force the survey to start at the first of the month and end at the last of the month!!!!
    dSTR_s = dSTR_s[0:7] + '-01T00:00:00Z'
    if dSTR_e[5:7] == '12':
        dSTR_e = str(int(dSTR_e[0:4]) + 1) + '-01' + '-01T00:00:00Z'
    else:
        dSTR_e = dSTR_e[0:5] + str(int(dSTR_e[5:7]) + 1).zfill(2) + '-01T00:00:00Z'

    d_s = DT.datetime.strptime(dSTR_s, '%Y-%m-%dT%H:%M:%SZ')

    # how many months, years between my start and end times?
    year_end = dSTR_e[0:4]
    month_end = dSTR_e[5:7]
    year_start = dSTR_s[0:4]
    month_start = dSTR_s[5:7]
    # how many years between them?
    num_yrs = int(year_end) - int(year_start)

    # show time....
    for ii in range(0, num_yrs):

        # make year directories!!!
        yrs_dir = str(int(year_start) + int(ii))
        # check to see if year directory exists
        if os.path.isdir(os.path.join(dir_loc, yrs_dir)):
            pass
        else:
            # if not, make year directory
            os.makedirs(os.path.join(dir_loc, yrs_dir))

        # where am I saving these nc's as I make them
        nc_loc = os.path.join(dir_loc, yrs_dir)

        # make a list of the months in this year
        if (yrs_dir == year_start) and (yrs_dir == year_end):
            # just one year?
            num_months = int(month_end) - int(month_start)
            months = [str(int(month_start) + int(jj)).zfill(2) for jj in range(0, num_months + 1)]
        elif (yrs_dir == year_start):
            # this is the start year
            num_months = int('12') - int(month_start)
            months = [str(int(month_start) + int(jj)).zfill(2) for jj in range(0, num_months + 1)]
        elif (yrs_dir == year_end):
            # this is the end year
            num_months = int(month_end) - int('01')
            months = [str(int('01') + int(jj)).zfill(2) for jj in range(0, num_months + 1)]
        else:
            # I need all months
            num_months = int('12') - int('01')
            months = [str(int('01') + int(jj)).zfill(2) for jj in range(0, num_months + 1)]

        # ok, now to make my nc files, I just need to go through and find all surveys that fall in these months
        bathy = nc.Dataset(filelist[0])
        # pull down all the times....
        times = nc.num2date(bathy.variables['time'][:], bathy.variables['time'].units, bathy.variables['time'].calendar)

        all_surveys = bathy.variables['surveyNumber'][:]

        for jj in range(0, len(months)):
            # pull out the beginning and end time associated with this month
            d1STR = yrs_dir + '-' + months[jj] + '-01T00:00:00Z'
            d1 = DT.datetime.strptime(d1STR, '%Y-%m-%dT%H:%M:%SZ')
            if int(months[jj]) == 12:
                d2STR = str(int(yrs_dir) + 1) + '-' + '01' + '-01T00:00:00Z'
                d2 = DT.datetime.strptime(d2STR, '%Y-%m-%dT%H:%M:%SZ')
            else:
                d2STR = yrs_dir + '-' + str(int(months[jj]) + 1) + '-01T00:00:00Z'
                d2 = DT.datetime.strptime(d2STR, '%Y-%m-%dT%H:%M:%SZ')

            # find some stuff here...
            mask = (times >= d1) & (times < d2)  # boolean true/false of time
            idx = np.where(mask)[0]

            # what surveys are in this range?
            surveys = np.unique(bathy.variables['surveyNumber'][idx])

            # if there are no surveys here, then skip the rest of this loop...
            if len(surveys) < 1:
                print('No surveys found for ' + yrs_dir + '-' + months[jj])
                continue
            else:
                pass

            # otherwise..., check to see the times of all the surveys...?
            for tt in range(0, len(surveys)):
                ids = (all_surveys == surveys[tt])
                surv_times = times[ids]
                # pull out the mean time
                surv_timeM = surv_times[0] + (surv_times[-1] - surv_times[0]) / 2
                # round it to nearest 12 hours.
                surv_timeM = sb.roundtime(surv_timeM, roundTo=1 * 12 * 3600)

                # if the rounded time IS in the month, great
                if (surv_timeM >= d1) and (surv_timeM < d2):
                    pass
                else:
                    # if not set it to a fill value
                    surveys[tt] == -1000
            # drop all the surveys that we decided are not going to go into this monthly file!
            surveys = surveys[surveys >= 0]

            # SEARCH FOR MOST RECENT BATHY HERE!!!
            try:
                # look for the .nc file that I just wrote!!!
                old_bathy = nc.Dataset(os.path.join(prev_nc_loc, prev_nc_name))
                ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units,
                                       old_bathy.variables['time'].calendar)

                # find newest time prior to this
                t_mask = (ob_times <= d1)  # boolean true/false of time
                t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one

                Zi = old_bathy.variables['elevation'][t_idx, :]
                xFRFi_vec = old_bathy.variables['xFRF'][:]
                yFRFi_vec = old_bathy.variables['yFRF'][:]
            except:
                try:
                    # look for the most up to date bathy in the ncml file....
                    old_bathy = nc.Dataset(nc_url)
                    ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units,
                                           old_bathy.variables['time'].calendar)
                    # find newest time prior to this
                    t_mask = (ob_times <= d_s)  # boolean true/false of time
                    t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one

                    Zi = old_bathy.variables['elevation'][t_idx, :]
                    xFRFi_vec = old_bathy.variables['xFRF'][:]
                    yFRFi_vec = old_bathy.variables['yFRF'][:]

                except:
                    # load the background bathy from netDCF file if you can't get the ncml
                    old_bathy = nc.Dataset(os.path.join(nc_b_loc, nc_b_name))
                    Zi = old_bathy.variables['elevation'][:]
                    xFRFi_vec = old_bathy.variables['xFRF'][:]
                    yFRFi_vec = old_bathy.variables['yFRF'][:]


            # read out the dx and dy of the background grid!!!
            # assume this is constant grid spacing!!!!!
            dx = abs(xFRFi_vec[1] - xFRFi_vec[0])
            dy = abs(yFRFi_vec[1] - yFRFi_vec[0])

            xFRFi, yFRFi = np.meshgrid(xFRFi_vec, yFRFi_vec)
            rows, cols = np.shape(xFRFi)

            # pre-allocate my netCDF dictionary variables here....
            elevation = np.zeros((len(surveys), rows, cols))
            xFRF = np.zeros(cols)
            yFRF = np.zeros(rows)
            latitude = np.zeros((rows, cols))
            longitude = np.zeros((rows, cols))
            surveyNumber = np.zeros(len(surveys))
            surveyTime = np.zeros(len(surveys))

            # ok, now that I have the list of the surveys I am going to keep.....
            for tt in range(0, len(surveys)):
                """
                # plot the initial bathymetry...
                fig_name = 'backgroundDEM_' + str(surveys[tt]) + '_orig' + '.png'
                plt.pcolor(xFRFi_vec, yFRFi_vec, Zi, cmap=plt.cm.jet, vmin=-13, vmax=5)
                cbar = plt.colorbar()
                cbar.set_label('(m)')
                plt.xlabel('xFRF (m)')
                plt.ylabel('yFRF (m)')
                plt.savefig(os.path.join(fig_loc, fig_name))
                plt.close()
                """

                # get the times of each survey
                ids = (all_surveys == surveys[tt])

                # pull out this NC stuf!!!!!!!!
                dataX, dataY, dataZ = [], [], []
                dataX = bathy['xFRF'][ids]
                dataY = bathy['yFRF'][ids]
                dataZ = bathy['elevation'][ids]
                profNum = bathy['profileNumber'][ids]
                survNum = bathy['surveyNumber'][ids]
                stimes = nc.num2date(bathy.variables['time'][ids], bathy.variables['time'].units, bathy.variables['time'].calendar)
                # pull out the mean time
                stimeM = min(stimes) + (max(stimes) - min(stimes)) / 2
                # round it to nearest 12 hours.
                stimeM = sb.roundtime(stimeM, roundTo=1 * 12 * 3600)

                assert len(np.unique(survNum)) == 1, 'MakeUpdatedBathyDEM error: You have pulled down more than one survey number!'
                assert isinstance(dataZ, np.ndarray), 'MakeUpdatedBathyDEM error: Script only handles np.ndarrays for the transect data at this time!'


                # build my new bathymetry from the FRF transect files

                #what are my subgrid bounds?
                surveyDict = {}
                surveyDict['dataX'] = dataX
                surveyDict['dataY'] = dataY
                surveyDict['profNum'] = profNum

                gridDict = {}
                gridDict['dx'] = dx
                gridDict['dy'] = dy
                gridDict['xFRFi_vec'] = xFRFi_vec
                gridDict['yFRFi_vec'] = yFRFi_vec

                temp = subgridBounds(surveyDict, gridDict, maxSpace=249)
                x0 = temp['x0']
                x1 = temp['x1']
                y0 = temp['y0']
                y1 = temp['y1']
                del temp

                # if you wound up throwing out this survey!!!
                if x0 is None:
                    newZi = Zi

                else:
                    print np.unique(survNum)
                    dict = {'x0': x0,  # gp.FRFcoord(x0, y0)['Lon'],  # -75.47218285,
                            'y0': y0,  # gp.FRFcoord(x0, y0)['Lat'],  #  36.17560399,
                            'x1': x1,  # gp.FRFcoord(x1, y1)['Lon'],  # -75.75004989,
                            'y1': y1,  # gp.FRFcoord(x1, y1)['Lat'],  #  36.19666112,
                            'lambdaX': dx,
                            # grid spacing in x  -  Here is where CMS would hand array of variable grid spacing
                            'lambdaY': dy,  # grid spacing in y
                            'msmoothx': x_smooth,  # smoothing length scale in x
                            'msmoothy': y_smooth,  # smoothing length scale in y
                            'msmootht': 1,  # smoothing length scale in Time
                            'filterName': 'hanning',
                            'nmseitol': 0.75,
                            'grid_coord_check': 'FRF',
                            'grid_filename': '',  # should be none if creating background Grid!  becomes best guess grid
                            'data_coord_check': 'FRF',
                            'xFRF_s': dataX,
                            'yFRF_s': dataY,
                            'Z_s': dataZ,
                            }

                    out = DEM_generator(dict)

                    # read some stuff from this dict like a boss
                    Zn = out['Zi']
                    MSEn = out['MSEi']
                    MSRn = out['MSRi']
                    NMSEn = out['NMSEi']
                    xFRFn_vec = out['x_out']
                    yFRFn_vec = out['y_out']

                    # make my the mesh for the new subgrid
                    xFRFn, yFRFn = np.meshgrid(xFRFn_vec, yFRFn_vec)

                    x1 = np.where(xFRFi_vec == min(xFRFn_vec))[0][0]
                    x2 = np.where(xFRFi_vec == max(xFRFn_vec))[0][0]
                    y1 = np.where(yFRFi_vec == min(yFRFn_vec))[0][0]
                    y2 = np.where(yFRFi_vec == max(yFRFn_vec))[0][0]

                    Zi_s = Zi[y1:y2 + 1, x1:x2 + 1]

                    # get the difference!!!!
                    Zdiff = Zn - Zi_s

                    # spline time?
                    wb = 1 - np.divide(MSEn, targetvar + MSEn)


                    newZdiff = DLY_bspline(Zdiff, splinebctype=10, off=20, lc=1)
                    #newZdiff = bspline_pertgrid(Zdiff, wb, splinebctype=splinebctype, lc=lc, dxm=dxm, dxi=dxi)


                    newZn = Zi_s + newZdiff

                    """
                    # plot X and Y transects from newZdiff to see if it looks correct?
                    # check near the midpoint
                    x_check = int(0.5*len(xFRFn_vec))
                    y_check = int(0.5 * len(yFRFn_vec))
                    fig_name = 'backgroundDEM_' + yrs_dir + '-' + months[jj] + '-' + str(surveys[tt]) + '_Xtrans' + '.png'
                    plt.plot(xFRFn[y_check, :], Zn[y_check, :], 'r', label='Original')
                    plt.plot(xFRFn[y_check, :], newZn[y_check, :], 'b', label='Splined')
                    plt.plot(xFRFn[y_check, :], Zi_s[y_check, :], 'k--', label='Background')
                    plt.xlabel('xFRF')
                    plt.ylabel('Z (m)')
                    plt.legend()
                    plt.savefig(os.path.join(os.path.join(fig_loc[0:85], 'SplineChecks'), fig_name))
                    plt.close()

                    fig_name = 'backgroundDEM_' + yrs_dir + '-' + months[jj] + '-' + str(surveys[tt]) + '_Ytrans' + '.png'
                    plt.plot(yFRFn[:, x_check], Zn[:, x_check], 'r', label='Original')
                    plt.plot(yFRFn[:, x_check], newZn[:, x_check], 'b', label='Splined')
                    plt.plot(yFRFn[:, x_check], Zi_s[:, x_check], 'k--', label='Background')
                    plt.xlabel('yFRF')
                    plt.ylabel('Z (m)')
                    plt.legend()
                    plt.savefig(os.path.join(os.path.join(fig_loc[0:85], 'SplineChecks'), fig_name))
                    plt.close()
                    """

                    # get my new pretty splined grid
                    newZi = Zi.copy()
                    newZi[y1:y2 + 1, x1:x2 + 1] = newZn

                    """
                    # plot each newZi to see if it looks ok
                    fig_name = 'backgroundDEM_' + str(surveys[tt]) + '.png'
                    plt.pcolor(xFRFi_vec, yFRFi_vec, newZi, cmap=plt.cm.jet, vmin=-13, vmax=5)
                    cbar = plt.colorbar()
                    cbar.set_label('(m)')
                    plt.scatter(dataX, dataY, marker='o', c='k', s=1, alpha=0.25, label='Transects')
                    plt.xlabel('xFRF (m)')
                    plt.ylabel('yFRF (m)')
                    plt.legend()
                    plt.savefig(os.path.join(fig_loc, fig_name))
                    plt.close()
                    """


                # update Zi for next iteration
                del Zi
                Zi = newZi

                elevation[tt, :, :] = newZi
                surveyNumber[tt] = np.unique(survNum)[0]
                timeunits = 'seconds since 1970-01-01 00:00:00'
                surveyTime[tt] = nc.date2num(stimeM, timeunits)
                # timeM is the mean time between the first and last time of the survey rounded to the nearest 12 hours
                # this is going to be the date and time of the survey to the closest noon.
                # remember it needs to be in seconds since 1970

            # get position stuff that will be constant for all surveys!!!
            xFRFi_vecN = xFRFi.reshape((1, xFRFi.shape[0] * xFRFi.shape[1]))[0]
            yFRFi_vecN = yFRFi.reshape((1, yFRFi.shape[0] * yFRFi.shape[1]))[0]
            # convert FRF coords to lat/lon
            test = gp.FRF2ncsp(xFRFi_vecN, yFRFi_vecN)
            # go through stateplane to avoid FRFcoords trying to guess the input coordinate systems
            temp = gp.ncsp2LatLon(test['StateplaneE'], test['StateplaneN'])
            lat_vec = temp['lat']
            lon_vec = temp['lon']

            lat = lat_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
            lon = lon_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])

            xFRF = xFRFi[0, :]
            yFRF = yFRFi[:, 1]
            latitude = lat
            longitude = lon

            # write the nc_file for this month, like a boss, with greatness
            nc_dict = {}
            nc_dict['elevation'] = elevation
            nc_dict['xFRF'] = xFRF
            nc_dict['yFRF'] = yFRF
            nc_dict['latitude'] = latitude
            nc_dict['longitude'] = longitude
            # also want survey number and survey time....
            nc_dict['surveyNumber'] = surveyNumber
            nc_dict['time'] = surveyTime

            nc_name = 'backgroundDEM_' + months[jj] + '.nc'

            # save this location for next time through the loop
            prev_nc_name = nc_name
            prev_nc_loc = nc_loc

            makenc.makenc_tiBATHY(os.path.join(nc_loc, nc_name), nc_dict, globalYaml=global_yaml, varYaml=var_yaml)

            # make my QA/QC plot
            # does the user want to make them?
            if plot is None:
                pass
            else:
                # where is the cross shore array?
                test = nc.Dataset(cs_array_url)
                Lat = test['latitude'][:]
                Lon = test['longitude'][:]
                # convert to FRF
                temp = gp.FRFcoord(Lon, Lat)
                CSarray_X = temp['xFRF']
                CSarray_Y = temp['yFRF']

                fig_name = 'backgroundDEM_' + yrs_dir + '-' + months[jj] + '.png'

                plt.figure()
                plt.pcolor(xFRF, yFRF, elevation[-1, :, :], cmap=plt.cm.jet, vmin=-13, vmax=5)
                cbar = plt.colorbar()
                cbar.set_label('(m)')
                # plt.scatter(dataX, dataY, marker='o', c='k', s=1, alpha=0.25, label='Transects')
                plt.plot(CSarray_X, CSarray_Y, 'rX', label='8m-array')
                plt.xlabel('xFRF (m)')
                plt.ylabel('yFRF (m)')
                plt.legend()
                plt.savefig(os.path.join(fig_loc, fig_name))
                plt.close()


def subgridBounds(surveyDict, gridDict, xMax=1000, maxSpace=149):
    """
    # this function determines the bounds of the subgrid we are going to generate from the trasect data

    # basic logic is that first we are only going to use the largest block of consecutive profile lines
    for which the mean yFRF position does not exceed maxSpace.  Then, of those that remain,
    the x bounds are the medians of the minimum x and maximum x of each profile line.
    The y-bounds are the min and max y positions observed
    # these numbers are always rounded down to the nearest dx (or dy) - rounded up for negative numbers...

    :param surveyDict: keys:
                        dataX - x data from the survey
                        dataY - y data from the survey
                        profNum - profile numbers from the survey
    :param gridDict: keys:
                        dx - dx of background grid
                        dy - dy of background grid
                        xFRFi_vec - xFRF positions of your background grid
                        yFRFi_vec - yFRF positions of your background grid

    :param xMax: maximum allowable x for the transects (i.e., the max x of the subgrid may never exceed this value)
                    default is xFRF = 1000 m
    :param maxSpace: maximum allowable spacing between the profile lines (in the alongshore direction)
                    default is 149 m in the yFRF direction BUT I THINK THIS IS TOO SMALL!!!!!
    :return:
        dictionary containing the coordinates of
    """

    dataX = surveyDict['dataX']
    dataY = surveyDict['dataY']
    profNum = surveyDict['profNum']

    dx = gridDict['dx']
    dy = gridDict['dy']
    xFRFi_vec = gridDict['xFRFi_vec']
    yFRFi_vec = gridDict['yFRFi_vec']

    # divide my survey up into the survey lines!!!
    profNum_list = np.unique(profNum)
    prof_minX = np.zeros(np.shape(profNum_list))
    prof_maxX = np.zeros(np.shape(profNum_list))
    prof_minY = np.zeros(np.shape(profNum_list))
    prof_maxY = np.zeros(np.shape(profNum_list))
    prof_meanY = np.zeros(np.shape(profNum_list))

    for ss in range(0, len(profNum_list)):
        # get mean y-values of each line
        Yprof = dataY[np.where(profNum == profNum_list[ss])]
        prof_meanY[ss] = np.mean(Yprof)

    #stick this in pandas df
    columns = ['profNum']
    df = pd.DataFrame(profNum_list, columns=columns)
    df['prof_meanY'] = prof_meanY
    # sort them by mean Y
    df.sort_values(['prof_meanY'], ascending=1, inplace=True)
    #reindex
    df.reset_index(drop=True, inplace=True)

    # figure out the differences!!!!!
    df['prof_meanY_su'] = df['prof_meanY'].shift(1)
    df['diff1'] = df['prof_meanY'] - df['prof_meanY_su']
    del df['prof_meanY_su']

    df['prof_meanY_sd'] = df['prof_meanY'].shift(-1)
    df['diff2'] = df['prof_meanY'] - df['prof_meanY_sd']
    del df['prof_meanY_sd']

    # ok, this tricky little piece of code lets you look both directions
    df['check1'] = np.where((abs(df['diff1']) <= maxSpace), 1, 0)
    df['check2'] = np.where((abs(df['diff2']) <= maxSpace), 1, 0)
    df['sum_check'] = df['check1'] + df['check2']
    del df['check1']
    del df['check2']
    # i want to find the biggest piece that works in both directions.
    # then I''l tack on the leading and trailing profiles at the end! if they exist that is...
    df['check'] = np.where((df['sum_check'] == 2), 1, 0)
    del df['sum_check']



    #this is some clever script I found on the interwebs, it divides the script into blocks of consecutive
    #jetty in's and jetty out sections
    df['block'] = (df.check.shift(1) != df.check).astype(int).cumsum()
    df['Counts'] = df.groupby(['block'])['check'].transform('count')

    #pull out the largest continuous block of inside the jetty and outside the jetty data!
    df_sub = df[df['check'] == 1]
    df_sub = df_sub[df_sub['Counts'] == df_sub['Counts'].max()]
    #if I have more than one block (because I have blocks of the same length) then just take the first one
    if len(df_sub['block'].unique()) > 1:
        df_sub = df_sub[df_sub['block'] == df_sub['block'].min()]
    else:
        pass

    # also include the line numbers immediately above and below this as well!!! (if they exist)
    try:
        df_sub = df_sub.append(df.iloc[int(min(df_sub.index) - 1)])
    except:
        pass

    try:
        df_sub = df_sub.append(df.iloc[int(max(df_sub.index) + 1)])
    except:
        pass

    del profNum_list
    del prof_minX
    del prof_maxX
    del prof_minY
    del prof_maxY
    del prof_meanY

    # sort them by mean Y
    df_sub.sort_values(['prof_meanY'], ascending=1, inplace=True)
    df_sub.reset_index(drop=True, inplace=True)
    profNum_list = df_sub['profNum'].apply(np.array)

    prof_minX = np.zeros(np.shape(profNum_list))
    prof_maxX = np.zeros(np.shape(profNum_list))
    prof_minY = np.zeros(np.shape(profNum_list))
    prof_maxY = np.zeros(np.shape(profNum_list))
    prof_meanY = np.zeros(np.shape(profNum_list))

    if np.size(profNum_list) == 0:
        out = {}
        out['x0'] = None
        out['x1'] = None
        out['y0'] = None
        out['y1'] = None

    else:

        for ss in range(0, len(profNum_list)):
            # pull out all x-values corresponding to this profNum
            Xprof = dataX[np.where(profNum == profNum_list[ss])]
            Yprof = dataY[np.where(profNum == profNum_list[ss])]
            prof_minX[ss] = min(Xprof)
            prof_maxX[ss] = max(Xprof)
            prof_minY[ss] = min(Yprof)
            prof_maxY[ss] = max(Yprof)

            # round it to nearest dx or dy
            # minX
            if prof_minX[ss] >= 0:
                prof_minX[ss] = prof_minX[ss] - (prof_minX[ss] % dx)
            else:
                prof_minX[ss] = prof_minX[ss] - (prof_minX[ss] % dx) + dx

            # maxX
            if prof_maxX[ss] >= 0:
                prof_maxX[ss] = prof_maxX[ss] - (prof_maxX[ss] % dx)
            else:
                prof_maxX[ss] = prof_maxX[ss] - (prof_maxX[ss] % dx) + dx

            # minY
            if prof_minY[ss] >= 0:
                prof_minY[ss] = prof_minY[ss] - (prof_minY[ss] % dy)
            else:
                prof_minY[ss] = prof_minY[ss] - (prof_minY[ss] % dy) + dy

            # maxY
            if prof_maxY[ss] >= 0:
                prof_maxY[ss] = prof_maxY[ss] - (prof_maxY[ss] % dy)
            else:
                prof_maxY[ss] = prof_maxY[ss] - (prof_maxY[ss] % dy) + dy


        #do not allow any prof_maxX to exceed xMax
        prof_maxX[prof_maxX > xMax] = xMax

        # ok, I am going to force the DEM generator function to always go to the grid specified by these bounds!!
        # if you want to hand it a best guess grid, i.e., 'grid_filename' make SURE it has these bounds!!!!!!
        # or just don't hand it grid_filename....
        x0, y0 = np.median(prof_maxX), np.max(prof_maxY)
        x1, y1 = np.median(prof_minX), np.min(prof_minY)
        # currently using the median of the min and max X extends of each profile,
        # and just the min and max of the y-extents of all the profiles.

        # check to see if this is past the bounds of your background DEM.
        # if so, truncate so that it does not exceed.
        if x0 > max(xFRFi_vec):
            x0 = max(xFRFi_vec)
        else:
            pass
        if x1 < min(xFRFi_vec):
            x1 = min(xFRFi_vec)
        else:
            pass
        if y0 > max(yFRFi_vec):
            y0 = max(yFRFi_vec)
        else:
            pass
        if y1 < min(yFRFi_vec):
            y1 = min(yFRFi_vec)
        else:
            pass

        out = {}
        out['x0'] = x0
        out['x1'] = x1
        out['y0'] = y0
        out['y1'] = y1

    return out


def makeUpdatedBATHY_grid(dSTR_s, dSTR_e, dir_loc, ncml_url, scalecDict=None, splineDict=None, plot=None):
    """

    :param dSTR_s: string that determines the start date of the times of the surveys you want to use to update the DEM
                    format is  dSTR_s = '2013-01-04T00:00:00Z'
                    no matter what you put here, it will always round it down to the beginning of the month
    :param dSTR_e: string that determines the end date of the times of the surveys you want to use to update the DEM
                    format is dSTR_e = '2014-12-22T23:59:59Z'
                    no matter what you put here, it will always round it up to the end of the month
    :param dir_loc: place where you want to save the .nc files that get written
                    the function will make the year directories inside of this location on its own.
    :param url: the url of the ncml file you are going to use for your grids to integrate into the background
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
                        if not specified it will default to:
                        splinebctype = 10
                        lc = 4
                        dxm = 2
                        dxi = 1
                        targetvar = 0.45

    :param plot: toggle for turning plot on or off.  Anything besides None will cause it to plot

    :return: writes out the .nc files for the new DEMs in the appropriate year directories
                also creates and saves plots of the updated DEM at the end of each month if desired

    # basic steps:
    1. figures out how many years and months you have and loops over them
    2. pulls all grids out for each month
    3. pulls most recent bathy grid that occurs right before the first survey
        -checks the previously written .nc file first, then the .ncml, then goes back to the original background DEM
    4. Loops over the surveys
        pulls them out, converts them to a subgrid using splinecInterp,
        then splines that subgrid back into the background DEM using the bsplineFunctions
    5. stacks all surveys for the month into one .nc file and writes it,
        also creates QA/QC plots of the last survey in the month if desired
    """

    # HARD CODED VARIABLES!!!
    # background_url = 'http://134.164.129.62:8080/thredds/dodsC/CMTB/grids/UpdatedBackgroundDEM/UpdatedBackgroundDEM.ncml'
    # this is just the location of the ncml for the already created UpdatedDEM

    nc_b_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestNCfiles_gridded'
    nc_b_name = 'backgroundDEMt0_TimeMean.nc'
    # these together are the location of the standard background bathymetry that we started from.

    # Yaml files for my .nc files!!!!!
    global_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFti_global.yml'
    var_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFti_grid_var.yml'

    # CS-array url - I just use this to get the position, not for any data
    cs_array_url = 'http://134.164.129.55/thredds/dodsC/FRF/oceanography/waves/8m-array/2017/FRF-ocean_waves_8m-array_201707.nc'
    # where do I want to save any QA/QC figures
    fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures\QAQCfigs_gridded'

    # check scalecDict and splineDict
    if scalecDict is None:
        x_smooth = 100  # scale c interp x-direction smoothing
        y_smooth = 200  # scale c interp y-direction smoothing
    else:
        x_smooth = scalecDict['x_smooth']  # scale c interp x-direction smoothing
        y_smooth = scalecDict['y_smooth']  # scale c interp y-direction smoothing

    if splineDict is None:
        splinebctype = 10
        lc = 4
        dxm = 2
        dxi = 1
        targetvar = 0.45
    else:
        splinebctype = splineDict['splinebctype']
        lc = splineDict['lc']
        dxm = splineDict['dxm']
        dxi = splineDict['dxi']
        targetvar = splineDict['targetvar']

    # force the survey to start at the first of the month and end at the last of the month!!!!
    dSTR_s = dSTR_s[0:7] + '-01T00:00:00Z'
    if dSTR_e[5:7] == '12':
        dSTR_e = str(int(dSTR_e[0:4]) + 1) + '-01' + '-01T00:00:00Z'
    else:
        dSTR_e = dSTR_e[0:5] + str(int(dSTR_e[5:7]) + 1).zfill(2) + '-01T00:00:00Z'

    d_s = DT.datetime.strptime(dSTR_s, '%Y-%m-%dT%H:%M:%SZ')

    # how many months, years between my start and end times?
    year_end = dSTR_e[0:4]
    month_end = dSTR_e[5:7]
    year_start = dSTR_s[0:4]
    month_start = dSTR_s[5:7]
    # how many years between them?
    num_yrs = int(year_end) - int(year_start)

    # show time....
    for ii in range(0, num_yrs+1):

        # make year directories!!!
        yrs_dir = str(int(year_start) + int(ii))
        # check to see if year directory exists
        if os.path.isdir(os.path.join(dir_loc, yrs_dir)):
            pass
        else:
            # if not, make year directory
            os.makedirs(os.path.join(dir_loc, yrs_dir))

        # where am I saving these nc's as I make them
        nc_loc = os.path.join(dir_loc, yrs_dir)

        # make a list of the months in this year
        if (yrs_dir == year_start) and (yrs_dir == year_end):
            # just one year?
            num_months = int(month_end) - int(month_start)
            months = [str(int(month_start) + int(jj)).zfill(2) for jj in range(0, num_months + 1)]
        elif (yrs_dir == year_start):
            # this is the start year
            num_months = int('12') - int(month_start)
            months = [str(int(month_start) + int(jj)).zfill(2) for jj in range(0, num_months + 1)]
        elif (yrs_dir == year_end):
            # this is the end year
            num_months = int(month_end) - int('01')
            months = [str(int('01') + int(jj)).zfill(2) for jj in range(0, num_months + 1)]
        else:
            # I need all months
            num_months = int('12') - int('01')
            months = [str(int('01') + int(jj)).zfill(2) for jj in range(0, num_months + 1)]


        for jj in range(0, len(months)):
            # pull out the beginning and end time associated with this month
            d1STR = yrs_dir + '-' + months[jj] + '-01T00:00:00Z'
            if int(months[jj]) == 12:
                d2STR = str(int(yrs_dir) + 1) + '-' + '01' + '-01T00:00:00Z'
            else:
                d2STR = yrs_dir + '-' + str(int(months[jj]) + 1) + '-01T00:00:00Z'

            # get the grids that fall within these dates
            bathy_dict = getGridded(ncml_url, d1STR, d2STR)

            # if there are no surveys here, then skip the rest of this loop...
            if bathy_dict is None:
                print('No surveys found for ' + yrs_dir + '-' + months[jj])
                continue
            else:
                pass

            # SEARCH FOR MOST RECENT BATHY HERE!!!
            try:
                # look for the .nc file that I just wrote!!!
                old_bathy = nc.Dataset(os.path.join(prev_nc_loc, prev_nc_name))
                ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units,
                                       old_bathy.variables['time'].calendar)

                # find newest time prior to this
                d1 = DT.datetime.strptime(d1STR, '%Y-%m-%dT%H:%M:%SZ')
                t_mask = (ob_times <= d1)  # boolean true/false of time
                t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one

                Zi = old_bathy.variables['elevation'][t_idx, :]
                xFRFi_vec = old_bathy.variables['xFRF'][:]
                yFRFi_vec = old_bathy.variables['yFRF'][:]
            except:
                try:
                    # look for the most up to date bathy in the ncml file....
                    old_bathy = nc.Dataset(background_url)
                    ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units, old_bathy.variables['time'].calendar)
                    # find newest time prior to this
                    t_mask = (ob_times <= d_s)  # boolean true/false of time
                    t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one

                    Zi = old_bathy.variables['elevation'][t_idx, :]
                    xFRFi_vec = old_bathy.variables['xFRF'][:]
                    yFRFi_vec = old_bathy.variables['yFRF'][:]

                except:
                    # load the background bathy from netDCF file if you can't get the ncml
                    old_bathy = nc.Dataset(os.path.join(nc_b_loc, nc_b_name))
                    Zi = old_bathy.variables['elevation'][:]
                    xFRFi_vec = old_bathy.variables['xFRF'][:]
                    yFRFi_vec = old_bathy.variables['yFRF'][:]

            # read out the dx and dy of the background grid!!!
            # assume this is constant grid spacing!!!!!
            dx = abs(xFRFi_vec[1] - xFRFi_vec[0])
            dy = abs(yFRFi_vec[1] - yFRFi_vec[0])

            xFRFi, yFRFi = np.meshgrid(xFRFi_vec, yFRFi_vec)
            rows, cols = np.shape(xFRFi)

            # pre-allocate my netCDF dictionary variables here....
            iter_rows = np.shape(bathy_dict['elevation'])[0]
            elevation = np.zeros((iter_rows, rows, cols))
            xFRF = np.zeros(cols)
            yFRF = np.zeros(rows)
            latitude = np.zeros((rows, cols))
            longitude = np.zeros((rows, cols))
            surveyTime = np.zeros(iter_rows)

            for tt in range(0, iter_rows):

                # pull out this NC stuf!!!!!!!!
                dataX, dataY = [], []
                dataZ = []
                xV, yV = [], []
                xV = bathy_dict['xFRF']
                yV = bathy_dict['yFRF']
                dataZ = bathy_dict['elevation'][tt][:]

                # check to see if this is a stupid masked array....
                if isinstance(dataZ, np.ma.MaskedArray):
                    if np.sum(dataZ.mask) == 0:
                        pass
                    else:
                        xV = xV[~np.all(dataZ.mask, axis=0)]
                        yV = yV[~np.all(dataZ.mask, axis=1)]
                        dataZ = dataZ[~np.all(dataZ.mask, axis=1), :]
                        dataZ = dataZ[:, ~np.all(dataZ.mask, axis=0)]
                        if dataZ.ndim == 2:
                            dataZ = np.ma.expand_dims(dataZ, axis=0)
                        else:
                            pass
                        if len(np.shape(dataZ)) > 2:
                            dataZ = dataZ[0, :, :]
                        else:
                            pass
                else:
                    pass

                dataX, dataY = np.meshgrid(xV, yV)

                stimeM = bathy_dict['time'][tt]


                # what does this input grid look like
                fig_name = 'InputGrid_' + yrs_dir + '-' + months[jj] + '_' + str(tt + 1) + '.png'
                plt.figure()
                plt.pcolor(xV, yV, dataZ, cmap=plt.cm.jet, vmin=-13, vmax=5)
                cbar = plt.colorbar()
                cbar.set_label('(m)')
                plt.xlabel('xFRF (m)')
                plt.ylabel('yFRF (m)')
                plt.legend()
                plt.savefig(os.path.join(fig_loc, fig_name))
                plt.close()


                assert isinstance(dataZ, np.ndarray), 'MakeUpdatedBathyDEM error: Script only handles np.ndarrays for the grid data at this time!'

                # build my new bathymetry from the FRF transect files

                # what are my subgrid bounds?
                x0 = np.max(dataX)
                y0 = np.max(dataY)
                x1 = np.min(dataX)
                y1 = np.min(dataY)

                # round it to nearest dx or dy
                # minX
                if x1 >= 0:
                    x1 = x1 - (x1 % dx)
                else:
                    x1 = x1 - (x1 % dx) + dx

                # maxX
                if x0 >= 0:
                    x0 = x0 - (x0 % dx)
                else:
                    x0 = x0 - (x0 % dx) + dx

                # minY
                if y1 >= 0:
                    y1 = y1 - (y1 % dy)
                else:
                    y1 = y1 - (y1 % dy) + dy

                # maxY
                if y0 >= 0:
                    y0 = y0 - (y0 % dy)
                else:
                    y0 = y0 - (y0 % dy) + dy

                # make sure they are inside the bounds of my bigger grid
                # if so, truncate so that it does not exceed.
                if x0 > max(xFRFi_vec):
                    x0 = max(xFRFi_vec)
                else:
                    pass
                if x1 < min(xFRFi_vec):
                    x1 = min(xFRFi_vec)
                else:
                    pass
                if y0 > max(yFRFi_vec):
                    y0 = max(yFRFi_vec)
                else:
                    pass
                if y1 < min(yFRFi_vec):
                    y1 = min(yFRFi_vec)
                else:
                    pass

                print stimeM
                dict = {'x0': x0,  # gp.FRFcoord(x0, y0)['Lon'],  # -75.47218285,
                        'y0': y0,  # gp.FRFcoord(x0, y0)['Lat'],  #  36.17560399,
                        'x1': x1,  # gp.FRFcoord(x1, y1)['Lon'],  # -75.75004989,
                        'y1': y1,  # gp.FRFcoord(x1, y1)['Lat'],  #  36.19666112,
                        'lambdaX': dx,
                        # grid spacing in x  -  Here is where CMS would hand array of variable grid spacing
                        'lambdaY': dy,  # grid spacing in y
                        'msmoothx': x_smooth,  # smoothing length scale in x
                        'msmoothy': y_smooth,  # smoothing length scale in y
                        'msmootht': 1,  # smoothing length scale in Time
                        'filterName': 'hanning',
                        'nmseitol': 0.75,
                        'grid_coord_check': 'FRF',
                        'grid_filename': '',  # should be none if creating background Grid!  becomes best guess grid
                        'data_coord_check': 'FRF',
                        'xFRF_s': np.reshape(dataX, (np.shape(dataX)[0] * np.shape(dataX)[1], 1)).flatten(),
                        'yFRF_s': np.reshape(dataY, (np.shape(dataY)[0] * np.shape(dataY)[1], 1)).flatten(),
                        'Z_s': np.reshape(dataZ, (np.shape(dataZ)[0] * np.shape(dataZ)[1], 1)).flatten(),
                        }

                out = DEM_generator(dict)

                # read some stuff from this dict like a boss
                Zn = out['Zi']
                MSEn = out['MSEi']
                MSRn = out['MSRi']
                NMSEn = out['NMSEi']
                xFRFn_vec = out['x_out']
                yFRFn_vec = out['y_out']

                # make my the mesh for the new subgrid
                xFRFn, yFRFn = np.meshgrid(xFRFn_vec, yFRFn_vec)

                x1 = np.where(xFRFi_vec == min(xFRFn_vec))[0][0]
                x2 = np.where(xFRFi_vec == max(xFRFn_vec))[0][0]
                y1 = np.where(yFRFi_vec == min(yFRFn_vec))[0][0]
                y2 = np.where(yFRFi_vec == max(yFRFn_vec))[0][0]

                Zi_s = Zi[y1:y2 + 1, x1:x2 + 1]

                # get the difference!!!!
                Zdiff = Zn - Zi_s

                # spline time?
                wb = 1 - np.divide(MSEn, targetvar + MSEn)

                newZdiff = DLY_bspline(Zdiff, splinebctype=10, off=20, lc=1)
                #newZdiff = bspline_pertgrid(Zdiff, wb, splinebctype=splinebctype, lc=lc, dxm=dxm, dxi=dxi)
                newZn = Zi_s + newZdiff

                # get my new pretty splined grid
                newZi = Zi.copy()
                newZi[y1:y2 + 1, x1:x2 + 1] = newZn

                # update Zi for next iteration
                del Zi
                Zi = newZi

                elevation[tt, :, :] = newZi
                timeunits = 'seconds since 1970-01-01 00:00:00'
                surveyTime[tt] = nc.date2num(stimeM, timeunits)
                # timeM is the mean time between the first and last time of the survey rounded to the nearest 12 hours
                # this is going to be the date and time of the survey to the closest noon.
                # remember it needs to be in seconds since 1970

            # get position stuff that will be constant for all surveys!!!
            xFRFi_vecN = xFRFi.reshape((1, xFRFi.shape[0] * xFRFi.shape[1]))[0]
            yFRFi_vecN = yFRFi.reshape((1, yFRFi.shape[0] * yFRFi.shape[1]))[0]
            # convert FRF coords to lat/lon
            test = gp.FRF2ncsp(xFRFi_vecN, yFRFi_vecN)
            # go through stateplane to avoid FRFcoords trying to guess the input coordinate systems
            temp = gp.ncsp2LatLon(test['StateplaneE'], test['StateplaneN'])
            lat_vec = temp['lat']
            lon_vec = temp['lon']

            lat = lat_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])
            lon = lon_vec.reshape(xFRFi.shape[0], xFRFi.shape[1])

            xFRF = xFRFi[0, :]
            yFRF = yFRFi[:, 1]
            latitude = lat
            longitude = lon

            # write the nc_file for this month, like a boss, with greatness
            nc_dict = {}
            nc_dict['elevation'] = elevation
            nc_dict['xFRF'] = xFRF
            nc_dict['yFRF'] = yFRF
            nc_dict['latitude'] = latitude
            nc_dict['longitude'] = longitude
            # also want survey time....
            nc_dict['time'] = surveyTime

            nc_name = 'backgroundDEM_' + months[jj] + '.nc'

            # save this location for next time through the loop
            prev_nc_name = nc_name
            prev_nc_loc = nc_loc

            makenc.makenc_tiBATHY(os.path.join(nc_loc, nc_name), nc_dict, globalYaml=global_yaml, varYaml=var_yaml)

            # make my QA/QC plot
            # does the user want to make them?
            if plot is None:
                pass
            else:
                # where is the cross shore array?
                test = nc.Dataset(cs_array_url)
                Lat = test['latitude'][:]
                Lon = test['longitude'][:]
                # convert to FRF
                temp = gp.FRFcoord(Lon, Lat)
                CSarray_X = temp['xFRF']
                CSarray_Y = temp['yFRF']

                fig_name = 'backgroundDEM_' + yrs_dir + '-' + months[jj] + '.png'

                plt.figure()
                plt.pcolor(xFRF, yFRF, elevation[-1, :, :], cmap=plt.cm.jet, vmin=-13, vmax=5)
                cbar = plt.colorbar()
                cbar.set_label('(m)')
                plt.plot(CSarray_X, CSarray_Y, 'rX', label='8m-array')
                plt.xlabel('xFRF (m)')
                plt.ylabel('yFRF (m)')
                plt.legend()
                plt.savefig(os.path.join(fig_loc, fig_name))
                plt.close()


def getGridded(ncml_url, d1, d2):

    """
    :param ncml_url: this is the url of the ncml for the type of data we are going to use.  this script will look
                      for key phrases in this url in order to tell what the dictionary keys will be!!!
    :param d1: this is the date STRING of the start time you want to look for!!!!!
    :param d2: this is the date STRING of the end time you want to look for!!!!!
    :return: this function will return a dictionary with standardized keys of the bathy information (x, y, z, times)
                for the particular gridded product, first index of each variable will be time
                unless it is time invariant (i.e., xFRF. yFRF)
                keys:
                xFRF
                yFRF
                elevation
                time - as a DATETIME
    """

    # convert those strings to datetimes!!!!!
    d1 = DT.datetime.strptime(d1, '%Y-%m-%dT%H:%M:%SZ')
    d2 = DT.datetime.strptime(d2, '%Y-%m-%dT%H:%M:%SZ')

    # check to see what product I am supposed to be using?
    type = None
    if 'cbathy' in ncml_url:
        type = 1
        # this is a cbathy product
    elif 'lidar' in ncml_url:
        type = 2
        # this is a lidar product
    elif 'survey' in ncml_url:
        type = 0
        # this is a regular gridded product
    else:
        pass

    assert type is not None, 'getGridded error: the ncml_url provided does not match any known type!'

    if type == 1:
        print 'cBathy grid functionality not implemented'
    elif type == 0:


        # error testing for Spicer
        frf_data = getDataFRF.getObs(d1, d2)
        temp = frf_data.getBathyGridFromNC(method=0)


        bathy = nc.Dataset(ncml_url)
        # pull down all the times....
        times = nc.num2date(bathy.variables['time'][:], bathy.variables['time'].units, bathy.variables['time'].calendar)
        mask = (times >= d1) & (times < d2)  # boolean true/false of time
        idx = np.where(mask)[0]

        # check to see if I have any surveys
        if len(idx) == 0:
            out = None
            print 'No grids found between %s and %s' %(d1, d2)
        else:
            out = {}
            out['xFRF'] = bathy['xFRF'][:]
            out['yFRF'] = bathy['yFRF'][:]
            out['elevation'] = bathy['elevation'][idx][:]
            out['time'] = nc.num2date(bathy.variables['time'][idx], bathy.variables['time'].units, bathy.variables['time'].calendar)

    elif type == 2:
        print 'LiDAR grid functionality not implemented'
    else:
        pass

    return out





















