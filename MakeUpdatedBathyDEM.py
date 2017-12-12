import os
import netCDF4 as nc
import numpy as np
from sblib import geoprocess as gp
from sblib import sblib as sb
import makenc
from matplotlib import pyplot as plt
from bsplineFunctions import bspline_pertgrid, DLY_bspline
import datetime as DT
from scalecInterp_python.DEM_generator import DEM_generator, makeWBflow2D
import pandas as pd
from getdatatestbed import getDataFRF
from scipy.interpolate import griddata



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

    # nc_url = 'http://134.164.129.62:8080/thredds/dodsC/CMTB/grids/UpdatedBackgroundDEM/UpdatedBackgroundDEM.ncml'
    # this is just the location of the ncml for the already created UpdatedDEM

    # these together are the location of the standard background bathymetry that we started from.
    nc_b_url = 'http://134.164.129.55/thredds/dodsC/cmtb/grids/TimeMeanBackgroundDEM/backgroundDEMt0_TimeMean.nc'

    # pull the background from the THREDDS

    # Yaml files for my .nc files!!!!!
    global_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFti_global.yml'
    var_yaml = 'C:\Users\dyoung8\PycharmProjects\makebathyinterp\yamls\BATHY\FRFti_var.yml'

    # CS-array url - I just use this to get the position, not for any data
    cs_array_url = 'http://134.164.129.55/thredds/dodsC/FRF/oceanography/waves/8m-array/2017/FRF-ocean_waves_8m-array_201707.nc'
    # where do I want to save any QA/QC figures
    fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\Test Figures\QAQCfigs_transects_off20'


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
        wbysmooth = 300
        wbxsmooth = 100
    else:
        splinebctype = splineDict['splinebctype']
        lc = splineDict['lc']
        dxm = splineDict['dxm']
        dxi = splineDict['dxi']
        targetvar = splineDict['targetvar']
        wbysmooth = splineDict['wbysmooth']
        wbxsmooth = splineDict['wbxsmooth']


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
                ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units, old_bathy.variables['time'].calendar)

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
                    ob_times = nc.num2date(old_bathy.variables['time'][:], old_bathy.variables['time'].units, old_bathy.variables['time'].calendar)
                    # find newest time prior to this
                    t_mask = (ob_times <= d_s)  # boolean true/false of time
                    t_idx = np.where(t_mask)[0][-1]  # I want the MOST RECENT ONE - i.e., the last one

                    Zi = old_bathy.variables['elevation'][t_idx, :]
                    xFRFi_vec = old_bathy.variables['xFRF'][:]
                    yFRFi_vec = old_bathy.variables['yFRF'][:]

                except:
                    # load the background bathy from netDCF file if you can't get the ncml
                    # old_bathy = nc.Dataset(os.path.join(nc_b_loc, nc_b_name))
                    old_bathy = nc.Dataset(nc_b_url)
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
            surveyNumber = np.zeros(len(surveys))
            surveyTime = np.zeros(len(surveys))
            smoothAL = np.zeros(len(surveys))

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

                # temp = subgridBounds(surveyDict, gridDict, maxSpace=249)
                maxSpace = 249
                surveyFilter = True
                temp = subgridBounds2(surveyDict, gridDict, maxSpace=maxSpace, surveyFilter=surveyFilter)

                x0 = temp['x0']
                x1 = temp['x1']
                y0 = temp['y0']
                y1 = temp['y1']

                if surveyFilter is True:
                    xS0 = temp['xS0']
                    xS1 = temp['xS1']
                    yS0 = temp['yS0']
                    yS1 = temp['yS1']
                    # throw out all points in the survey that are outside of these bounds!!!!
                    test1 = np.where(dataX <= xS0, 1, 0)
                    test2 = np.where(dataX >= xS1, 1, 0)
                    test3 = np.where(dataY <= yS0, 1, 0)
                    test4 = np.where(dataY >= yS1, 1, 0)
                    test_sum = test1 + test2 + test3 + test4
                    dataXn = dataX[test_sum >= 4]
                    dataYn = dataY[test_sum >= 4]
                    dataZn = dataZ[test_sum >= 4]
                    del dataX
                    del dataY
                    del dataZ
                    dataX = dataXn
                    dataY = dataYn
                    dataZ = dataZn
                    del dataXn
                    del dataYn
                    del dataZn
                else:
                    pass

                max_spacing = temp['max_spacing']

                # if the max spacing is too high, bump up the smoothing!!
                y_smooth_u = y_smooth  # reset y_smooth if I changed it during last step
                if max_spacing is None:
                    pass
                elif 2 * max_spacing > y_smooth:
                    y_smooth_u = int(dy * round(float(2 * max_spacing) / dy))
                else:
                    pass

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
                            'msmoothy': y_smooth_u,  # smoothing length scale in y
                            'msmootht': 1,  # smoothing length scale in Time
                            'filterName': 'hanning',
                            # 'nmseitol': 0.75, # why did Spicer use 0.75?  Meg uses 0.25
                            'nmseitol': 0.25,
                            'grid_coord_check': 'FRF',
                            'grid_filename': '',  # should be none if creating background Grid!  becomes best guess grid
                            'data_coord_check': 'FRF',
                            'xFRF_s': dataX,
                            'yFRF_s': dataY,
                            'Z_s': dataZ,
                            'xFRFi_vec': xFRFi_vec,  # x-positions from the full background bathy
                            'yFRFi_vec': yFRFi_vec,  # y-positions from the full background bathy
                            'Zi': Zi,  # full background bathymetry elevations
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


                    """
                    # try a standard interpolation here and see if it gives me the same thing?
                    xFRFn2 = xFRFn.reshape((1, xFRFn.shape[0] * xFRFn.shape[1]))[0]
                    yFRFn2 = yFRFn.reshape((1, yFRFn.shape[0] * yFRFn.shape[1]))[0]
                    vecZ2 = griddata((dataX, dataY), dataZ, (xFRFn2, yFRFn2))
                    gridZ2 = vecZ2.reshape((xFRFn.shape[0], xFRFn.shape[1]))

                    # location of these figures
                    temp_fig_loc = fig_loc
                
                    # plot the Zn from stamdard gridata to compare with my DEM_generator output
                    fig_name = 'griddataDEM_' + str(surveys[tt]) + '.png'
                    plt.pcolor(xFRFn, yFRFn, gridZ2, cmap=plt.cm.jet, vmin=-13, vmax=5)
                    cbar = plt.colorbar()
                    cbar.set_label('(m)')
                    plt.scatter(dataX, dataY, marker='o', c='k', s=1, alpha=0.25, label='Transects')
                    plt.xlabel('xFRF (m)')
                    plt.ylabel('yFRF (m)')
                    plt.legend()
                    plt.savefig(os.path.join(temp_fig_loc, fig_name))
                    plt.close()
                    """


                    """
                    # Fig 4 in the TN?
                    # what does the new grid look like.
                    fig_name = 'newSurveyGrid_' + str(surveys[tt]) + '.png'
                    temp_fig_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TechNote\Figures'
                    plt.pcolor(xFRFn_vec, yFRFn_vec, Zn, cmap=plt.cm.jet, vmin=-10, vmax=3)
                    cbar = plt.colorbar()
                    cbar.set_label('Elevation ($m$)', fontsize=16)
                    plt.scatter(dataX, dataY, marker='o', c='k', s=1, alpha=0.25, label='Transects')
                    plt.xlabel('Cross-shore - $x$ ($m$)', fontsize=16)
                    plt.ylabel('Alongshore - $y$ ($m$)', fontsize=16)
                    plt.legend(prop={'size': 14})
                    plt.tick_params(axis='both', which='major', labelsize=12)
                    plt.tick_params(axis='both', which='minor', labelsize=10)
                    ax1 = plt.gca()
                    ax1.spines['right'].set_visible(False)
                    ax1.spines['top'].set_visible(False)
                    plt.axis('tight')
                    plt.tight_layout()
                    plt.savefig(os.path.join(temp_fig_loc, fig_name))
                    plt.close()
                    """

                    x1 = np.where(xFRFi_vec == min(xFRFn_vec))[0][0]
                    x2 = np.where(xFRFi_vec == max(xFRFn_vec))[0][0]
                    y1 = np.where(yFRFi_vec == min(yFRFn_vec))[0][0]
                    y2 = np.where(yFRFi_vec == max(yFRFn_vec))[0][0]

                    Zi_s = Zi[y1:y2 + 1, x1:x2 + 1]

                    # get the difference!!!!
                    Zdiff = Zn - Zi_s

                    # spline time?
                    MSEn = np.power(MSEn, 2)
                    wb = 1 - np.divide(MSEn, targetvar + MSEn)

                    wb_dict = {'x_grid': xFRFn,
                               'y_grid': yFRFn,
                               'ax': wbxsmooth / float(max(xFRFn_vec)),
                               'ay': wbysmooth / float(max(yFRFn_vec)),
                               }

                    wb_spline = makeWBflow2D(wb_dict)
                    wb = np.multiply(wb, wb_spline)

                    newZdiff = bspline_pertgrid(Zdiff, wb, splinebctype=splinebctype, lc=lc, dxm=dxm, dxi=dxi)
                    newZn = Zi_s + newZdiff

                    # Fig 5 in the TN?
                    # sample cross sections!!!!!!

                    # location of these figures
                    temp_fig_loc = fig_loc

                    try:
                        x_loc_check1 = int(100)
                        x_loc_check2 = int(200)
                        x_loc_check3 = int(350)
                        x_check1 = np.where(xFRFn_vec == x_loc_check1)[0][0]
                        x_check2 = np.where(xFRFn_vec == x_loc_check2)[0][0]
                        x_check3 = np.where(xFRFn_vec == x_loc_check3)[0][0]


                        # plot X and Y transects from newZdiff to see if it looks correct?
                        fig_name = 'backgroundDEM_' + yrs_dir + '-' + months[jj] + '-' + str(surveys[tt]) + '_Ytrans_X' + str(x_loc_check1) + '_X' + str(x_loc_check2) + '_X' + str(x_loc_check3) + '.png'

                        fig = plt.figure(figsize=(8, 9))
                        ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
                        ax1.plot(yFRFn[:, x_check1], Zn[:, x_check1], 'r', label='Original')
                        ax1.plot(yFRFn[:, x_check1], newZn[:, x_check1], 'b', label='Splined')
                        ax1.plot(yFRFn[:, x_check1], Zi_s[:, x_check1], 'k--', label='Background')
                        ax4 = ax1.twinx()
                        ax4.plot(yFRFn[:, x_check1], wb[:, x_check1], 'g--', label='Weights')
                        ax4.tick_params('y', colors='g')
                        ax4.set_ylabel('Weights', fontsize=16)
                        ax4.yaxis.label.set_color('green')
                        ax1.set_xlabel('Alongshore - $y$ ($m$)', fontsize=16)
                        ax1.set_ylabel('Elevation ($m$)', fontsize=16)
                        ax1.set_title('$X=%s$' %(str(x_loc_check1)), fontsize=16)
                        for tick in ax1.xaxis.get_major_ticks():
                            tick.label.set_fontsize(14)
                        for tick in ax1.yaxis.get_major_ticks():
                            tick.label.set_fontsize(14)
                        ax1.tick_params(labelsize=14)
                        ax1.legend()
                        ax1.text(0.10, 0.95, '(a)', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, fontsize=16)

                        ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)
                        ax2.plot(yFRFn[:, x_check2], Zn[:, x_check2], 'r', label='Original')
                        ax2.plot(yFRFn[:, x_check2], newZn[:, x_check2], 'b', label='Splined')
                        ax2.plot(yFRFn[:, x_check2], Zi_s[:, x_check2], 'k--', label='Background')
                        ax5 = ax2.twinx()
                        ax5.plot(yFRFn[:, x_check2], wb[:, x_check2], 'g--', label='Weights')
                        ax5.tick_params('y', colors='g')
                        ax5.set_ylabel('Weights', fontsize=16)
                        ax5.yaxis.label.set_color('green')
                        ax2.set_xlabel('Alongshore - $y$ ($m$)', fontsize=16)
                        ax2.set_ylabel('Elevation ($m$)', fontsize=16)
                        ax2.set_title('$X=%s$' % (str(x_loc_check2)), fontsize=16)
                        for tick in ax2.xaxis.get_major_ticks():
                            tick.label.set_fontsize(14)
                        for tick in ax2.yaxis.get_major_ticks():
                            tick.label.set_fontsize(14)
                        ax2.tick_params(labelsize=14)
                        ax2.legend()
                        ax2.text(0.10, 0.95, '(b)', horizontalalignment='left', verticalalignment='top', transform=ax2.transAxes, fontsize=16)

                        ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1)
                        ax3.plot(yFRFn[:, x_check3], Zn[:, x_check3], 'r', label='Original')
                        ax3.plot(yFRFn[:, x_check3], newZn[:, x_check3], 'b', label='Splined')
                        ax3.plot(yFRFn[:, x_check3], Zi_s[:, x_check3], 'k--', label='Background')
                        ax6 = ax3.twinx()
                        ax6.plot(yFRFn[:, x_check3], wb[:, x_check3], 'g--', label='Weights')
                        ax6.set_ylabel('Weights', fontsize=16)
                        ax6.tick_params('y', colors='g')
                        ax6.yaxis.label.set_color('green')
                        ax3.set_xlabel('Alongshore - $y$ ($m$)', fontsize=16)
                        ax3.set_ylabel('Elevation ($m$)', fontsize=16)
                        ax3.set_title('$X=%s$' % (str(x_loc_check3)), fontsize=16)
                        for tick in ax3.xaxis.get_major_ticks():
                            tick.label.set_fontsize(14)
                        for tick in ax3.yaxis.get_major_ticks():
                            tick.label.set_fontsize(14)
                        ax3.tick_params(labelsize=14)
                        ax3.legend()
                        ax3.text(0.10, 0.95, '(c)', horizontalalignment='left', verticalalignment='top', transform=ax3.transAxes, fontsize=16)

                        fig.subplots_adjust(wspace=0.4, hspace=0.1)
                        fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
                        fig.savefig(os.path.join(temp_fig_loc, fig_name), dpi=300)
                        plt.close()
                    except:
                        pass

                    try:
                        y_loc_check1 = int(250)
                        y_loc_check2 = int(500)
                        y_loc_check3 = int(750)
                        y_check1 = np.where(yFRFn_vec == y_loc_check1)[0][0]
                        y_check2 = np.where(yFRFn_vec == y_loc_check2)[0][0]
                        y_check3 = np.where(yFRFn_vec == y_loc_check3)[0][0]
                        # plot a transect going in the cross-shore just to check it
                        fig_name = 'backgroundDEM_' + yrs_dir + '-' + months[jj] + '-' + str(
                            surveys[tt]) + '_Xtrans_Y' + str(y_loc_check1) + '_Y' + str(y_loc_check2) + '_Y' + str(
                            y_loc_check3) + '.png'

                        fig = plt.figure(figsize=(8, 9))
                        ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
                        ax1.plot(xFRFn[y_check1, :], Zn[y_check1, :], 'r', label='Original')
                        ax1.plot(xFRFn[y_check1, :], newZn[y_check1, :], 'b', label='Splined')
                        ax1.plot(xFRFn[y_check1, :], Zi_s[y_check1, :], 'k--', label='Background')
                        ax4 = ax1.twinx()
                        ax4.plot(xFRFn[y_check1, :], wb[y_check1, :], 'g--', label='Weights')
                        ax4.tick_params('y', colors='g')
                        ax4.set_ylabel('Weights', fontsize=16)
                        ax4.yaxis.label.set_color('green')
                        ax1.set_xlabel('Cross-shore - $x$ ($m$)', fontsize=16)
                        ax1.set_ylabel('Elevation ($m$)', fontsize=16)
                        ax1.set_title('$Y=%s$' % (str(y_loc_check1)), fontsize=16)
                        for tick in ax1.xaxis.get_major_ticks():
                            tick.label.set_fontsize(14)
                        for tick in ax1.yaxis.get_major_ticks():
                            tick.label.set_fontsize(14)
                        ax1.tick_params(labelsize=14)
                        ax1.legend()
                        ax1.text(0.10, 0.95, '(a)', horizontalalignment='left', verticalalignment='top',
                                 transform=ax1.transAxes, fontsize=16)

                        ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)
                        ax2.plot(xFRFn[y_check2, :], Zn[y_check2, :], 'r', label='Original')
                        ax2.plot(xFRFn[y_check2, :], newZn[y_check2, :], 'b', label='Splined')
                        ax2.plot(xFRFn[y_check2, :], Zi_s[y_check2, :], 'k--', label='Background')
                        ax5 = ax2.twinx()
                        ax5.plot(xFRFn[y_check2, :], wb[y_check2, :], 'g--', label='Weights')
                        ax5.tick_params('y', colors='g')
                        ax5.set_ylabel('Weights', fontsize=16)
                        ax5.yaxis.label.set_color('green')
                        ax2.set_xlabel('Cross-shore - $x$ ($m$)', fontsize=16)
                        ax2.set_ylabel('Elevation ($m$)', fontsize=16)
                        ax2.set_title('$Y=%s$' % (str(y_loc_check2)), fontsize=16)
                        for tick in ax2.xaxis.get_major_ticks():
                            tick.label.set_fontsize(14)
                        for tick in ax2.yaxis.get_major_ticks():
                            tick.label.set_fontsize(14)
                        ax2.tick_params(labelsize=14)
                        ax2.legend()
                        ax2.text(0.10, 0.95, '(b)', horizontalalignment='left', verticalalignment='top',
                                 transform=ax2.transAxes, fontsize=16)

                        ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1)
                        ax3.plot(xFRFn[y_check3, :], Zn[y_check3, :], 'r', label='Original')
                        ax3.plot(xFRFn[y_check3, :], newZn[y_check3, :], 'b', label='Splined')
                        ax3.plot(xFRFn[y_check3, :], Zi_s[y_check3, :], 'k--', label='Background')
                        ax6 = ax3.twinx()
                        ax6.plot(xFRFn[y_check3, :], wb[y_check3, :], 'g--', label='Weights')
                        ax6.set_ylabel('Weights', fontsize=16)
                        ax6.tick_params('y', colors='g')
                        ax6.yaxis.label.set_color('green')
                        ax3.set_xlabel('Cross-shore - $x$ ($m$)', fontsize=16)
                        ax3.set_ylabel('Elevation ($m$)', fontsize=16)
                        ax3.set_title('$Y=%s$' % (str(y_loc_check3)), fontsize=16)
                        for tick in ax3.xaxis.get_major_ticks():
                            tick.label.set_fontsize(14)
                        for tick in ax3.yaxis.get_major_ticks():
                            tick.label.set_fontsize(14)
                        ax3.tick_params(labelsize=14)
                        ax3.legend()
                        ax3.text(0.10, 0.95, '(c)', horizontalalignment='left', verticalalignment='top',
                                 transform=ax3.transAxes, fontsize=16)

                        fig.subplots_adjust(wspace=0.4, hspace=0.1)
                        fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
                        fig.savefig(os.path.join(temp_fig_loc, fig_name), dpi=300)
                        plt.close()
                    except:
                        pass

                    # plot each newZn to see if it looks ok
                    fig_name = 'newDEM_' + str(surveys[tt]) + '.png'
                    plt.pcolor(xFRFn, yFRFn, newZn, cmap=plt.cm.jet, vmin=-13, vmax=5)
                    cbar = plt.colorbar()
                    cbar.set_label('(m)')
                    plt.scatter(dataX, dataY, marker='o', c='k', s=1, alpha=0.25, label='Transects')
                    plt.xlabel('xFRF (m)')
                    plt.ylabel('yFRF (m)')
                    plt.legend()
                    plt.savefig(os.path.join(temp_fig_loc, fig_name))
                    plt.close()



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
                    plt.savefig(os.path.join(temp_fig_loc, fig_name))
                    plt.close()
                    """

                # update Zi for next iteration
                del Zi
                Zi = newZi

                elevation[tt, :, :] = newZi
                surveyNumber[tt] = np.unique(survNum)[0]
                timeunits = 'seconds since 1970-01-01 00:00:00'
                surveyTime[tt] = nc.date2num(stimeM, timeunits)
                smoothAL[tt] = y_smooth_u

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
            nc_dict['easting'] = easting
            nc_dict['northing'] = northing
            # also want survey number and survey time....
            nc_dict['surveyNumber'] = surveyNumber
            nc_dict['time'] = surveyTime
            nc_dict['y_smooth'] = smoothAL

            nc_name = 'CMTB-integratedBathyProduct_survey_' + yrs_dir + months[jj] + '.nc'

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
                        wbysmooth - y-edge smoothing length scale
                        wbxsmooth - x-edge smoothing length scale

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

    nc_b_loc = 'C:\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\TestNCfiles_CBATHY'
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
        wbysmooth = 300  # y-edge smoothing scale
        wbxsmooth = 100  # x-edge smoothing scale
    else:
        splinebctype = splineDict['splinebctype']
        lc = splineDict['lc']
        dxm = splineDict['dxm']
        dxi = splineDict['dxi']
        targetvar = splineDict['targetvar']
        wbysmooth = splineDict['wbysmooth']
        wbxsmooth = splineDict['wbxsmooth']


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

                # this is telling you where to plop in your new subgrid into your larger background grid
                x1 = np.where(xFRFi_vec == min(xFRFn_vec))[0][0]
                x2 = np.where(xFRFi_vec == max(xFRFn_vec))[0][0]
                y1 = np.where(yFRFi_vec == min(yFRFn_vec))[0][0]
                y2 = np.where(yFRFi_vec == max(yFRFn_vec))[0][0]

                # this is where you pull out the overlapping region from the background
                # so you can take the difference between the original and the subgrid
                Zi_s = Zi[y1:y2 + 1, x1:x2 + 1]

                # get the difference!!!!
                Zdiff = Zn - Zi_s

                # spline time?

                # this is the spline weights that you get from the scale C routine
                # It also incorporates a target variance to bound the weights
                MSEn = np.power(MSEn, 2)
                wb = 1 - np.divide(MSEn, targetvar + MSEn)

                # do my edge spline weight stuff
                wb_dict = {'x_grid': xFRFn,
                           'y_grid': yFRFn,
                           'ax': wbxsmooth / float(max(xFRFn_vec)),
                           'ay': wbysmooth / float(max(yFRFn_vec)),
                           }

                wb_spline = makeWBflow2D(wb_dict)
                wb = np.multiply(wb, wb_spline)

                newZdiff = bspline_pertgrid(Zdiff, wb, splinebctype=splinebctype, lc=lc, dxm=dxm, dxi=dxi)
                newZn = Zi_s + newZdiff

                # get my new pretty splined grid
                newZi = Zi.copy()
                # replace the subgrid nodes with the new bathymetry
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
            # also want survey time....
            nc_dict['time'] = surveyTime

            nc_name = 'FRF-updated_bathy_dem_grids_' + yrs_dir + months[jj] + '.nc'

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

        frf_data = getDataFRF.getObs(d1, d2)
        temp = frf_data.getBathyGridcBathy(xbound=[0, 500], ybound=[0, 1000])
        # DLY note - this function is doing something funny and I have not figured out why yet.
        t = 1


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


def subgridBounds2(surveyDict, gridDict, xMax=1290, maxSpace=149, surveyFilter=False):
    """
    # this function determines the bounds of the subgrid we are going to generate from the transect data

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

    # stick this in pandas df
    columns = ['profNum']
    df = pd.DataFrame(profNum_list, columns=columns)
    df['prof_meanY'] = prof_meanY
    # sort them by mean Y
    df.sort_values(['prof_meanY'], ascending=1, inplace=True)
    # reindex
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

    if np.size(profNum_list) == 0:
        out = {}
        out['x0'] = None
        out['x1'] = None
        out['y0'] = None
        out['y1'] = None
        out['max_spacing'] = None

    else:

        # report the max spacing of my remaining lines
        max_spacing = np.nanmax(np.diff(df_sub['prof_meanY']))

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
        x0, y0 = 1.5*np.median(prof_maxX), np.max(prof_maxY)

        # can't exceed longest profile line length
        if x0 > max(prof_maxX):
            x0 = max(prof_maxX)
        else:
            pass

        x1, y1 = np.median(prof_minX), np.min(prof_minY)
        # currently using the median of the min and max X extends of each profile,
        # and just the min and max of the y-extents of all the profiles.


        # round them again because somehow this is giving us non-whole numbers
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
        if y0 >= 0:
            y0 = y0 - (y0 % dy)
        else:
            y0 = y0 - (y0 % dy) + dy

        # maxY
        if y1 >= 0:
            y1 = y1 - (y1 % dy)
        else:
            y1 = y1 - (y1 % dy) + dy

        if surveyFilter is True:
            # what do I want my survey stuff to be?
            xS0 = x0.copy()
            xS1 = x1.copy()
            yS0 = y0.copy()
            yS1 = y1.copy()

            """
            # do I want to come in some here?  i..e, throw out points that are close to the edge?
            xS0 = xS0 - 20 * dx
            xS1 = xS1 + 20 * dx
            yS0 = yS0 - 20 * dy
            yS1 = yS1 + 20 * dy
            """

            """
            # also artificially push my bounds out a little bit...
            x0 = x0 + 20 * dx
            x1 = x1 - 20 * dx
            y0 = y0 + 20 * dy
            y1 = y1 - 20 * dy
            """


        else:
            pass

        """
        # go ahead and move in some buffer spacing?
        base = 50
        buffer = int(base * round(float(2*maxSpace) / base))
        x0 = x0 + buffer
        x1 = x1 - buffer
        y0 = y0 + buffer
        y1 = y1 - buffer
        """



        # go IN one node?
        x0 = x0 - dx
        x1 = x1 + dx
        y0 = y0 - dy
        y1 = y1 + dy


        # check to see if this is past the bounds of your background DEM.
        # if so, truncate so that it does not exceed.
        if x0 > max(xFRFi_vec):
            x0 = xMax
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
        out['max_spacing'] = max_spacing

        if surveyFilter is True:
            # check to see if this is past the bounds of your background DEM.
            # if so, truncate so that it does not exceed.
            if xS0 > max(xFRFi_vec):
                xS0 = xMax - dx
            else:
                pass
            if xS1 < min(xFRFi_vec):
                xS1 = min(xFRFi_vec) + dx
            else:
                pass
            if yS0 > max(yFRFi_vec):
                yS0 = max(yFRFi_vec) - dy
            else:
                pass
            if yS1 < min(yFRFi_vec):
                yS1 = min(yFRFi_vec) + dy
            else:
                pass

            out['xS0'] = xS0
            out['xS1'] = xS1
            out['yS0'] = yS0
            out['yS1'] = yS1
        else:
            pass



    return out


def makeUpdatedBATHY(backgroundDict, newDict, scalecDict=None, splineDict=None):
    """

    :param backgroundDict: keys are:
                        elevation - 2D matrix containing the elevations at every node for
                                    whatever my background is supposed to be for this run
                        xFRF - 1D array of xFRF positions corresponding to the second dimension of elevation
                        yFRF - 1D array of yFRF positions corresponding to the first dimension of elevation
    :param newDict: keys are:
                        elevation - this will probably either be a 1D array of elevations from a survey or a 3D array of
                                    elevations where the first dimension is time and the next two are y and X, respectively.
                                    If it gets a 2D array it assumes it is a gridded bathymetry with only one time.
                        xFRF - this will either be a 1D array of x-values corresponding to the points of the survey,
                               a 1D array corresponding to the x dimension of elevation, or a 2D array that is a
                               meshgrid of the x-values corresponding to the elevations.
                               The script should be smart enough to tell which type of 1D array it is.
                        yFRF - this will either be a 1D array of y-values corresponding to the points of the survey,
                               a 1D array corresponding to the y dimension of elevation, or a 2D array that is a
                               meshgrid of the y-values corresponding to the elevations.
                               The script should be smart enough to tell which type of 1D array it is.
                        surveyNumber - these are the survey numbers of every point in the survey
                                       (so one for every point in elevation if elevation comes from survey data).
                                       This is not required if the new data is gridded
                        profileNumber - these are the profile numbers of every point in the survey
                                       (so one for every point in elevation if elevation comes from survey data).
                                       This is not required if the new data is gridded
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
    :return:
            out:
            keys are:
            elevation: will always be a 3D array, first dim either corresponds to each survey number
                       or the first dimension of the input elevation data and the second
                       two dimensions the same size as backgroundDict['evelation']
            smoothAL: 1D array corresponding to the smoothing scale used for each grid
                      (they will all be identical if using gridded data because there is no mechanism to change it)
            xFRF:   exact same as backgroundDict['xFRF']
            yFRF:   exact same as backgroundDict['xFRF']
            surveyNumber: 1D array corresponding to the survey numbers for each grid.
                          This will not exist if the input data was a grid
    """


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
        wbysmooth = 300  # y-edge smoothing scale
        wbxsmooth = 100  # x-edge smoothing scale
    else:
        splinebctype = splineDict['splinebctype']
        lc = splineDict['lc']
        dxm = splineDict['dxm']
        dxi = splineDict['dxi']
        targetvar = splineDict['targetvar']
        wbysmooth = splineDict['wbysmooth']
        wbxsmooth = splineDict['wbxsmooth']


    # load my background grid information
    Zi = backgroundDict['elevation']
    xFRFi_vec = backgroundDict['xFRF']
    yFRFi_vec = backgroundDict['yFRF']
    # read out the dx and dy of the background grid!!!
    # assume this is constant grid spacing!!!!!
    dx = abs(xFRFi_vec[1] - xFRFi_vec[0])
    dy = abs(yFRFi_vec[1] - yFRFi_vec[0])
    xFRFi, yFRFi = np.meshgrid(xFRFi_vec, yFRFi_vec)
    rows, cols = np.shape(xFRFi)

    # pull some stuff from my new data and check the dimension size
    newX = np.array(newDict['xFRF'])
    newY = np.array(newDict['yFRF'])
    newZ = np.array(newDict['elevation'])

    # check number of dimensions of dataZ
    if newZ.ndim <= 1:
        # this is survey data
        grid = 0
        survNum = newDict['surveyNumber']
        surveyList = np.unique(survNum)
        profNum = newDict['profileNumber']
        num_iter = len(surveyList)
    else:
        grid = 1
        num_iter = np.shape(newZ)[0]

    # show time

    # pre-allocate my netCDF dictionary variables here....
    elevation = np.zeros((num_iter, rows, cols))
    smoothAL = np.zeros(num_iter)
    if grid:
        pass
    else:
        surveyNumber = np.zeros(num_iter)

    for tt in range(0, num_iter):

        if grid:

            # get my stuff out
            if newZ.ndim <= 2:
                # this would mean you only handed it a 2D matrix containing ONE cbathy!!!!!!!!!!
                zV = newZ
            else:
                zV = newZ[tt, :, :]

            # were you handed a 2D array of X's and Y's or a 1D vector corresponding to that dimension?
            if np.size(zV) == np.size(newX):
                # must have been handed a meshgrid
                xV = newX
                yV = newY
            else:
                # just a 1D array, turn it into a meshgrid
                xV, yV = np.meshgrid(newX, newY)


            # what are my subgrid bounds?
            x0 = np.max(xV)
            y0 = np.max(yV)
            x1 = np.min(xV)
            y1 = np.min(yV)

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

            #reshape it to pass to DEM generator
            dataX, dataY, dataZ = [], [], []
            dataX = np.reshape(xV, (np.shape(xV)[0] * np.shape(xV)[1], 1)).flatten()
            dataY = np.reshape(yV, (np.shape(yV)[0] * np.shape(yV)[1], 1)).flatten()
            dataZ = np.reshape(zV, (np.shape(zV)[0] * np.shape(zV)[1], 1)).flatten()

            # specify y_smoothing
            y_smooth_u = y_smooth

        else:

            # get the times of each survey
            ids = (survNum == surveyList[tt])

            # pull out this stuf!!!!!!!!
            dataX, dataY, dataZ = [], [], []
            dataX = newX[ids]
            dataY = newY[ids]
            dataZ = newZ[ids]

            # what are my subgrid bounds?
            surveyDict = {}
            surveyDict['dataX'] = dataX
            surveyDict['dataY'] = dataY
            surveyDict['profNum'] = profNum[ids]

            gridDict = {}
            gridDict['dx'] = dx
            gridDict['dy'] = dy
            gridDict['xFRFi_vec'] = xFRFi_vec
            gridDict['yFRFi_vec'] = yFRFi_vec

            # temp = subgridBounds(surveyDict, gridDict, maxSpace=249)
            maxSpace = 249
            surveyFilter = True
            temp = subgridBounds2(surveyDict, gridDict, maxSpace=maxSpace, surveyFilter=surveyFilter)

            x0 = temp['x0']
            x1 = temp['x1']
            y0 = temp['y0']
            y1 = temp['y1']

            if surveyFilter is True:
                xS0 = temp['xS0']
                xS1 = temp['xS1']
                yS0 = temp['yS0']
                yS1 = temp['yS1']
                # throw out all points in the survey that are outside of these bounds!!!!
                test1 = np.where(dataX <= xS0, 1, 0)
                test2 = np.where(dataX >= xS1, 1, 0)
                test3 = np.where(dataY <= yS0, 1, 0)
                test4 = np.where(dataY >= yS1, 1, 0)
                test_sum = test1 + test2 + test3 + test4
                dataXn = dataX[test_sum >= 4]
                dataYn = dataY[test_sum >= 4]
                dataZn = dataZ[test_sum >= 4]
                del dataX
                del dataY
                del dataZ
                dataX = dataXn
                dataY = dataYn
                dataZ = dataZn
                del dataXn
                del dataYn
                del dataZn
            else:
                pass

            max_spacing = temp['max_spacing']

            # if the max spacing is too high, bump up the smoothing!!
            y_smooth_u = y_smooth  # reset y_smooth if I changed it during last step
            if max_spacing is None:
                pass
            elif 2 * max_spacing > y_smooth:
                y_smooth_u = int(dy * round(float(2 * max_spacing) / dy))
            else:
                pass

            del temp

        # ok, now it should be on to DEM generator at this point?
        # I think the only difference after this will be if I return the survey number or not?
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
                    'msmoothy': y_smooth_u,  # smoothing length scale in y
                    'msmootht': 1,  # smoothing length scale in Time
                    'filterName': 'hanning',
                    # 'nmseitol': 0.75, # why did Spicer use 0.75?  Meg uses 0.25
                    'nmseitol': 0.25,
                    'grid_coord_check': 'FRF',
                    'grid_filename': '',  # should be none if creating background Grid!  becomes best guess grid
                    'data_coord_check': 'FRF',
                    'xFRF_s': dataX,
                    'yFRF_s': dataY,
                    'Z_s': dataZ,
                    'xFRFi_vec': xFRFi_vec,  # x-positions from the full background bathy
                    'yFRFi_vec': yFRFi_vec,  # y-positions from the full background bathy
                    'Zi': Zi,  # full background bathymetry elevations
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

            # where does this subgrid fit in my larger background grid?
            x1 = np.where(xFRFi_vec == min(xFRFn_vec))[0][0]
            x2 = np.where(xFRFi_vec == max(xFRFn_vec))[0][0]
            y1 = np.where(yFRFi_vec == min(yFRFn_vec))[0][0]
            y2 = np.where(yFRFi_vec == max(yFRFn_vec))[0][0]

            Zi_s = Zi[y1:y2 + 1, x1:x2 + 1]

            # get the difference!!!!
            Zdiff = Zn - Zi_s

            # spline time?
            MSEn = np.power(MSEn, 2)
            wb = 1 - np.divide(MSEn, targetvar + MSEn)

            wb_dict = {'x_grid': xFRFn,
                       'y_grid': yFRFn,
                       'ax': wbxsmooth / float(max(xFRFn_vec)),
                       'ay': wbysmooth / float(max(yFRFn_vec)),
                       }

            wb_spline = makeWBflow2D(wb_dict)
            wb = np.multiply(wb, wb_spline)

            newZdiff = bspline_pertgrid(Zdiff, wb, splinebctype=splinebctype, lc=lc, dxm=dxm, dxi=dxi)
            newZn = Zi_s + newZdiff

            # get my new pretty splined grid
            newZi = Zi.copy()
            newZi[y1:y2 + 1, x1:x2 + 1] = newZn


        # update Zi for next iteration
        del Zi
        Zi = newZi

        # go ahead and stack this stuff in my new variables I am building
        elevation[tt, :, :] = newZi
        smoothAL[tt] = y_smooth_u
        if grid:
            pass
        else:
            surveyNumber[tt] = np.unique(survNum)[0]

    # go ahead and return my dictionary
    out = {}
    out['elevation'] = elevation
    out['smoothAL'] = smoothAL
    out['xFRF'] = xFRFi_vec
    out['yFRF'] = yFRFi_vec

    if grid:
        pass
    else:
        out['surveyNumber'] = surveyNumber
    return out














































